#!/usr/bin/env python3
"""
LoTUS-BF backend: Location-or-Term Unified Search for Brain Functions.

A Flask + PostgreSQL/PostGIS API over the Neurosynth database, plus (optionally) the built React frontend.
It is a drop-in replacement for the course template's app.py: the same endpoints and JSON shapes, with
correct boolean logic and the problems found while validating the template fixed (see README, "Backend").

  GET /help                          service description (JSON)
  GET /health                        database + data sanity check
  GET /terms                         the term vocabulary
  GET /terms/<term>                  terms that co-occur with <term> (Jaccard similarity)
  GET /query/<q>/studies             studies matching a mixed query      ?r= &limit= &offset=
  GET /query/<q>/locations           activation peaks of those studies   ?r= &limit= &offset=
  GET /query/<q>/nii                 meta-analytic map (gzipped NIfTI)   ?r= &fwhm= &kernel= &voxel=

Query language (identical to the frontend, src/lib/query.js):

    query   := or
    or      := and ( OR and )*
    and     := unary ( [AND] unary )*          AND may be omitted:  "a NOT b" == "a AND NOT b"
    unary   := NOT unary | primary
    primary := "(" or ")" | [x,y,z] | "exact phrase" | word | word*

  * [x,y,z]  studies reporting a peak closer than r mm (default 6) to that MNI point
  * words    PostgreSQL full-text search (English stemming) over the study metadata
  * Each atom becomes a set of study ids; AND / OR / NOT are INTERSECT / UNION / EXCEPT, so terms and
    locations can be mixed freely, with parentheses.

Configuration (environment variables): DB_URL (or DATABASE_URL, required), LOTUS_SCHEMA (default "ns"),
LOTUS_STATIC_DIR (default ./dist), LOTUS_STATEMENT_TIMEOUT_MS (default 20000), PORT (python app.py only).
"""
from __future__ import annotations

import gzip
import logging
import math
import os
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import nibabel as nib
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from scipy.ndimage import gaussian_filter
from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError
from werkzeug.exceptions import HTTPException

__version__ = "2.0.0"
log = logging.getLogger("lotus-bf")

# =============================================================================
# Configuration
# =============================================================================
SCHEMA = os.getenv("LOTUS_SCHEMA", "ns")
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", SCHEMA):  # it is spliced into SQL, so it must be a plain identifier
    raise RuntimeError("LOTUS_SCHEMA must be a plain SQL identifier")

TERM_PREFIX = "terms_abstract_tfidf__"  # prefix the course loader leaves on stored term names
STATEMENT_TIMEOUT_MS = int(os.getenv("LOTUS_STATEMENT_TIMEOUT_MS", "20000"))
DIST_DIR = Path(os.getenv("LOTUS_STATIC_DIR", Path(__file__).resolve().parent / "dist"))

DEFAULT_RADIUS = 6.0
MAX_RADIUS = 50.0
MAX_LIMIT = 10_000
MAX_QUERY_CHARS = 1000
MAX_ATOMS = 40  # terms + coordinates in one query
MAX_POINTS = 12  # coordinates in one query (each one is a spatial join)

# The MNI152 2 mm template grid the maps are written on.
TEMPLATE_SHAPE = (91, 109, 91)
TEMPLATE_AFFINE = np.array(
    [[-2.0, 0.0, 0.0, 90.0], [0.0, 2.0, 0.0, -126.0], [0.0, 0.0, 2.0, -72.0], [0.0, 0.0, 0.0, 1.0]]
)
VOXEL_MM = 2.0
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


class ApiError(Exception):
    """An error that is the caller's fault (or a known operational state); rendered as JSON."""

    def __init__(self, message: str, status: int = 400, **extra: Any):
        super().__init__(message)
        self.message, self.status, self.extra = message, status, extra


# =============================================================================
# Database engine (created lazily, so importing this module never needs the database)
# =============================================================================
_engine = None
_engine_lock = threading.Lock()


def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            url = os.getenv("DB_URL") or os.getenv("DATABASE_URL")
            if not url:
                raise ApiError("The server is not configured: set the DB_URL environment variable.", 503)
            if url.startswith("postgres://"):  # old scheme still handed out by some hosts
                url = "postgresql://" + url[len("postgres://"):]
            if url.startswith("postgresql://"):  # name the driver (SQLAlchemy >= 2.1 would default to psycopg 3)
                url = "postgresql+psycopg2://" + url[len("postgresql://"):]
            _engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=5,
                connect_args={"connect_timeout": 10},  # fail fast when the database host is unreachable
            )

            @event.listens_for(_engine, "connect")
            def _limit_statement_time(dbapi_connection, _record):
                """No runaway queries. A post-connect SET (not a startup option) works behind poolers too; failure is non-fatal."""
                try:
                    with dbapi_connection.cursor() as cur:
                        cur.execute(f"SET statement_timeout = {int(STATEMENT_TIMEOUT_MS)}")
                    dbapi_connection.commit()
                except Exception:  # noqa: BLE001
                    dbapi_connection.rollback()
                    log.warning("could not set statement_timeout; continuing without it")

        return _engine


# =============================================================================
# Query language: tokenizer -> parser -> AST        (pure Python, no database; unit-tested)
# =============================================================================
_NUMBER = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)")
_OPERATORS = {"AND", "OR", "NOT"}


class QueryError(ApiError):
    def __init__(self, message: str, start: int = 0, end: int = 0):
        super().__init__(message, 400, position=[start, end])


def tokenize(source: str) -> list[dict]:
    """Split a query into tokens. Raises QueryError on malformed pieces ([1,2, unclosed quotes, ...)."""
    tokens: list[dict] = []
    n, i = len(source), 0
    while i < n:
        ch = source[i]
        if ch.isspace():
            i += 1
        elif ch in "()":
            tokens.append({"type": "lparen" if ch == "(" else "rparen", "text": ch, "start": i, "end": i + 1})
            i += 1
        elif ch == "[":
            close = source.find("]", i)
            if close == -1:
                raise QueryError('Missing "]" to close the coordinate.', i, n)
            raw = source[i : close + 1]
            parts = [p.strip() for p in raw[1:-1].split(",")]
            if len(parts) != 3 or not all(_NUMBER.fullmatch(p) for p in parts):
                raise QueryError(f"A coordinate needs three numbers, like [-22, -4, 18] (got {raw}).", i, close + 1)
            x, y, z = (float(p) for p in parts)
            if not all(math.isfinite(v) and abs(v) < 1e4 for v in (x, y, z)):
                raise QueryError(f"Coordinate out of range: {raw}.", i, close + 1)
            tokens.append({"type": "loc", "text": raw, "x": x, "y": y, "z": z, "start": i, "end": close + 1})
            i = close + 1
        elif ch == "]":
            raise QueryError('Unexpected "]".', i, i + 1)
        elif ch in "\"'":
            j = i + 1
            while j < n and source[j] != ch:
                j += 2 if source[j] == "\\" else 1
            if j >= n:
                raise QueryError(f"Missing closing {ch} for the phrase.", i, n)
            value = re.sub(r"\\(.)", r"\1", source[i + 1 : j]).strip()
            if not value:
                raise QueryError("Empty phrase.", i, j + 1)
            tokens.append({"type": "phrase", "text": source[i : j + 1], "value": value, "start": i, "end": j + 1})
            i = j + 1
        else:
            j = i
            while j < n and not (source[j].isspace() or source[j] in "()[]"):
                j += 1
            word = source[i:j]
            if word.upper() in _OPERATORS:
                tokens.append({"type": "op", "op": word.upper(), "text": word, "start": i, "end": j})
            else:
                tokens.append({"type": "term", "text": word, "value": word, "start": i, "end": j})
            i = j
    return tokens


_STARTS_OPERAND = {"term", "phrase", "loc", "lparen"}


def _starts_operand(tok: Optional[dict]) -> bool:
    return tok is not None and (tok["type"] in _STARTS_OPERAND or (tok["type"] == "op" and tok["op"] == "NOT"))


def parse(tokens: list[dict]) -> dict:
    """Recursive-descent parser. NOT binds tighter than AND, which binds tighter than OR; AND may be implicit."""
    pos = 0

    def peek() -> Optional[dict]:
        return tokens[pos] if pos < len(tokens) else None

    def is_op(tok: Optional[dict], name: str) -> bool:
        return tok is not None and tok["type"] == "op" and tok["op"] == name

    def parse_or() -> dict:
        nonlocal pos
        kids = [parse_and()]
        while is_op(peek(), "OR"):
            pos += 1
            kids.append(parse_and())
        return kids[0] if len(kids) == 1 else {"type": "or", "children": kids}

    def parse_and() -> dict:
        nonlocal pos
        kids = [parse_unary()]
        while True:
            tok = peek()
            if is_op(tok, "AND"):
                pos += 1
                kids.append(parse_unary())
            elif _starts_operand(tok):
                kids.append(parse_unary())
            else:
                break
        return kids[0] if len(kids) == 1 else {"type": "and", "children": kids}

    def parse_unary() -> dict:
        nonlocal pos
        if is_op(peek(), "NOT"):
            pos += 1
            return {"type": "not", "child": parse_unary()}
        return parse_primary()

    def parse_primary() -> dict:
        nonlocal pos
        tok = peek()
        if tok is None:
            last = tokens[-1] if tokens else {"start": 0, "end": 0}
            raise QueryError("The query ends too early: expected a term or a coordinate.", last["start"], last["end"])
        if tok["type"] == "lparen":
            pos += 1
            if peek() is not None and peek()["type"] == "rparen":
                raise QueryError("Empty parentheses.", tok["start"], tok["end"])
            inner = parse_or()
            if peek() is None or peek()["type"] != "rparen":
                raise QueryError('Missing ")" to close this group.', tok["start"], tok["end"])
            pos += 1
            return inner
        if tok["type"] == "loc":
            pos += 1
            return {"type": "loc", "x": tok["x"], "y": tok["y"], "z": tok["z"]}
        if tok["type"] in ("term", "phrase"):
            pos += 1
            return {"type": "term", "value": tok["value"], "phrase": tok["type"] == "phrase"}
        if tok["type"] == "rparen":
            raise QueryError('Unexpected ")".', tok["start"], tok["end"])
        where = "after" if pos == 0 else "on both sides of"
        raise QueryError(f'"{tok["op"]}" needs a term or a coordinate {where} it.', tok["start"], tok["end"])

    ast = parse_or()
    if pos < len(tokens):
        tok = tokens[pos]
        raise QueryError('Unexpected ")".' if tok["type"] == "rparen" else f'Unexpected "{tok["text"]}".', tok["start"], tok["end"])
    return ast


def parse_query(source: str) -> dict:
    """Parse and validate a query string -> AST (raises QueryError)."""
    source = (source or "").strip()
    if not source:
        raise QueryError("The query is empty. Give at least a term or a [x,y,z] coordinate.")
    if len(source) > MAX_QUERY_CHARS:
        raise QueryError(f"The query is too long (more than {MAX_QUERY_CHARS} characters).")
    ast = parse(tokenize(source))
    atoms = list(_atoms(ast))
    if len(atoms) > MAX_ATOMS:
        raise QueryError(f"The query has too many terms and coordinates (maximum {MAX_ATOMS}).")
    if sum(1 for a in atoms if a["type"] == "loc") > MAX_POINTS:
        raise QueryError(f"The query has too many coordinates (maximum {MAX_POINTS}).")
    return ast


def _atoms(node: dict):
    if node["type"] in ("term", "loc"):
        yield node
    elif node["type"] == "not":
        yield from _atoms(node["child"])
    else:
        for child in node["children"]:
            yield from _atoms(child)


# =============================================================================
# From AST to SQL: every atom is a set of study ids, AND/OR/NOT are INTERSECT/UNION/EXCEPT
# =============================================================================
def _term_kind(node: dict) -> tuple[str, str]:
    """('plain'|'phrase'|'prefix', normalised text) for a term atom."""
    value = node["value"].strip()
    if node["phrase"]:
        return "phrase", value
    if value.endswith("*"):
        stem = re.sub(r"[^\w-]", "", value.rstrip("*"))  # keep tsquery syntax characters out of the prefix
        return "prefix", stem
    return "plain", value


def _tsquery_sql(kind: str, param: str) -> str:
    if kind == "phrase":
        return f"phraseto_tsquery('english', :{param})"
    if kind == "prefix":
        return f"to_tsquery('english', :{param})"
    return f"plainto_tsquery('english', :{param})"


def _tsquery_param(kind: str, value: str) -> str:
    return f"'{value}':*" if kind == "prefix" else value


_searchable_cache: dict[tuple[str, str], bool] = {}


def _searchable(conn, kind: str, value: str) -> bool:
    """False for stop words ('of', 'the'): the full-text parser drops them, so they must not constrain the search."""
    if kind == "prefix" and not value:
        return False
    key = (kind, value.lower())
    cached = _searchable_cache.get(key)
    if cached is None:
        n = conn.execute(text(f"SELECT numnode({_tsquery_sql(kind, 'v')})"), {"v": _tsquery_param(kind, value)}).scalar()
        cached = bool(n)
        if len(_searchable_cache) < 5000:  # bounded: user input must not be able to grow it forever
            _searchable_cache[key] = cached
    return cached


def _prune(node: Optional[dict], ignored: set[int]) -> Optional[dict]:
    """Drop atoms flagged in `ignored` (by id()) and the operators that only existed to join them."""
    if node is None:
        return None
    t = node["type"]
    if t == "term":
        return None if id(node) in ignored else node
    if t == "loc":
        return node
    if t == "not":
        child = _prune(node["child"], ignored)
        return None if child is None else {"type": "not", "child": child}
    kids = [k for k in (_prune(c, ignored) for c in node["children"]) if k is not None]
    if not kids:
        return None
    return kids[0] if len(kids) == 1 else {"type": t, "children": kids}


def compile_matched(conn, ast: dict, radius: float) -> tuple[str, dict, dict]:
    """
    Build `WITH ... matched AS (<set of study ids>)`.
    Returns (sql_prefix, params, info) where info = {"locations": bool, "terms": bool, "ignored": [words]}.
    """
    terms: list[tuple[str, str]] = []
    points: list[tuple[float, float, float]] = []
    ignored_words: list[str] = []
    ignored_ids: set[int] = set()
    for atom in _atoms(ast):
        if atom["type"] == "term":
            kind, value = _term_kind(atom)
            if not _searchable(conn, kind, value):
                ignored_ids.add(id(atom))
                ignored_words.append(atom["value"])
    pruned = _prune(ast, ignored_ids)
    if pruned is None:
        raise QueryError(
            "Nothing searchable: the query only contains very common words (" + ", ".join(ignored_words) + ") that are ignored by the full-text search."
        )

    needs_universe = False

    def sql_of(node: dict) -> str:
        nonlocal needs_universe
        t = node["type"]
        if t == "term":
            terms.append(_term_kind(node))
            return f"SELECT study_id FROM t{len(terms) - 1}"
        if t == "loc":
            key = (node["x"], node["y"], node["z"])
            if key not in points:
                points.append(key)
            return f"SELECT study_id FROM hits WHERE idx = {points.index(key)}"
        if t == "not":
            needs_universe = True
            return f"SELECT study_id FROM universe EXCEPT ({sql_of(node['child'])})"
        op = "INTERSECT" if t == "and" else "UNION"
        return f" {op} ".join(f"({sql_of(c)})" for c in node["children"])

    expression = sql_of(pruned)

    ctes: list[str] = []
    params: dict[str, Any] = {}
    if needs_universe:
        ctes.append(f"universe AS (SELECT study_id FROM {SCHEMA}.metadata)")
    for j, (kind, value) in enumerate(terms):
        params[f"tq{j}"] = _tsquery_param(kind, value)
        q = _tsquery_sql(kind, f"tq{j}")
        ctes.append(
            f"t{j} AS (SELECT study_id FROM {SCHEMA}.metadata "
            f"WHERE fts @@ {q} OR to_tsvector('english', COALESCE(title, '')) @@ {q})"
        )
    if points:
        srid = _coordinate_srid(conn)
        values = []
        for i, (x, y, z) in enumerate(points):
            params.update({f"x{i}": x, f"y{i}": y, f"z{i}": z})
            values.append(f"({i}, ST_SetSRID(ST_MakePoint(:x{i}, :y{i}, :z{i}), {srid}))")
        params["r"] = radius
        ctes.append(f"pts(idx, geom) AS (VALUES {', '.join(values)})")
        # `&&` uses the GIST index to find candidates; ST_3DDistance then applies the exact 3-D radius
        ctes.append(
            f"hits AS (SELECT c.study_id, p.idx FROM {SCHEMA}.coordinates c JOIN pts p "
            "ON c.geom && ST_Expand(p.geom, :r) AND ST_3DDistance(c.geom, p.geom) < :r GROUP BY c.study_id, p.idx)"
        )
    ctes.append(f"matched AS ({expression})")
    info = {"locations": bool(points), "terms": bool(terms), "ignored": ignored_words}
    return "WITH " + ",\n".join(ctes), params, info


_srid_cache: dict[str, int] = {}


def _coordinate_srid(conn) -> int:
    """SRID of coordinates.geom (the query points must carry the same one). A failed lookup is not cached."""
    if "srid" not in _srid_cache:
        try:
            _srid_cache["srid"] = int(conn.execute(text("SELECT Find_SRID(:s, 'coordinates', 'geom')"), {"s": SCHEMA}).scalar() or 0)
        except SQLAlchemyError:
            conn.rollback()
            return 0
    return _srid_cache["srid"]


# =============================================================================
# Request parameters
# =============================================================================
def _float_arg(name: str, default: float, low: float, high: float, *, exclusive_low: bool = False) -> float:
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ApiError(f"Parameter '{name}' must be a number (got {raw!r}).") from None
    if not math.isfinite(value) or value > high or value < low or (exclusive_low and value == low):
        bound = f"> {low:g}" if exclusive_low else f">= {low:g}"
        raise ApiError(f"Parameter '{name}' must be {bound} and <= {high:g} (got {raw}).")
    return value


def _page_args() -> tuple[Optional[int], int]:
    """(limit, offset). Without `limit` every row is returned, as in the course template."""
    limit = offset = None
    for name in ("limit", "offset"):
        raw = request.args.get(name)
        if raw in (None, ""):
            continue
        try:
            value = int(raw)
        except ValueError:
            raise ApiError(f"Parameter '{name}' must be an integer (got {raw!r}).") from None
        if name == "limit":
            limit = max(1, min(value, MAX_LIMIT))
        else:
            offset = max(0, value)
    return limit, offset or 0


def _limit_sql(limit: Optional[int], offset: int, params: dict) -> str:
    sql = ""
    if limit is not None:
        sql += " LIMIT :limit"
        params["limit"] = limit
    if offset:
        sql += " OFFSET :offset"
        params["offset"] = offset
    return sql


def _json_with_cache(payload: dict, max_age: int = 300):
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = f"public, max-age={max_age}"
    return resp


# =============================================================================
# Terms: vocabulary and related terms
# =============================================================================
_term_state: dict[str, Any] = {"source": None, "vocabulary": None, "df": None}
_related_cache: "OrderedDict[tuple[str, Optional[int]], list[dict]]" = OrderedDict()
_term_lock = threading.Lock()


def _strip_prefix(term: str) -> str:
    return term[len(TERM_PREFIX):] if term.startswith(TERM_PREFIX) else term


def _term_source(conn) -> Optional[str]:
    """Where term annotations live: the sparse table (always built by create_db.py) or the JSONB table (optional)."""
    if _term_state["source"] is None:
        for name, label in (("annotations_terms", "terms"), ("annotations_json", "json")):
            if conn.execute(text("SELECT to_regclass(:n)"), {"n": f"{SCHEMA}.{name}"}).scalar():
                n = conn.execute(text(f"SELECT EXISTS (SELECT 1 FROM {SCHEMA}.{name})")).scalar()
                if n:
                    _term_state["source"] = label
                    break
    return _term_state["source"]


def get_vocabulary(conn) -> list[str]:
    with _term_lock:
        if _term_state["vocabulary"] is None:
            source = _term_source(conn)
            if source == "terms":
                rows = conn.execute(text(f"SELECT DISTINCT term FROM {SCHEMA}.annotations_terms")).scalars().all()
            elif source == "json":
                rows = conn.execute(text(f"SELECT DISTINCT jsonb_object_keys(terms) FROM {SCHEMA}.annotations_json")).scalars().all()
            else:
                raise ApiError("The term annotations are missing or empty in the database (see /health).", 503)
            names = {_strip_prefix(t) for t in rows}
            _term_state["vocabulary"] = sorted(t for t in names if t and not t[0].isdigit())
        return _term_state["vocabulary"]


def _document_frequencies(conn) -> dict[str, int]:
    """In how many studies does each stored term occur? (computed once: the data are static)"""
    if _term_state["df"] is None:
        if _term_source(conn) == "terms":
            sql = f"SELECT term, COUNT(DISTINCT study_id) FROM {SCHEMA}.annotations_terms GROUP BY term"
        else:
            sql = (
                f"SELECT k.term, COUNT(DISTINCT aj.study_id) FROM {SCHEMA}.annotations_json aj "
                "JOIN LATERAL jsonb_object_keys(aj.terms) AS k(term) ON TRUE GROUP BY k.term"
            )
        _term_state["df"] = {t: int(n) for t, n in conn.execute(text(sql)).all()}
    return _term_state["df"]


def related_terms(conn, term: str, limit: Optional[int]) -> list[dict]:
    key = (term, limit)
    with _term_lock:
        if key in _related_cache:
            _related_cache.move_to_end(key)
            return _related_cache[key]
    source = _term_source(conn)
    if source is None:
        raise ApiError("The term annotations are missing or empty in the database (see /health).", 503)
    params = {"t": term, "tp": TERM_PREFIX + term}
    if source == "terms":
        sql = f"""
            WITH a AS (SELECT DISTINCT study_id FROM {SCHEMA}.annotations_terms WHERE term IN (:t, :tp))
            SELECT x.term AS term, COUNT(DISTINCT x.study_id) AS co_count, (SELECT COUNT(*) FROM a) AS n_a
            FROM {SCHEMA}.annotations_terms x JOIN a USING (study_id)
            WHERE x.term NOT IN (:t, :tp) GROUP BY x.term"""
    else:
        sql = f"""
            WITH a AS (SELECT DISTINCT study_id FROM {SCHEMA}.annotations_json WHERE terms ? :t OR terms ? :tp)
            SELECT k.term AS term, COUNT(DISTINCT aj.study_id) AS co_count, (SELECT COUNT(*) FROM a) AS n_a
            FROM {SCHEMA}.annotations_json aj JOIN LATERAL jsonb_object_keys(aj.terms) AS k(term) ON TRUE
            WHERE aj.study_id IN (SELECT study_id FROM a) AND k.term NOT IN (:t, :tp) GROUP BY k.term"""
    rows = conn.execute(text(sql), params).all()
    df = _document_frequencies(conn)
    out = []
    for name, co, n_a in rows:
        clean = _strip_prefix(name)
        if not clean or clean[0].isdigit():
            continue
        union = n_a + df.get(name, 0) - co  # |A| + |B| - |A and B|
        out.append({"term": clean, "co_count": int(co), "jaccard": (co / union) if union else None})
    out.sort(key=lambda d: (-(d["jaccard"] if d["jaccard"] is not None else -1.0), -d["co_count"], d["term"]))
    out = out[:limit] if limit is not None else out
    with _term_lock:
        _related_cache[key] = out
        if len(_related_cache) > 300:
            _related_cache.popitem(last=False)
    return out


# =============================================================================
# Maps (NIfTI)
# =============================================================================
def make_map(xyz: np.ndarray, fwhm: float, kernel: str) -> bytes:
    """
    Peak-density map on the MNI152 2 mm grid as a gzipped NIfTI-1 file.
    Every peak adds 1 to its voxel; kernel='gauss' with fwhm > 0 then smooths with
    scipy.ndimage.gaussian_filter (sigma = FWHM / 2.3548 / voxel size, truncate 3, zero padding).
    The map is built in memory: nothing is written to disk, so nothing can leak.
    """
    volume = np.zeros(TEMPLATE_SHAPE, dtype=np.float32)
    if len(xyz):
        homogeneous = np.column_stack([xyz, np.ones(len(xyz))])
        ijk = np.rint(homogeneous @ np.linalg.inv(TEMPLATE_AFFINE)[:3].T).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.array(TEMPLATE_SHAPE)), axis=1)
        flat = np.ravel_multi_index(tuple(ijk[inside].T), TEMPLATE_SHAPE)
        volume = np.bincount(flat, minlength=volume.size).reshape(TEMPLATE_SHAPE).astype(np.float32)
    if kernel == "gauss" and fwhm > 0:
        sigma = fwhm * FWHM_TO_SIGMA / VOXEL_MM
        volume = gaussian_filter(volume, sigma=(sigma, sigma, sigma), mode="constant", truncate=3.0)
    image = nib.Nifti1Image(volume.astype(np.float32), TEMPLATE_AFFINE)
    image.header.set_xyzt_units("mm", "sec")
    image.set_sform(TEMPLATE_AFFINE, code=4)
    image.set_qform(TEMPLATE_AFFINE, code=4)
    return gzip.compress(image.to_bytes(), compresslevel=6, mtime=0)


# =============================================================================
# Static frontend (the built React app), served only if dist/ exists
# =============================================================================
def _frontend_available() -> bool:
    return (DIST_DIR / "index.html").is_file()


# =============================================================================
# Flask application
# =============================================================================
def help_payload() -> dict:
    return {
        "service": {"name": "LoTUS-BF", "tagline": "Location-or-Term Unified Search for Brain Functions", "version": __version__},
        "ok": True,
        "endpoints": [
            "/terms",
            "/terms/<term>",
            "/query/<query_string>/studies",
            "/query/<query_string>/locations",
            "/query/<query_string>/nii",
            "/health",
        ],
        "notes": "Use /query/<mixed>/(studies|locations|nii). 'mixed' supports coordinates like [x,y,z] and boolean terms with AND/OR/NOT.",
        "docs": {
            "syntax": {
                "mixed_query": "/query/<mixed_query>/(studies|locations|nii)",
                "operators": ["AND", "OR", "NOT", "(", ")"],
                "precedence": "NOT binds tighter than AND, which binds tighter than OR; AND may be omitted",
                "locations": "[x,y,z] with optional ?r=float (default 6.0, max 50)",
                "terms": "bare words are ANDed (use quotes for an exact phrase); word* matches by prefix",
            },
            "params": {
                "r": "sphere radius around [x,y,z] in mm (default 6.0)",
                "voxel": "voxel size of the NIfTI map; only 2 (the MNI152 2 mm grid) is supported",
                "fwhm": "Gaussian smoothing FWHM in mm (default 10.0; 0 = raw peak counts)",
                "kernel": "gauss | uniform (default gauss); 'uniform' returns the unsmoothed counts",
                "limit_offset": "limit, offset for pagination on studies/locations (the response also reports 'total')",
            },
            "examples": [
                '/query/[-2,50,-6] NOT "ventromedial prefrontal"/studies',
                "/query/([30,-60,50] OR [40,-50,45]) AND default mode/locations?r=8",
                "/query/[-2,50,-6] AND reward/nii?voxel=2&fwhm=8&kernel=gauss",
            ],
        },
    }


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)

    # -- errors are always JSON, never an HTML page or a stack trace ---------------------------------
    @app.errorhandler(ApiError)
    def on_api_error(err: ApiError):
        return jsonify({"error": err.message, **err.extra}), err.status

    @app.errorhandler(OperationalError)
    def on_operational_error(err: OperationalError):
        log.warning("database unavailable: %s", err.orig)
        return jsonify({"error": "The database is unavailable right now. Please try again later."}), 503

    @app.errorhandler(DBAPIError)
    def on_db_error(err: DBAPIError):
        if getattr(err.orig, "pgcode", None) == "57014":  # query_canceled (statement_timeout)
            return jsonify({"error": "The query took too long. Try a more specific query."}), 504
        log.exception("database error")
        return jsonify({"error": "The database could not run this query."}), 500

    @app.errorhandler(404)
    def on_not_found(_err):
        return jsonify({"error": "Not found."}), 404

    @app.errorhandler(Exception)
    def on_unexpected(err: Exception):
        if isinstance(err, HTTPException):  # e.g. 405 Method Not Allowed
            return jsonify({"error": err.description}), err.code
        if isinstance(err, SQLAlchemyError):
            log.exception("database error")
            return jsonify({"error": "The database could not run this query."}), 500
        log.exception("unexpected error")
        return jsonify({"error": "Internal server error."}), 500

    @app.after_request
    def cors(resp: Response):
        resp.headers["Access-Control-Allow-Origin"] = "*"  # any frontend origin may fetch() the API
        return resp

    # -- service endpoints -----------------------------------------------------------------------------
    @app.get("/help")
    def help_():
        return jsonify(help_payload())

    @app.get("/health")
    def health():
        problems: list[str] = []
        body: dict[str, Any] = {"version": __version__, "frontend": _frontend_available()}
        try:
            with get_engine().connect() as conn:
                body["database"] = conn.execute(text("SHOW server_version")).scalar()
                body["studies"] = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.metadata")).scalar()
                body["peaks"] = conn.execute(text(f"SELECT COUNT(*) FROM {SCHEMA}.coordinates")).scalar()
                body["term_source"] = _term_source(conn)
                if body["term_source"] is None:
                    problems.append(
                        "No term annotations found (annotations_terms is empty or missing). UNLOGGED tables are emptied after a "
                        "database crash: re-run create_db.py to rebuild them."
                    )
        except ApiError as err:
            problems.append(err.message)
        except SQLAlchemyError as err:
            problems.append(f"Cannot use the database: {type(err).__name__}")
        body["ok"] = not problems
        body["problems"] = problems
        resp = jsonify(body)
        resp.headers["Cache-Control"] = "no-store"
        return resp, (200 if not problems else 503)

    @app.get("/terms")
    def terms():
        with get_engine().connect() as conn:
            return _json_with_cache({"terms": get_vocabulary(conn)}, max_age=3600)

    @app.get("/terms/<term>")
    def terms_related(term: str):
        limit, _ = _page_args()
        name = term.strip().lower()
        with get_engine().connect() as conn:
            return _json_with_cache({"term": term, "related": related_terms(conn, name, limit)}, max_age=3600)

    # -- query endpoints ---------------------------------------------------------------------------------
    def run_query(query_string: str):
        ast = parse_query(query_string)
        radius = _float_arg("r", DEFAULT_RADIUS, 0.0, MAX_RADIUS, exclusive_low=True)
        return ast, radius

    @app.get("/query/<path:query_string>/studies")
    def query_studies(query_string: str):
        ast, radius = run_query(query_string)
        limit, offset = _page_args()
        with get_engine().connect() as conn:
            prefix, params, info = compile_matched(conn, ast, radius)
            columns = _metadata_columns(conn)
            page = _limit_sql(limit, offset, params)
            sql = (
                f"{prefix}\nSELECT {', '.join('m.' + c for c in columns)}, COUNT(*) OVER () AS _total\n"
                f"FROM {SCHEMA}.metadata m JOIN matched x USING (study_id)\nORDER BY m.study_id{page}"
            )
            rows = [dict(r) for r in conn.execute(text(sql), params).mappings().all()]
            if rows:
                total = rows[0]["_total"]
            elif offset:
                total = conn.execute(text(f"{prefix}\nSELECT COUNT(*) FROM matched"), {k: v for k, v in params.items() if k not in ("limit", "offset")}).scalar()
            else:
                total = 0
        for r in rows:
            r.pop("_total", None)
        applied = {"locations": info["locations"], "terms": info["terms"], "r": radius if info["locations"] else None}
        return _json_with_cache({"count": len(rows), "total": int(total), "applied": applied, "ignored": info["ignored"], "results": rows})

    @app.get("/query/<path:query_string>/locations")
    def query_locations(query_string: str):
        ast, radius = run_query(query_string)
        limit, offset = _page_args()
        with get_engine().connect() as conn:
            prefix, params, info = compile_matched(conn, ast, radius)
            page = _limit_sql(limit, offset, params)
            sql = f"""{prefix},
                base AS (SELECT c.study_id, ST_X(c.geom)::float8 AS x, ST_Y(c.geom)::float8 AS y, ST_Z(c.geom)::float8 AS z
                         FROM {SCHEMA}.coordinates c JOIN matched USING (study_id)),
                tot AS (SELECT COUNT(*) AS n, COUNT(DISTINCT study_id) AS s FROM base),
                pg AS (SELECT * FROM base ORDER BY study_id COLLATE "C", x, y, z{page})
                SELECT pg.study_id, pg.x, pg.y, pg.z, tot.n AS _n, tot.s AS _s FROM tot LEFT JOIN pg ON TRUE
                ORDER BY pg.study_id COLLATE "C", pg.x, pg.y, pg.z"""
            rows = conn.execute(text(sql), params).mappings().all()
        total_peaks = int(rows[0]["_n"]) if rows else 0
        total_studies = int(rows[0]["_s"]) if rows else 0
        results = [{"study_id": r["study_id"], "x": r["x"], "y": r["y"], "z": r["z"]} for r in rows if r["study_id"] is not None]
        return _json_with_cache(
            {
                "location_counts": len(results),
                "study_counts": len({r["study_id"] for r in results}),
                "total_locations": total_peaks,
                "total_studies": total_studies,
                "ignored": info["ignored"],
                "results": results,
            }
        )

    @app.get("/query/<path:query_string>/nii")
    def query_nii(query_string: str):
        ast, radius = run_query(query_string)
        fwhm = _float_arg("fwhm", 10.0, 0.0, 50.0)
        kernel = (request.args.get("kernel") or "gauss").lower()
        if kernel not in ("gauss", "uniform"):
            raise ApiError("Parameter 'kernel' must be 'gauss' or 'uniform'.")
        voxel = _float_arg("voxel", VOXEL_MM, 0.0, 100.0, exclusive_low=True)
        if voxel != VOXEL_MM:
            raise ApiError("Parameter 'voxel' must be 2: the maps use the MNI152 2 mm grid.")
        with get_engine().connect() as conn:
            prefix, params, _info = compile_matched(conn, ast, radius)
            sql = (
                f"{prefix}\nSELECT ST_X(c.geom)::float8, ST_Y(c.geom)::float8, ST_Z(c.geom)::float8\n"
                f"FROM {SCHEMA}.coordinates c JOIN matched USING (study_id)"
            )
            rows = conn.execute(text(sql), params).all()
        # np.array(<list of Row objects>) is ~70x slower than reading the tuples straight into a float array
        xyz = np.fromiter((v for row in rows for v in row), dtype=float, count=len(rows) * 3).reshape(-1, 3)
        data = make_map(xyz, fwhm, kernel)
        resp = Response(data, mimetype="application/octet-stream")
        resp.headers["Content-Disposition"] = 'attachment; filename="query_mixed_map.nii.gz"'
        resp.headers["Cache-Control"] = "public, max-age=300"
        resp.headers["X-Peak-Count"] = str(len(xyz))
        resp.headers["Access-Control-Expose-Headers"] = "X-Peak-Count"
        return resp

    # -- the built frontend ------------------------------------------------------------------------------
    @app.get("/")
    def index():
        if _frontend_available():
            resp = send_from_directory(DIST_DIR, "index.html")
            resp.headers["Cache-Control"] = "no-cache"
            return resp
        return jsonify({"service": "LoTUS-BF API", "docs": "/help", "health": "/health", "frontend": "not built (run `npm run build:portable`)"})

    @app.get("/<path:filename>")
    def frontend_file(filename: str):
        if not _frontend_available() or not (DIST_DIR / filename).is_file():
            raise ApiError("Not found.", 404)
        resp = send_from_directory(DIST_DIR, filename)  # refuses paths outside dist/
        if filename.startswith("assets/"):
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"  # content-hashed file names
        return resp

    return app


_columns_cache: list[str] = []


def _metadata_columns(conn) -> list[str]:
    """Every metadata column except the full-text index (which is large and not useful to clients)."""
    if not _columns_cache:
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = :s AND table_name = 'metadata' AND column_name <> 'fts' ORDER BY ordinal_position"
            ),
            {"s": SCHEMA},
        ).scalars().all()
        if not rows:
            raise ApiError(f"The table {SCHEMA}.metadata was not found. Run create_db.py first.", 503)
        _columns_cache.extend(rows)
    return _columns_cache


app = create_app()

if __name__ == "__main__":  # python app.py  (development server; use gunicorn in production)
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
