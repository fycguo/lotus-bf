# LoTUS-BF (deployment copy)

Flask + PostgreSQL/PostGIS backend (`app.py`) that also serves the built React frontend (`dist/`).
Deployed on Render with `render.yaml` (set the `DB_URL` environment variable to the database's Internal URL).
`GET /health` shows whether the database and data are fine, `GET /help` documents the API.

The full source (React/Vite sources, tests, documentation) is in the course submission `HW09_R14250002_lotus-bf.zip`.
