# TruthLens frontend

React + Vite + TypeScript frontend for the TruthLens verification API.

## Run locally

```bash
npm install
npm run dev
```

The app expects the FastAPI server at `http://localhost:8000` by default. Copy `.env.example` to `.env` to point `VITE_API_BASE_URL` at another API origin.

The workspace maps to the two backend endpoints:

- `POST /verify-text` with `{ "text": "..." }`
- `POST /verify-image` with a multipart `file` field
