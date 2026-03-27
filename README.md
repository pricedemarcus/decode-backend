# DECODE — Backend

FastAPI + Librosa + Claude AI analysis engine.

## Local Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env

uvicorn main:app --reload --port 8000
```

API is now live at `http://localhost:8000`
- `GET  /health` — status check
- `POST /analyze` — upload audio file, receive analysis JSON

## Deploy to Railway (recommended — free tier)

1. Push this `/backend` folder to a GitHub repo
2. Go to railway.app → New Project → Deploy from GitHub
3. Set env var: `ANTHROPIC_API_KEY=your_key`
4. Railway auto-detects Python and deploys
5. Copy your Railway URL (e.g. `https://decode-api.up.railway.app`)
6. Update `API_URL` in `index.html` to that URL

## Deploy to Render (alternative)

1. Push to GitHub
2. render.com → New Web Service → connect repo
3. Set env var: `ANTHROPIC_API_KEY=your_key`
4. Deploy — uses `render.yaml` config automatically

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |

## How It Works

1. Client uploads audio file (MP3/WAV/M4A/FLAC)
2. Librosa analyzes: tempo, key (Krumhansl-Schmuckler), dominant notes, spectral features
3. Analysis data is sent to Claude Sonnet with a structured guitar instructor prompt
4. Claude returns: key, chord progression, techniques, difficulty, guitar tab, 4-step practice plan
5. JSON is returned to the frontend for rendering
