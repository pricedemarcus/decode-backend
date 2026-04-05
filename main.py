"""
DECODE — AI Riff Analysis Backend
FastAPI + Librosa + Claude API
"""

import os
import json
import tempfile
import math
import subprocess
import numpy as np
import librosa
import anthropic

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DECODE API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
MAX_DURATION_SECONDS = 120


# ─── Audio Analysis Helpers ───────────────────────────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_key(chroma_mean: np.ndarray) -> str:
    """Krumhansl-Schmuckler key estimation from chroma vector."""
    best_corr = -2.0
    best_key = "E Minor"

    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        corr_major = float(np.corrcoef(rotated, MAJOR_PROFILE)[0, 1])
        corr_minor = float(np.corrcoef(rotated, MINOR_PROFILE)[0, 1])

        if corr_major > best_corr:
            best_corr = corr_major
            best_key = f"{NOTE_NAMES[i]} Major"
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = f"{NOTE_NAMES[i]} Minor"

    return best_key


def hz_to_note_name(hz: float):
    """Convert frequency in Hz to note name (e.g. 'E', 'A#')."""
    if hz <= 0 or math.isnan(hz) or math.isinf(hz):
        return None
    midi = 12 * math.log2(hz / 440.0) + 69
    return NOTE_NAMES[int(round(midi)) % 12]


def extract_pitch_notes(y: np.ndarray, sr: int) -> list[str]:
    """Extract unique note names from audio using pyin pitch detection."""
    y_harm, _ = librosa.effects.hpss(y)
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y_harm,
            fmin=float(librosa.note_to_hz("E1")),
            fmax=float(librosa.note_to_hz("E6")),
            hop_length=512,
        )
        notes = set()
        if f0 is not None:
            for freq, voiced in zip(f0, voiced_flag):
                if voiced and freq and not np.isnan(freq):
                    n = hz_to_note_name(float(freq))
                    if n:
                        notes.add(n)
        return list(notes)
    except Exception:
        return []


def dominant_chroma_notes(chroma_mean: np.ndarray, top_n: int = 6) -> list[str]:
    """Return top N most prominent pitch classes from chroma vector."""
    indices = np.argsort(chroma_mean)[-top_n:][::-1]
    return [NOTE_NAMES[i] for i in indices]


def analyze_audio(file_path: str) -> dict:
    """
    Core audio analysis — returns structured data ready for Claude.
    """
    y, sr = librosa.load(file_path, duration=MAX_DURATION_SECONDS, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))

    # Tempo
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(round(float(tempo_arr)))

    # Chroma → key
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key = estimate_key(chroma_mean)
    dominant_notes = dominant_chroma_notes(chroma_mean)

    # Pitch-level note detection
    pitch_notes = extract_pitch_notes(y, sr)

    # RMS energy — rough proxy for dynamics
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # Spectral centroid — higher = brighter tone (lead/single-note), lower = fuller (chords)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # Zero crossing rate — higher suggests distorted/noisy signal (electric guitar)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))

    return {
        "duration": round(duration, 1),
        "tempo": tempo,
        "key_estimate": key,
        "dominant_notes": dominant_notes,
        "pitch_notes": pitch_notes,
        "spectral_centroid_hz": round(centroid, 1),
        "zero_crossing_rate": round(zcr, 4),
        "rms_energy": round(rms, 4),
    }


# ─── Claude Interpretation ───────────────────────────────────────────────────

def build_prompt(data: dict, filename: str) -> str:
    notes_str = ", ".join(data["pitch_notes"]) if data["pitch_notes"] else "see dominant chroma below"
    dominant_str = ", ".join(data["dominant_notes"])

    # Infer signal character for Claude context
    signal_desc = []
    if data["zero_crossing_rate"] > 0.1:
        signal_desc.append("distorted/electric signal")
    if data["spectral_centroid_hz"] > 2000:
        signal_desc.append("bright/cutting tone (likely lead or single-note)")
    elif data["spectral_centroid_hz"] < 800:
        signal_desc.append("warm/full tone (likely chords or bass)")
    else:
        signal_desc.append("balanced tone (rhythm guitar or mixed)")

    signal_context = "; ".join(signal_desc) if signal_desc else "standard guitar signal"

    return f"""You are an expert guitar instructor and music theorist analyzing a student's uploaded audio clip.

AUDIO ANALYSIS DATA:
- File: {filename}
- Duration: {data['duration']} seconds
- Detected tempo: {data['tempo']} BPM
- Key estimate (Krumhansl-Schmuckler): {data['key_estimate']}
- Dominant pitch classes (chroma, most prominent first): {dominant_str}
- Individual notes detected via pitch tracking: {notes_str}
- Signal character: {signal_context}
- RMS energy: {data['rms_energy']}

Based on this data, provide a complete beginner-focused guitar lesson breakdown.
Be accurate to the musical data. If it sounds like metal/rock use power chords; if acoustic use open chords, etc.

Respond with ONLY valid JSON — no markdown, no code fences:

{{
  "key": "string (e.g. 'E Minor')",
  "tempo": {data['tempo']},
  "time_signature": "4/4",
  "chord_progression": [
    {{"name": "Em", "beats": 2}},
    {{"name": "G",  "beats": 2}},
    {{"name": "D",  "beats": 2}},
    {{"name": "A",  "beats": 2}}
  ],
  "notes_in_scale": ["E", "G", "A", "B", "D"],
  "techniques": ["Alternate Picking", "Palm Muting"],
  "difficulty": "Beginner",
  "difficulty_weeks": "2–3 weeks with daily practice",
  "tab": "e|------------|\\nB|------------|\\nG|------------|\\nD|-2--4--2----|\\nA|-2--4--2----|\\nE|-0--2--0----|",
  "practice_plan": [
    {{
      "step": 1,
      "title": "Learn the root chord shape",
      "description": "Detailed beginner-friendly instruction, 2–3 sentences.",
      "bpm": "Start slow",
      "tip": "One short technique tip"
    }},
    {{
      "step": 2,
      "title": "Practice the picking pattern",
      "description": "Detailed instruction.",
      "bpm": "60 BPM",
      "tip": "Short tip"
    }},
    {{
      "step": 3,
      "title": "Add the chord transitions",
      "description": "Detailed instruction.",
      "bpm": "80 BPM",
      "tip": "Short tip"
    }},
    {{
      "step": 4,
      "title": "Build up to full tempo",
      "description": "Detailed instruction.",
      "bpm": "{int(data['tempo'])} BPM",
      "tip": "Short tip"
    }}
  ],
  "insight": "One expert observation about this riff in 1–2 sentences. Mention genre, feel, or what makes it interesting to play.",
  "tone_profile": {{
    "amp": "e.g. Marshall JCM800",
    "gain": "e.g. 7/10",
    "bass": "e.g. 6/10",
    "mid": "e.g. 4/10",
    "treble": "e.g. 7/10",
    "effects": ["e.g. Overdrive", "Light Reverb"],
    "pickup": "e.g. Bridge humbucker",
    "notes": "1-2 sentences on how to dial in this tone."
  }}
}}"""


def call_claude(prompt: str) -> dict:
    """Send prompt to Claude, return parsed JSON result."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured.")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Strip code fences if Claude wraps the JSON
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

    return json.loads(raw)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "DECODE API"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Upload an audio file and receive a full AI-powered guitar breakdown.
    Returns JSON with key, tempo, chords, techniques, tab, and practice plan.
    """
    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        if len(content) > 50 * 1024 * 1024:  # 50MB hard cap
            raise HTTPException(status_code=413, detail="File too large. Max 50MB.")
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 1 — Audio analysis via librosa
        audio_data = analyze_audio(tmp_path)

        # Step 2 — Claude interpretation
        prompt = build_prompt(audio_data, file.filename or "unknown")
        result = call_claude(prompt)

        # Merge raw audio stats into result for frontend
        result["duration"] = audio_data["duration"]
        result["filename"] = file.filename
        result["raw_tempo"] = audio_data["tempo"]

        return result

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"AI response parse error: {e}")
    except librosa.util.exceptions.ParameterError as e:
        raise HTTPException(status_code=422, detail=f"Could not process audio: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/analyze-url")
async def analyze_url(request: Request):
    """
    Accept a YouTube (or other yt-dlp compatible) URL, download the audio,
    and run the same AI analysis pipeline as /analyze.
    """
    body = await request.json()
    url = (body.get("url") or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "%(id)s.%(ext)s")

        dl_cmd = [
            "yt-dlp",
            "-f", "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio",
            "--max-filesize", "50m",
            "--no-playlist",
            "-o", output_template,
            url,
        ]
        dl = subprocess.run(dl_cmd, capture_output=True, text=True, timeout=90)
        if dl.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail=f"Could not download audio. Make sure the URL is a public video. ({dl.stderr[-200:]})"
            )

        files = [f for f in os.listdir(tmpdir) if not f.startswith(".")]
        if not files:
            raise HTTPException(status_code=500, detail="Download produced no audio file.")

        audio_path = os.path.join(tmpdir, files[0])

        title_cmd = ["yt-dlp", "--get-title", "--no-playlist", url]
        title_res = subprocess.run(title_cmd, capture_output=True, text=True, timeout=30)
        title = title_res.stdout.strip() if title_res.returncode == 0 else "YouTube Riff"

        try:
            audio_data = analyze_audio(audio_path)
            prompt = build_prompt(audio_data, title)
            result = call_claude(prompt)
            result["duration"] = audio_data["duration"]
            result["filename"] = title
            result["raw_tempo"] = audio_data["tempo"]
            result["source_url"] = url
            return result
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"AI response parse error: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat(request: Request):
    """
    AI guitar teacher — conversational endpoint.
    Accepts { messages: [{role, content}], context: {key, tempo, techniques, filename, difficulty} }
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured.")

    body = await request.json()
    messages = body.get("messages", [])
    ctx = body.get("context", {})

    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    techniques_str = ", ".join(ctx.get("techniques", [])) or "unknown"
    system = f"""You are DECODE's AI guitar teacher — an expert guitarist and music educator with 20 years of teaching experience.
You are helping a student learn a specific riff that was just analyzed.

Current riff context:
- Song/File: {ctx.get('filename', 'Unknown')}
- Key: {ctx.get('key', 'Unknown')}
- Tempo: {ctx.get('tempo', 'Unknown')} BPM
- Techniques detected: {techniques_str}
- Difficulty: {ctx.get('difficulty', 'Unknown')}

Rules:
- Be encouraging, specific, and practical
- Keep responses under 4 sentences unless the student asks for detailed explanation
- Give actionable advice the student can try immediately
- If asked about technique, describe hand position and motion concisely
- Reference the specific riff context when relevant"""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=system,
        messages=messages,
    )

    return {"response": response.content[0].text}
