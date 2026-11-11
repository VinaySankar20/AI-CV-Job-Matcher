# 🎯 CV ↔ Job Matcher

A local AI tool that compares your CV with a job description, gives you a **match score out of 5**
and, when the score is **above 3.5**, writes a tailored **cover letter** in the language of the job ad.
It runs entirely on your own computer with [Ollama](https://ollama.com), so your CV never leaves your machine.

## Features (v1)

| | Feature |
|---|---|
| 🧮 | **Hybrid scoring**: AI judgement + requirement checklist + semantic similarity (embeddings) + ATS keyword coverage |
| ✅ | **Must-have vs nice-to-have**: every missing must-have costs 0.25 points (max 1.0) |
| 🇩🇪 | **Germany checks**: required German level (CEFR) vs yours, Werkstudent/part-time/full-time, weekly hours vs the usual 20 h student limit |
| 🛡️ | **Anti-hallucination**: the AI must quote evidence from your CV; claims it can't support are downgraded |
| ✍️ | **CV tailoring**: rewrites of your existing CV lines for the specific job (nothing invented) |
| ✉️ | **Cover letter**: formal/modern tone, short/standard/detailed, extra instructions, edit in the app, export **DOCX / PDF (DIN 5008)** / TXT |
| 💾 | **Saved CV**: upload once, then just paste job ads |
| 📋 | **Application tracker**: every analysis saved (SQLite) with status, notes and CSV export |

### How the score works

```
skills     = 50% AI + 30% requirement checklist + 20% semantic similarity
experience = AI          education = AI
languages  = AI, capped if the job's German level is above yours
keywords   = share of the ad's ATS keywords that appear in your CV

overall = 35% skills + 25% experience + 15% languages + 15% education + 10% keywords
          − 0.25 per missing must-have (max 1.0)
cover letter if overall > 3.5
```
Weights and thresholds are in `matcher/config.py`.

## Setup on Windows (one time)

1. **Install Python 3.11+** from python.org (tick *“Add python.exe to PATH”*).
2. **Install Ollama for Windows** from https://ollama.com/download and start it.
3. **Download the models** (in PowerShell):
   ```powershell
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
   `qwen2.5:7b` (~4.7 GB) fits a 6 GB GPU and handles German well. `nomic-embed-text` (~270 MB) powers the semantic check.
4. **Create the environment** in the project folder:
   ```powershell
   cd "C:\Users\<you>\MyProjects\cv-job-matcher"
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Run

```powershell
.venv\Scripts\activate
streamlit run app.py
```
or double-click **`run.bat`**. The app opens at http://localhost:8501.

First use:
1. **My profile** → name, address, place, German level, student status, hours you already work elsewhere, availability.
2. **Analyse** → upload your CV (PDF, text-based) once, paste a job ad, click **Analyse match**.
3. Check the breakdown, Germany checks and requirements → **Suggest CV rewrites** → **Generate cover letter** → download.
4. **Tracker** → update status (applied / interview / …) and notes; export CSV.

The first analysis takes longer while the model loads into memory.

## Using a different model

```powershell
$env:CVM_CHAT_MODEL = "llama3.1:8b"   # any Ollama chat model
streamlit run app.py
```
Other settings: `OLLAMA_HOST`, `CVM_EMBED_MODEL`, `CVM_NUM_CTX`, `CVM_DATA_DIR`.

## Tests

```powershell
python -m pytest -q
```
Tests use a fake model, so they run without Ollama.

## Project structure

```
cv-job-matcher/
├── app.py                 # Streamlit UI (Analyse / Tracker / My profile)
├── matcher/
│   ├── config.py          # weights, thresholds, model names
│   ├── llm.py             # Ollama wrapper (JSON-schema outputs, embeddings)
│   ├── prompts.py         # all prompts + JSON schemas
│   ├── pdf_reader.py      # CV text extraction
│   ├── germany.py         # rule-based German-market checks
│   ├── textutils.py       # keyword matching, chunking, cosine similarity
│   ├── scorer.py          # hybrid scoring pipeline
│   ├── tailor.py          # CV rewrite suggestions
│   ├── cover_letter.py    # letter content
│   ├── export.py          # DOCX / PDF in DIN 5008 layout
│   └── storage.py         # SQLite: profile, CV, applications
├── tests/
└── data/                  # your local database (git-ignored)
```

## Privacy

Everything (CV, profile, applications) is stored in `data/cv_matcher.db` on your computer and is git-ignored.
No data is sent anywhere except to your local Ollama server.

## Roadmap (v2)

Batch mode with ranked jobs · job description from URL · interview prep questions · radar-chart dashboard ·
model switcher (Gemini / Claude) · Docker · CI.

## License

MIT
