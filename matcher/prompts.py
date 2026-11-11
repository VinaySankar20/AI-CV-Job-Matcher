"""All prompts and JSON schemas in one place, so they are easy to tune."""

# ---------------------------------------------------------------- 1. read the job ad
EXTRACT_SYSTEM = """You are an expert recruiter in Germany. You read job descriptions (German or English)
and extract the requirements precisely. You only use information that is in the job description.
Answer with JSON only."""

EXTRACT_USER = """Extract the requirements from this job description.

Rules:
- "must_have": requirements the ad presents as required (e.g. "required", "you have", "Voraussetzung",
  "du bringst mit", "Anforderungen", "erforderlich"). Maximum 12.
- "nice_to_have": things marked optional (e.g. "nice to have", "a plus", "von Vorteil", "wünschenswert",
  "idealerweise"). Maximum 10.
- Each requirement "name" is short (2-8 words), in the language of the ad.
- "category" is one of: skill, experience, education, language, other.
- "keywords": 10-25 short technical terms / tools / methods an ATS would search for (1-3 words each,
  e.g. "Python", "SQL", "Power BI", "Machine Learning", "Scrum"). No soft skills.
- "min_years_experience": a number, 0 if not stated.
- Use "" if company, title, location or contact person are not stated.

JOB DESCRIPTION:
\"\"\"
{jd}
\"\"\""""

_REQ_ITEM = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "category": {"type": "string", "enum": ["skill", "experience", "education", "language", "other"]},
    },
    "required": ["name", "category"],
}

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "job_title": {"type": "string"},
        "company": {"type": "string"},
        "location": {"type": "string"},
        "contact_person": {"type": "string"},
        "must_have": {"type": "array", "items": _REQ_ITEM},
        "nice_to_have": {"type": "array", "items": _REQ_ITEM},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "min_years_experience": {"type": "number"},
    },
    "required": ["job_title", "company", "location", "contact_person", "must_have",
                 "nice_to_have", "keywords", "min_years_experience"],
}

# ---------------------------------------------------------------- 2. compare CV to requirements
EVALUATE_SYSTEM = """You are a strict but fair recruiter in Germany. You compare a candidate's CV with a
job's requirements. You never assume skills that are not written in the CV. Evidence must be copied
word-for-word from the CV. Answer with JSON only, in English."""

EVALUATE_USER = """Compare the CV with the job.

For EVERY requirement id below (you must return one entry per id — do not skip nice-to-haves), decide:
- "met": the CV clearly shows it
- "partial": related or weaker experience (e.g. a similar tool, less years, only academic use)
- "missing": nothing in the CV supports it
and copy the supporting text from the CV into "evidence" (exact words, max 25 words; "" if missing).

Then score these categories from 0 to 5 (decimals allowed, 5 = perfect fit):
- skills: technical and professional skills vs. what the job needs
- experience: relevant work/project experience and years (job wants {years} years; 0 = not stated)
- education: degree/field vs. what the job needs. B.Tech, B.E., B.Eng, B.Sc., BCA = a Bachelor's degree;
  M.Sc., M.Tech, M.Eng = a Master's degree; a Master's student already holds a Bachelor's.
  Read the EDUCATION section carefully before saying a degree is missing.
- languages: language requirements (German/English level) vs. the CV
Give a one-sentence reason for each score.

Also give:
- "strengths": 3-5 strongest points of this candidate for THIS job
- "gaps": the most important missing points (max 5)
- "cv_tips": 3-5 concrete, honest tips to improve the CV for this job (no inventing experience)

JOB: {title} at {company}

REQUIREMENTS:
{requirements}

CV:
\"\"\"
{cv}
\"\"\""""

_SCORE = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "reason": {"type": "string"}},
    "required": ["score", "reason"],
}

EVALUATE_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "status": {"type": "string", "enum": ["met", "partial", "missing"]},
                    "evidence": {"type": "string"},
                },
                "required": ["id", "status", "evidence"],
            },
        },
        "scores": {
            "type": "object",
            "properties": {"skills": _SCORE, "experience": _SCORE, "education": _SCORE, "languages": _SCORE},
            "required": ["skills", "experience", "education", "languages"],
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "cv_tips": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["requirements", "scores", "strengths", "gaps", "cv_tips"],
}

RECHECK_USER = """For EACH requirement id below, decide if the CV shows it:
"met" (clearly shown), "partial" (related or weaker) or "missing" (nothing supports it),
and copy the supporting text word-for-word from the CV into "evidence" ("" if missing).
Return one entry per id.

REQUIREMENTS:
{requirements}

CV:
\"\"\"
{cv}
\"\"\""""

RECHECK_SCHEMA = {
    "type": "object",
    "properties": {"requirements": EVALUATE_SCHEMA["properties"]["requirements"]},
    "required": ["requirements"],
}

# ---------------------------------------------------------------- 3. CV tailoring
TAILOR_SYSTEM = """You are a CV coach for the German job market. You rewrite existing CV bullet points so
they speak to a specific job, using the job's keywords where they are TRUE for the candidate.
You never invent tools, numbers, employers or responsibilities. Answer with JSON only."""

TAILOR_USER = """Pick the {n} CV lines that matter most for this job and rewrite each one.

Rules:
- "original": copy the line exactly from the CV.
- "suggested": the improved line, in the same language as the CV ({cv_lang}). Start with a strong verb,
  add the job's keywords only where the original supports them.
- "why": one short sentence on what the change does for this job.
- "needs_confirmation": true if the suggestion adds anything (a number, a tool, a result) the candidate
  must confirm is true; false if it only rephrases.

JOB KEYWORDS: {keywords}
JOB MUST-HAVES: {must}

CV:
\"\"\"
{cv}
\"\"\""""

TAILOR_SCHEMA = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "original": {"type": "string"},
                    "suggested": {"type": "string"},
                    "why": {"type": "string"},
                    "needs_confirmation": {"type": "boolean"},
                },
                "required": ["original", "suggested", "why", "needs_confirmation"],
            },
        }
    },
    "required": ["suggestions"],
}

# ---------------------------------------------------------------- 4. cover letter
COVER_SYSTEM = """You write cover letters (Anschreiben) for the German job market. You write in {language}.
You only use facts from the candidate's CV and profile — never invent experience, tools, grades or numbers.
You do not mention any match score. Answer with JSON only."""

COVER_USER = """Write a cover letter for this application.

Language: {language}
Tone: {tone_text}
Length of the body: about {words} words in {paragraphs} paragraphs. Stay close to this length.

Structure of the body:
1. Opening: which position, why this company / role specifically (use details from the job ad).
2. Main part: 2-3 of the candidate's strongest matching points, each tied to a job requirement,
   using ONLY the verified matches listed below.
3. Honest handling of the most important gap (if any), framed positively (e.g. eager to learn it).
4. Close: {availability_rule} Invite them to an interview.

STRICT FACT RULES — breaking any of these makes the letter unusable:
- Only claim skills, tools, degrees, numbers and experience that appear in the CV text.
- NEVER claim or imply experience with any of these (the CV does not show them): {do_not_claim}
  You may only mention one of them as something the candidate is keen to learn.
- Education in the CV: {education}. Do not say the candidate lacks a degree that is listed here.
- German: {german_rule}
- Do not write the candidate's name anywhere in the body or closing (it is added below the letter).

VERIFIED MATCHES (requirement → evidence from the CV):
{matches}

Salutation: {salutation_hint}
Subject line: a short subject naming the position and company (in {language}; no word like "Betreff").
Closing formula: exactly "{closing}" (nothing else, no name).

{extra}

JOB DESCRIPTION:
\"\"\"
{jd}
\"\"\"

CANDIDATE STRENGTHS FOR THIS JOB: {strengths}

CV:
\"\"\"
{cv}
\"\"\""""

COVER_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string"},
        "salutation": {"type": "string"},
        "paragraphs": {"type": "array", "items": {"type": "string"}},
        "closing": {"type": "string"},
    },
    "required": ["subject", "salutation", "paragraphs", "closing"],
}
