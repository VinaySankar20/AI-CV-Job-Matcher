"""CV ↔ Job Matcher — Streamlit app.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import os
import re

import pandas as pd
import streamlit as st

from matcher import config
from matcher.cover_letter import (LENGTHS, fact_check, format_date, generate_cover_letter, letter_to_text,
                                  text_to_letter)
from matcher.export import to_docx, to_pdf
from matcher.germany import CEFR
from matcher.llm import LLMError, OllamaLLM
from matcher.pdf_reader import PDFTextError, extract_pdf_text
from matcher.scorer import analyse
from matcher.storage import Storage
from matcher.tailor import tailor_cv

st.set_page_config(page_title="CV ↔ Job Matcher", page_icon="🎯", layout="wide")

STATUS_ICON = {"met": "✅ met", "partial": "🟡 partial", "missing": "❌ missing"}
FLAG_FN = {"success": st.success, "info": st.info, "warning": st.warning, "error": st.error}


@st.cache_resource
def get_storage() -> Storage:
    return Storage()


@st.cache_resource
def get_llm() -> OllamaLLM:
    return OllamaLLM()


def ss_default(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


for k, v in {"analysis": None, "app_id": None, "jd_text": "", "job_url": "", "letter": None,
             "letter_text": "", "tailor": None, "uploader_key": 0, "cv_text_used": ""}.items():
    ss_default(k, v)

storage = get_storage()
llm = get_llm()


def safe_filename(*parts: str) -> str:
    name = "_".join(p for p in parts if p) or "cover_letter"
    return re.sub(r"[^\w\-]+", "_", name).strip("_")[:80]


# =============================================================== sidebar
def sidebar() -> dict:
    health = llm.health()
    with st.sidebar:
        st.header("🎯 CV ↔ Job Matcher")
        st.caption("Runs 100% locally with Ollama.")
        if not health["ok"]:
            st.error("Ollama is not running.")
            st.caption(health["error"])
            st.markdown("Start the **Ollama** app, then reload this page.")
        else:
            st.markdown(f"{'✅' if health['chat_model'] else '⚠️'} Chat model: `{llm.chat_model}`")
            if not health["chat_model"]:
                st.code(f"ollama pull {llm.chat_model}", language="bash")
            st.markdown(f"{'✅' if health['embed_model'] else '⚠️'} Embeddings: `{llm.embed_model}`")
            if not health["embed_model"]:
                st.caption("Optional — without it the semantic check is skipped.")
                st.code(f"ollama pull {llm.embed_model}", language="bash")
        st.divider()
        st.markdown("**Score weights**")
        for k, w in config.DEFAULT_WEIGHTS.items():
            st.caption(f"{config.CATEGORY_LABELS[k]}: {int(w * 100)}%")
        st.caption(f"− {config.MUST_HAVE_PENALTY} per missing must-have (max {config.MUST_HAVE_PENALTY_MAX})")
        st.caption(f"Cover letter when score > {config.COVER_LETTER_THRESHOLD}")
    return health


# =============================================================== analyse page
def cv_section() -> str | None:
    st.subheader("1 · Your CV")
    active = storage.get_active_cv()
    if active:
        c1, c2 = st.columns([3, 1])
        c1.success(f"Using saved CV **{active['filename']}** (uploaded {active['uploaded_at'][:10]})")
        with c2.popover("Replace CV"):
            new = st.file_uploader("Upload a new CV (PDF)", type=["pdf"], key=f"up{st.session_state.uploader_key}")
            if new is not None:
                _store_upload(new)
        with st.expander("Show the text the app read from your CV"):
            st.text(active["text"][:6000])
        return active["text"]

    st.info("Upload your CV once — it will be remembered for every job you check.")
    new = st.file_uploader("CV as PDF", type=["pdf"], key=f"up{st.session_state.uploader_key}")
    if new is not None:
        _store_upload(new)
    return None


def _store_upload(file) -> None:
    try:
        text = extract_pdf_text(file.getvalue())
    except PDFTextError as exc:
        st.error(str(exc))
        return
    storage.save_cv(file.name, text)
    st.session_state.uploader_key += 1
    st.rerun()


def run_analysis(cv_text: str, jd_text: str, job_url: str) -> None:
    profile = storage.get_profile()
    with st.status("Analysing … (first run can take a minute while the model loads)", expanded=True) as status:
        try:
            result = analyse(llm, cv_text, jd_text, profile, progress=lambda m: status.write(m))
        except LLMError as exc:
            status.update(label="Analysis failed", state="error")
            st.error(str(exc))
            return
        status.update(label="Analysis complete", state="complete", expanded=False)
    st.session_state.update(analysis=result, jd_text=jd_text, job_url=job_url, letter=None,
                            letter_text="", tailor=None, cv_text_used=cv_text)
    st.session_state.app_id = storage.add_application(result, jd_text, job_url)


def render_score(a: dict) -> None:
    overall = a["overall"]
    verdict = ("Excellent match" if overall >= 4.3 else "Good match" if overall > config.COVER_LETTER_THRESHOLD
               else "Partial match" if overall >= 2.5 else "Weak match")
    job = a["job"]
    title = " · ".join(x for x in (job["job_title"], job["company"], job["location"]) if x) or "This job"
    st.subheader(f"2 · Result — {title}")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Overall match", f"{overall:.2f} / 5", verdict)
        st.caption(f"Weighted categories {a['weighted_score']:.2f} − must-have penalty {a['penalty']['points']:.2f}"
                   f" ({a['penalty']['missing_must_have']} missing)")
        if a["cover_letter_eligible"]:
            st.success("Above 3.5 → cover letter unlocked below.")
        else:
            st.warning("3.5 or below → no cover letter. See the gaps and CV tips.")
        if not a["semantic_available"]:
            st.caption("ℹ️ Semantic check skipped (embedding model not installed).")
    with c2:
        for key, cat in a["categories"].items():
            w = a["weights"][key]
            lc, rc = st.columns([3, 1])
            lc.markdown(f"**{config.CATEGORY_LABELS[key]}** · weight {int(w * 100)}%")
            rc.markdown(f"**{cat['score']:.2f}** / 5")
            st.progress(min(1.0, cat["score"] / 5))
            details = cat["reason"]
            if len(cat["parts"]) > 1:
                details += "  \n" + " · ".join(f"{k}: {v}" for k, v in cat["parts"].items())
            st.caption(details)


def render_details(a: dict) -> None:
    st.markdown("#### 🇩🇪 Germany checks")
    for f in a["germany"]["flags"]:
        FLAG_FN.get(f["level"], st.info)(f"**{f['title']}**" + (f"  \n{f['detail']}" if f["detail"] else ""))

    st.markdown("#### Requirements")
    rows = [{
        "Type": "Must-have" if r["type"] == "must" else "Nice-to-have",
        "Requirement": r["name"],
        "Status": STATUS_ICON[r["status"]],
        "Evidence from your CV": r["evidence"] + ("" if r["evidence_verified"] or not r["evidence"] else " ⚠️ not found in CV"),
        "Keyword": "✔" if r["keyword_found"] else "",
        "Similarity": r["similarity"],
        "Note": r["note"],
    } for r in a["requirements"]]
    if rows:
        df = pd.DataFrame(rows)
        if not a["semantic_available"]:
            df = df.drop(columns=["Similarity"])
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.caption("No requirements could be extracted from the ad.")

    kw = a["categories"]["keywords"]
    if kw.get("found") or kw.get("missing"):
        c1, c2 = st.columns(2)
        c1.markdown("**Keywords in your CV:** " + (", ".join(f"`{k}`" for k in kw["found"]) or "—"))
        c2.markdown("**Missing keywords:** " + (", ".join(f"`{k}`" for k in kw["missing"]) or "—"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 💪 Strengths")
        for s in a["strengths"]:
            st.markdown(f"- {s}")
    with c2:
        st.markdown("#### 🕳️ Gaps")
        for s in a["gaps"]:
            st.markdown(f"- {s}")
    with c3:
        st.markdown("#### 🛠️ CV tips")
        for s in a["cv_tips"]:
            st.markdown(f"- {s}")


def render_tailoring(a: dict, cv_text: str) -> None:
    st.subheader("3 · Tailor your CV for this job")
    st.caption("Rewrites of lines that are already in your CV — nothing is invented. "
               "Items marked ⚠️ add something you must confirm is true.")
    if st.button("✍️ Suggest CV rewrites", key="tailor_btn"):
        with st.spinner("Writing suggestions …"):
            try:
                st.session_state.tailor = tailor_cv(llm, cv_text, a)
            except LLMError as exc:
                st.error(str(exc))
    if st.session_state.tailor is not None:
        if not st.session_state.tailor:
            st.info("No suggestions returned — try again.")
        for i, s in enumerate(st.session_state.tailor, 1):
            with st.container(border=True):
                st.markdown(f"**{i}. Before:** {s['original']}"
                            + ("" if s["original_found_in_cv"] else "  _(line not found exactly in CV)_"))
                st.markdown(f"**After:** {s['suggested']}" + ("  ⚠️ **confirm this is true**" if s["needs_confirmation"] else ""))
                st.caption(s["why"])


def render_cover_letter(a: dict, cv_text: str) -> None:
    st.subheader("4 · Cover letter")
    profile = storage.get_profile()
    allowed = a["cover_letter_eligible"]
    if not allowed:
        st.info(f"Score is {a['overall']:.2f} — the cover letter is generated only above {config.COVER_LETTER_THRESHOLD}.")
        allowed = st.checkbox("Generate one anyway")
    if not allowed:
        return
    if not profile.get("full_name"):
        st.warning("Add your name and address in **My profile** so the letter header is complete.")

    lang = a["germany"]["job_language"]
    c1, c2, c3 = st.columns([1, 1, 2])
    tones = ["formal", "modern"]
    lengths = ["short", "standard", "detailed"]
    tone = c1.radio("Tone", tones, index=tones.index(profile.get("default_tone", "formal")), horizontal=True)
    length = c2.radio("Length", lengths, index=lengths.index(profile.get("default_length", "standard")), horizontal=True)
    extra = c3.text_input("Extra instructions (optional)",
                          placeholder="e.g. mention my thesis on computer vision; keep it humble")
    st.caption(f"Language: **{'German' if lang == 'de' else 'English'}** (same as the job ad)")

    target_words = LENGTHS[length][0]
    label = "🔁 Regenerate" if st.session_state.letter else "✉️ Generate cover letter"
    clicked = st.button(label, type="primary")
    fix_list = st.session_state.pop("pending_fix", None)
    if clicked or fix_list:
        instructions = extra
        if fix_list:
            instructions += (" Fix these problems from the previous draft: "
                             + " ".join(w.replace("**", "") for w in fix_list))
        with st.spinner("Writing your cover letter …"):
            try:
                letter = generate_cover_letter(llm, cv_text, st.session_state.jd_text, a, profile,
                                               tone=tone, length=length, extra_instructions=instructions)
            except LLMError as exc:
                st.error(str(exc))
                return
        st.session_state.letter = letter
        st.session_state.letter_text = letter_to_text(letter)
        st.session_state["letter_editor"] = st.session_state.letter_text
        if st.session_state.app_id:
            storage.update_application(st.session_state.app_id, cover_letter_json=letter)

    if not st.session_state.letter:
        return

    st.caption("Edit freely. Keep blank lines between blocks: subject / salutation / paragraphs / closing. "
               "Your name is added automatically under the closing.")
    edited = st.text_area("Your letter", key="letter_editor", height=420)
    letter = text_to_letter(edited, lang)
    if edited != st.session_state.letter_text:
        st.session_state.letter_text = edited
        st.session_state.letter = letter
        if st.session_state.app_id:
            storage.update_application(st.session_state.app_id, cover_letter_json=letter)
    words = len(" ".join(letter["paragraphs"]).split())
    st.caption(f"{words} words in the body (target ≈ {target_words})"
               + ("  ⚠️ quite long — consider 'short' or trimming" if words > target_words * 1.35 else ""))

    issues = fact_check(letter, cv_text, a, profile)
    if issues:
        with st.container(border=True):
            st.markdown("**🔎 Fact check — please review before sending:**")
            for w in issues:
                st.markdown(f"- ⚠️ {w}")
            st.button("🛠️ Regenerate and fix these issues", key="fix_btn",
                      on_click=lambda ws=issues: st.session_state.update(pending_fix=ws))
    else:
        st.success("🔎 Fact check passed: no claims found that your CV doesn't support.")

    missing_address = not (a["job"].get("company") and profile.get("full_name"))
    with st.expander("Recipient & date (for the DIN 5008 letter layout)", expanded=missing_address):
        if missing_address:
            st.caption("Fill in the company address so the address field of the letter isn't empty.")
        r1, r2 = st.columns(2)
        recipient = {
            "company": r1.text_input("Company", a["job"].get("company", "")),
            "contact": r2.text_input("Contact person", a["job"].get("contact_person", "")),
            "street": r1.text_input("Street & number", ""),
            "postal_city": r2.text_input("Postcode & city", a["job"].get("location", "")),
        }
        place = r1.text_input("Place before the date", profile.get("place", ""))
        date_str = r2.text_input("Date", format_date(lang))

    base = safe_filename("Anschreiben" if lang == "de" else "Cover_Letter", a["job"].get("company", ""))
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ DOCX", to_docx(letter, profile, recipient, place, date_str), f"{base}.docx",
                       "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    d2.download_button("⬇️ PDF", to_pdf(letter, profile, recipient, place, date_str), f"{base}.pdf", "application/pdf")
    d3.download_button("⬇️ TXT", edited, f"{base}.txt", "text/plain")


def page_analyse() -> None:
    st.title("Check a job against your CV")
    cv_text = cv_section()
    st.subheader("Job description")
    jd = st.text_area("Paste the full job ad", value=st.session_state.jd_text, height=260,
                      placeholder="Copy everything from the job posting: tasks, requirements, benefits …")
    url = st.text_input("Job link (optional, saved in your tracker)", value=st.session_state.job_url)
    if st.button("🔍 Analyse match", type="primary"):
        if not cv_text:
            st.error("Upload your CV first.")
        elif len(jd.strip()) < 200:
            st.error("Paste the full job description (at least a few sentences).")
        elif not HEALTH["ok"]:
            st.error("Ollama is not running — start the Ollama app and reload the page.")
        else:
            run_analysis(cv_text, jd.strip(), url.strip())

    a = st.session_state.analysis
    if a:
        st.divider()
        render_score(a)
        render_details(a)
        st.divider()
        used_cv = st.session_state.cv_text_used or cv_text or ""
        render_tailoring(a, used_cv)
        st.divider()
        render_cover_letter(a, used_cv)


# =============================================================== tracker page
def _clean(value) -> str:
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def page_tracker() -> None:
    st.title("📋 Application tracker")
    apps = storage.list_applications()
    if not apps:
        st.info("Every job you analyse is saved here automatically.")
        return

    df = pd.DataFrame(apps)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jobs analysed", len(df))
    c2.metric("Applied", int(df["status"].isin(["applied", "interview", "offer", "rejected"]).sum()))
    c3.metric("Interviews", int(df["status"].isin(["interview", "offer"]).sum()))
    c4.metric("Average score", f"{df['overall_score'].mean():.2f}")

    view = df[["id", "created_at", "company", "role", "location", "overall_score", "status", "notes", "has_letter",
               "job_url"]].copy()
    view["created_at"] = view["created_at"].str[:10]
    view["has_letter"] = view["has_letter"].astype(bool)
    edited = st.data_editor(
        view, hide_index=True, use_container_width=True, key="tracker_editor",
        disabled=["id", "created_at", "overall_score", "has_letter"],
        column_config={
            "id": st.column_config.NumberColumn("#", width="small"),
            "created_at": "Date", "company": "Company", "role": "Role", "location": "Location",
            "overall_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=5, format="%.2f"),
            "status": st.column_config.SelectboxColumn("Status", options=config.APPLICATION_STATUSES, required=True),
            "notes": st.column_config.TextColumn("Notes", width="medium"),
            "has_letter": st.column_config.CheckboxColumn("Letter"),
            "job_url": st.column_config.LinkColumn("Link"),
        })
    if st.button("💾 Save changes"):
        orig = view.set_index("id")
        for row in edited.to_dict("records"):
            before = orig.loc[row["id"]]
            changes = {k: _clean(row[k]) for k in ("company", "role", "location", "status", "notes", "job_url")
                       if _clean(row[k]) != _clean(before[k])}
            if changes:
                storage.update_application(int(row["id"]), **changes)
        st.success("Saved.")
        st.rerun()

    buf = io.StringIO()
    df.drop(columns=["has_letter"]).to_csv(buf, index=False)
    st.download_button("⬇️ Export CSV", buf.getvalue(), "applications.csv", "text/csv")

    st.divider()
    labels = {a["id"]: f"#{a['id']} · {a['company'] or '?'} — {a['role'] or '?'} ({a['overall_score']:.2f})" for a in apps}
    chosen = st.selectbox("Open an application", list(labels), format_func=labels.get)
    item = storage.get_application(chosen)
    if item:
        an = item["analysis"]
        with st.expander("Score breakdown", expanded=True):
            for k, c in an.get("categories", {}).items():
                st.markdown(f"- **{config.CATEGORY_LABELS.get(k, k)}**: {c['score']:.2f} — {c['reason']}")
            st.markdown(f"- **Overall: {an.get('overall', 0):.2f} / 5**")
        if item["cover_letter"]:
            with st.expander("Cover letter"):
                st.text(letter_to_text(item["cover_letter"]))
        b1, b2 = st.columns(2)
        if b1.button("↩️ Load into analyser"):
            st.session_state.update(analysis=an, app_id=item["id"], jd_text=item["jd_text"] or "",
                                    job_url=item["job_url"] or "", letter=item["cover_letter"], tailor=None,
                                    cv_text_used="")
            st.session_state.letter_text = letter_to_text(item["cover_letter"]) if item["cover_letter"] else ""
            st.session_state["letter_editor"] = st.session_state.letter_text
            st.switch_page(PAGES["analyse"])
        if b2.button("🗑️ Delete", type="secondary"):
            storage.delete_application(item["id"])
            st.rerun()


# =============================================================== profile page
def page_profile() -> None:
    st.title("👤 My profile")
    st.caption("Used for the letter header and the Germany checks. Stored only on your computer.")
    p = storage.get_profile()
    with st.form("profile"):
        c1, c2 = st.columns(2)
        p["full_name"] = c1.text_input("Full name", p["full_name"])
        p["email"] = c2.text_input("Email", p["email"])
        p["street"] = c1.text_input("Street & number", p["street"])
        p["phone"] = c2.text_input("Phone", p["phone"])
        p["postal_city"] = c1.text_input("Postcode & city", p["postal_city"])
        p["place"] = c2.text_input("Place before the date (e.g. Berlin)", p["place"])
        st.divider()
        levels = [""] + CEFR
        p["german_level"] = c1.selectbox("Your German level (CEFR)", levels,
                                         index=levels.index(p["german_level"]) if p["german_level"] in levels else 0,
                                         help="Leave empty to read it from your CV.")
        p["is_student"] = c2.checkbox("I am currently enrolled as a student", bool(p["is_student"]))
        p["other_job_hours"] = c1.number_input("Hours/week I already work elsewhere", 0, 40, int(p["other_job_hours"] or 0))
        p["availability"] = c2.text_input("Availability (for the letter)", p["availability"],
                                          placeholder="e.g. from 01.12.2026, 15–20 h/week")
        st.divider()
        tones, lengths = ["formal", "modern"], ["short", "standard", "detailed"]
        p["default_tone"] = c1.selectbox("Default letter tone", tones, index=tones.index(p["default_tone"]))
        p["default_length"] = c2.selectbox("Default letter length", lengths, index=lengths.index(p["default_length"]))
        if st.form_submit_button("Save profile", type="primary"):
            storage.save_profile(p)
            st.success("Profile saved.")

    st.subheader("Saved CV")
    active = storage.get_active_cv()
    if active:
        st.write(f"**{active['filename']}** · {len(active['text'])} characters · uploaded {active['uploaded_at'][:10]}")
        if st.button("Remove saved CV"):
            storage.clear_cv()
            st.rerun()
    else:
        st.caption("No CV saved yet — upload one on the Analyse page.")


HEALTH = sidebar()
START = os.getenv("CVM_START_PAGE", "analyse")  # used by the UI tests
PAGES = {
    "analyse": st.Page(page_analyse, title="Analyse", icon="🔍", url_path="analyse", default=START == "analyse"),
    "tracker": st.Page(page_tracker, title="Tracker", icon="📋", url_path="tracker", default=START == "tracker"),
    "profile": st.Page(page_profile, title="My profile", icon="👤", url_path="profile", default=START == "profile"),
}
st.navigation(list(PAGES.values())).run()
