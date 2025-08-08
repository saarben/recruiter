import os
import io
import re
import json
import time
import base64
from typing import List, Dict, Any
import tempfile

import streamlit as st
import pandas as pd

# Text extraction libs
from pypdf import PdfReader
import docx2txt
from pdfminer.high_level import extract_text as pdfminer_extract_text
import fitz  # PyMuPDF

# Optional: OpenAI-compatible client (OpenAI or Groq via base_url)
USE_LLM_DEFAULT = True
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_SDK_AVAILABLE = True
except Exception:
    GROQ_SDK_AVAILABLE = False

MODEL_DEFAULT = (
    os.getenv("CV_SCREENER_MODEL")
    or ("llama3-8b-8192" if os.getenv("GROQ_API_KEY") else "gpt-4o-mini")
)

st.set_page_config(page_title="CV Screener", page_icon="📄", layout="wide")

# ---------------------- Helpers ----------------------
PII_PATTERNS = [
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # email
    r"\b\+?\d[\d\s().-]{7,}\b",                      # phone-ish
    r"https?://\S+",                                    # urls
]

KEYWORD_DEFAULTS = [
    "python", "c++", "ros", "linux", "cuda", "ml", "cv", "docker", "kubernetes",
]

STOPWORDS = {
    "the", "and", "or", "for", "with", "you", "your", "our", "we", "they", "a", "an", "to",
    "in", "of", "on", "at", "as", "by", "is", "are", "be", "this", "that", "from", "will",
    "have", "has", "had", "it", "but", "if", "not", "can", "may", "must", "should", "more",
    "less", "about", "using", "use", "used", "preferred", "required", "responsibilities",
    "qualifications", "experience", "skills", "years", "year", "nice", "plus", "bonus",
}

SYMBOLIC_EQUIVALENTS: Dict[str, list[str]] = {
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    "node.js": ["node.js", "nodejs", "node"],
}

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "overall_score": {"type": "number"},
        "fit_reasoning": {"type": "string"},
        "pros": {"type": "array", "items": {"type": "string"}},
        "cons": {"type": "array", "items": {"type": "string"}},
        "hire_recommendation": {"type": "string", "enum": ["strong yes", "yes", "maybe", "no"]},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "years_experience": {"type": "number"}
    },
    "required": ["overall_score", "hire_recommendation"],
    "additionalProperties": True
}

# LLM client factory: supports OpenAI (default) or Groq SDK
def create_llm_client():
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key and GROQ_SDK_AVAILABLE:
        return Groq(api_key=groq_key)
    if OPENAI_AVAILABLE:
        return OpenAI()
    raise RuntimeError("No available LLM client. Install 'openai' or 'groq'.")

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file_bytes: bytes) -> str:
    # First try pypdf
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
                if txt:
                    text.append(txt)
            except Exception:
                pass
        combined = "\n".join(text).strip()
        if combined:
            return combined
    except Exception:
        pass
    # Fallback to pdfminer.six
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            mined = pdfminer_extract_text(tmp_path) or ""
            if mined.strip():
                return mined
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        pass
    # Fallback to PyMuPDF
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text("text") or "")
        text = "\n".join(parts).strip()
        return text
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def extract_text_from_docx(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return docx2txt.process(tmp_path) or ""
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def extract_text(upload) -> str:
    name = upload.name.lower()
    data = upload.read()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(data)
    if name.endswith(".docx"):
        return extract_text_from_docx(data)
    if name.endswith(".txt"):
        return data.decode(errors="ignore")
    raise ValueError("Unsupported file type. Use pdf, docx, or txt.")

def scrub_pii(text: str) -> str:
    scrubbed = text
    for pat in PII_PATTERNS:
        scrubbed = re.sub(pat, "[REDACTED]", scrubbed, flags=re.IGNORECASE)
    return scrubbed

def heuristic_score(text: str, keywords: List[str], job_desc: str) -> Dict[str, Any]:
    """Heuristic scoring that incorporates JD terms.

    - Extracts salient keywords from the job description
    - Counts unique JD term matches and default keyword hits in the resume
    - Combines JD matches, defaults, and years of experience into a 0-100 score
    """
    def tokenize_keywords(source_text: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9+\-#\.]{1,}", source_text.lower())
        filtered = [t for t in tokens if t not in STOPWORDS and len(t) >= 3]
        # frequency order, most common first
        freqs: Dict[str, int] = {}
        for t in filtered:
            freqs[t] = freqs.get(t, 0) + 1
        # keep top N distinct
        top = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in top[:30]]

    jd_keywords: List[str] = tokenize_keywords(job_desc) if job_desc else []
    combined_keywords: List[str] = []
    seen: set[str] = set()
    for k in (jd_keywords + keywords):
        if k not in seen:
            combined_keywords.append(k)
            seen.add(k)

    def count_hits(target_text: str, term: str) -> int:
        # Support symbolic equivalents and escape regex special chars
        candidates: list[str] = SYMBOLIC_EQUIVALENTS.get(term.lower(), [term])
        total = 0
        for cand in candidates:
            pattern = re.escape(cand)
            total += len(re.findall(rf"(?<!\w){pattern}(?!\w)", target_text, flags=re.IGNORECASE))
        return total

    default_hits_total = sum(count_hits(text, k) for k in keywords)
    jd_unique_hits = sum(1 for k in jd_keywords if count_hits(text, k) > 0)
    jd_total_hits = sum(count_hits(text, k) for k in jd_keywords)

    year_hits = re.findall(r"(\d{1,2})\+?\s*(?:years|yrs)", text, flags=re.IGNORECASE)
    years = max([int(y) for y in year_hits], default=0)

    # Weighted components: JD emphasis
    jd_component = min(50, jd_unique_hits * 5)  # up to 50
    default_component = min(30, default_hits_total * 3)  # up to 30
    years_component = min(20, years * 4)  # up to 20
    score = min(100, jd_component + default_component + years_component)

    rec = "no"
    if score >= 85:
        rec = "strong yes"
    elif score >= 70:
        rec = "yes"
    elif score >= 55:
        rec = "maybe"

    matched_jd_terms = [k for k in jd_keywords if count_hits(text, k) > 0][:10]

    return {
        "name": "(unknown)",
        "overall_score": score,
        "fit_reasoning": (
            f"JD unique term matches: {jd_unique_hits}/{len(jd_keywords)}; "
            f"JD total hits: {jd_total_hits}; Default keyword hits: {default_hits_total}; "
            f"Estimated years: {years}."
        ),
        "pros": [
            f"Matched JD terms: {', '.join(matched_jd_terms)}" if matched_jd_terms else "Some baseline keyword matches",
        ],
        "cons": ["Heuristic; may miss nuanced fit"],
        "hire_recommendation": rec,
        "risk_flags": [],
        "years_experience": years,
    }

def ensure_json(text: str) -> Dict[str, Any]:
    # Try to extract JSON from the model response even if there is extra text
    m = re.search(r"\{[\s\S]*\}$", text.strip())
    if m:
        text = m.group(0)
    return json.loads(text)

# LLM call using OpenAI-compatible client (OpenAI or Groq)
def call_openai(job_desc: str, resume_text: str, weights: Dict[str, int], model: str) -> Dict[str, Any]:
    if not OPENAI_AVAILABLE:
        # We'll still allow Groq path even if OpenAI is missing
        if not (os.getenv("GROQ_API_KEY") and GROQ_SDK_AVAILABLE):
            raise RuntimeError("OpenAI SDK not available. Install 'openai' or set GROQ_API_KEY.")
    rubric = {
        "impact_weight": weights.get("impact", 25),
        "skills_weight": weights.get("skills", 25),
        "experience_weight": weights.get("experience", 25),
        "culture_weight": weights.get("culture", 25),
    }

    system = (
        "You are an expert technical recruiter and hiring manager. "
        "Return ONLY valid JSON following the provided JSON Schema."
    )

    user = f"""
Evaluate the following candidate against this job description.

JOB DESCRIPTION:\n{job_desc}\n\nRESUME:\n{resume_text}\n\nScoring rubric weights (sum to 100):\n- Impact/Outcomes: {rubric['impact_weight']}\n- Skills/Tech match: {rubric['skills_weight']}\n- Relevant experience: {rubric['experience_weight']}\n- Culture/communication: {rubric['culture_weight']}\n
Return a JSON with keys: name (if present), overall_score (0-100), fit_reasoning, pros[], cons[], hire_recommendation (one of: strong yes, yes, maybe, no), risk_flags[], years_experience (estimate). Keep JSON concise.
"""

    try:
        client = create_llm_client()
        # If Groq client
        if isinstance(client, Groq):
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content
        else:
            # OpenAI client
            resp = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
            )
            raw = resp.choices[0].message.content
        return ensure_json(raw)
    except Exception as e:
        # Best-effort fallback: return a heuristic result
        return {
            **heuristic_score(resume_text, KEYWORD_DEFAULTS, job_desc),
            "fit_reasoning": f"LLM call failed ({e}); used heuristic.",
        }

# ---------------------- UI ----------------------
st.title("📄 CV Screener")

with st.sidebar:
    st.header("Settings")
    use_llm = st.toggle("Use LLM (OpenAI)", value=USE_LLM_DEFAULT)
    model = st.text_input("Model", MODEL_DEFAULT)
    pii = st.toggle("Scrub PII before evaluation", value=True)

    # Key/provider status
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    has_groq_key = bool(os.getenv("GROQ_API_KEY"))
    if use_llm:
        if has_groq_key and GROQ_SDK_AVAILABLE:
            st.markdown(f"✅ LLM provider: **Groq** (key detected) — Model: `{model}`")
        elif has_openai_key and OPENAI_AVAILABLE:
            st.markdown(f"✅ LLM provider: **OpenAI** (key detected) — Model: `{model}`")
        elif has_groq_key and not GROQ_SDK_AVAILABLE:
            st.markdown("⚠️ Groq key set, but `groq` SDK not installed. Running heuristic if LLM call fails.")
        elif has_openai_key and not OPENAI_AVAILABLE:
            st.markdown("⚠️ OpenAI key set, but `openai` SDK not installed. Running heuristic if LLM call fails.")
        else:
            st.markdown("❌ No LLM key found — using heuristic scoring.")

    st.subheader("Scoring Weights")
    impact = st.slider("Impact/Outcomes", 0, 100, 25)
    skills = st.slider("Skills/Tech match", 0, 100, 25)
    experience = st.slider("Relevant experience", 0, 100, 25)
    culture = st.slider("Culture/communication", 0, 100, 25)

    total = impact + skills + experience + culture
    if total != 100:
        st.info(f"Weights sum to {total}. They will be normalized to 100.")
    weights = {"impact": impact, "skills": skills, "experience": experience, "culture": culture}

st.subheader("Job Description")
job_desc = st.text_area("Paste the role description / must-haves / nice-to-haves", height=200)

uploads = st.file_uploader("Upload CVs (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

colA, colB = st.columns([1,1])
with colA:
    run = st.button("Score Candidates", type="primary")
with colB:
    clear = st.button("Clear")
    if clear:
        st.experimental_rerun()

results: List[Dict[str, Any]] = []

if run:
    if not job_desc:
        st.warning("Please paste a job description first.")
    elif not uploads:
        st.warning("Please upload at least one CV.")
    else:
        with st.spinner("Evaluating candidates…"):
            for upload in uploads:
                try:
                    raw_text = extract_text(upload)
                    resume_text = scrub_pii(raw_text) if pii else raw_text
                    if use_llm and ((OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY")) or (GROQ_SDK_AVAILABLE and os.getenv("GROQ_API_KEY"))):
                        r = call_openai(job_desc, resume_text, weights, model)
                    else:
                        r = heuristic_score(resume_text, KEYWORD_DEFAULTS, job_desc)
                    r["file"] = upload.name
                    results.append(r)
                except Exception as e:
                    results.append({
                        "file": upload.name,
                        "overall_score": 0,
                        "hire_recommendation": "no",
                        "fit_reasoning": f"Failed to parse CV: {e}",
                        "pros": [],
                        "cons": ["Parse error"],
                        "risk_flags": ["unreadable"],
                        "name": "(unknown)",
                        "years_experience": 0,
                    })

if results:
    df = pd.DataFrame(results)
    # Rank by score desc
    df_sorted = df.sort_values(by="overall_score", ascending=False, kind="mergesort").reset_index(drop=True)

    st.subheader("Results")
    st.dataframe(df_sorted[["file", "name", "overall_score", "hire_recommendation", "years_experience"]], use_container_width=True)

    top = df_sorted.iloc[0]
    st.markdown(f"**Top candidate:** {top.get('name') or '(unknown)'} — **Score:** {int(top['overall_score'])} — **Rec:** {top['hire_recommendation']}")

    with st.expander("Details per candidate"):
        for i, row in df_sorted.iterrows():
            with st.container(border=True):
                st.markdown(f"### {row.get('name') or '(unknown)'} — {row['file']}")
                st.markdown(f"**Score:** {int(row['overall_score'])} | **Recommendation:** {row['hire_recommendation']} | **Years exp (est):** {row.get('years_experience', 0)}")
                if row.get("pros"):
                    st.markdown("**Pros:**")
                    st.write(row["pros"])
                if row.get("cons"):
                    st.markdown("**Cons:**")
                    st.write(row["cons"])
                if row.get("risk_flags"):
                    st.markdown("**Risk flags:**")
                    st.write(row["risk_flags"])
                st.markdown("**Reasoning:**")
                st.write(row.get("fit_reasoning", ""))

    # CSV Download
    csv = df_sorted.to_csv(index=False).encode()
    st.download_button("Download CSV", data=csv, file_name="cv_scores.csv", mime="text/csv")

st.caption("Tip: Add role-specific keywords to the heuristic, or keep LLM enabled for nuanced scoring.")


