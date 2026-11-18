import streamlit as st
from pypdf import PdfReader
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK stopwords silently
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# Domain Skills Knowledge Base
COMMON_SKILLS = {
    "python", "java", "c++", "c", "sql", "javascript", "html", "css", "react",
    "node", "machine learning", "deep learning", "nlp", "data science", "pandas",
    "numpy", "scikit-learn", "tensorflow", "pytorch", "opencv", "git", "github",
    "docker", "aws", "azure", "excel", "power bi", "tableau", "communication",
    "leadership", "problem solving", "teamwork", "flask", "django", "fastapi",
    "mongodb", "mysql", "postgresql", "rest api", "data analysis", "statistics",
    "artificial intelligence", "neural networks", "llm", "generative ai", "api"
}

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + " "
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\+\#]', ' ', text)
    words = [word for word in text.split() if word not in STOPWORDS and len(word) > 1]
    return " ".join(words)

def extract_skills(text):
    text_lower = text.lower()
    found_skills = set()
    for skill in COMMON_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    return found_skills

def calculate_ats_score(resume_clean, jd_clean, resume_skills, jd_skills):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    
    if len(jd_skills) > 0:
        skill_match_ratio = len(resume_skills.intersection(jd_skills)) / len(jd_skills)
    else:
        skill_match_ratio = cosine_sim

    final_score = (0.6 * cosine_sim + 0.4 * skill_match_ratio) * 100
    scaled_score = min(round(final_score * 1.35, 2), 99.0)
    return scaled_score, round(cosine_sim * 100, 2)

# ---------------- PAGE CONFIG & FUTURISTIC CSS ----------------
st.set_page_config(page_title="NEXUS ATS | AI Resume Intelligence", page_icon="⚡", layout="wide")

st.markdown("""
<style>
    /* Main Dark Cyber Background */
    .stApp {
        background: radial-gradient(circle at top right, #0f172a, #050811 70%);
        color: #e2e8f0;
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
    }
    
    /* Futuristic Header */
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00f5d4, #00bbf9, #9b5de5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 0px;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(0, 245, 212, 0.2);
        border-radius: 16px;
        padding: 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    }
    
    /* Metric Boxes */
    .metric-box {
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.9), rgba(30, 41, 59, 0.6));
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 187, 249, 0.12);
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #00f5d4;
        text-shadow: 0 0 15px rgba(0, 245, 212, 0.4);
    }

    /* Skill Pills */
    .skill-match {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }
    .skill-miss {
        display: inline-block;
        background: rgba(244, 63, 94, 0.15);
        color: #fb7185;
        border: 1px solid rgba(251, 113, 133, 0.4);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 4px;
    }

    /* Glowing Button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00f5d4, #00bbf9);
        color: #050811;
        font-weight: 700;
        font-size: 1.05rem;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.5rem;
        box-shadow: 0 0 20px rgba(0, 245, 212, 0.4);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        box-shadow: 0 0 30px rgba(0, 245, 212, 0.8);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ⚡ NEXUS AI ENGINE")
    st.caption("NLP & Vector Space Resume Screener")
    st.divider()
    st.markdown("**Core Architecture:**")
    st.markdown("- 🔹 `PyPDF` Text Parser")
    st.markdown("- 🔹 `NLTK` Token & Stopword Filter")
    st.markdown("- 🔹 `TF-IDF` Vectorizer")
    st.markdown("- 🔹 `Cosine Similarity` Matrix")
    st.divider()
    st.info("Upload a candidate resume in PDF format and paste the target Job Description to run semantic ATS diagnostics.")

# ---------------- HERO HEADER ----------------
st.markdown('<div class="hero-title">⚡ NEXUS AI — Resume Intelligence & ATS Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Next-Gen Semantic Candidate Screening powered by TF-IDF Vectorization & Cosine Similarity</div>', unsafe_allow_html=True)

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="glass-card"><h4>📂 01. Candidate Resume (PDF)</h4></div>', unsafe_allow_html=True)
    uploaded_resume = st.file_uploader("Drop PDF Resume Here", type=["pdf"], label_visibility="collapsed")

with col2:
    st.markdown('<div class="glass-card"><h4>🎯 02. Target Job Description</h4></div>', unsafe_allow_html=True)
    job_description = st.text_area("Paste Job Description", placeholder="Paste target role requirements, skills, and qualifications here...", height=160, label_visibility="collapsed")

st.write("")
analyze_btn = st.button("⚡ INITIATE AI ATS SCAN", use_container_width=True)

# ---------------- ANALYSIS & DASHBOARD ----------------
if analyze_btn:
    if uploaded_resume is not None and job_description.strip() != "":
        with st.spinner("Running NLP Vectorization & Semantic Similarity Matrix..."):
            raw_resume_text = extract_text_from_pdf(uploaded_resume)
            
            if not raw_resume_text.strip():
                st.error("Extraction Failed: Unable to parse text from this PDF. Please upload a text-encoded PDF document.")
            else:
                clean_resume = clean_text(raw_resume_text)
                clean_jd = clean_text(job_description)

                resume_skills = extract_skills(raw_resume_text)
                jd_skills = extract_skills(job_description)

                matched_skills = sorted(list(resume_skills.intersection(jd_skills)))
                missing_skills = sorted(list(jd_skills - resume_skills))

                ats_score, raw_cosine = calculate_ats_score(clean_resume, clean_jd, resume_skills, jd_skills)

                st.divider()
                st.markdown("### 📊 DIAGNOSTIC TELEMETRY REPORT")

                # Top 3 Futuristic Metric Cards
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Overall ATS Compatibility</div>
                        <div class="metric-value">{ats_score}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">TF-IDF Cosine Similarity</div>
                        <div class="metric-value" style="color:#00bbf9;">{raw_cosine}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-label">Skill Vector Alignment</div>
                        <div class="metric-value" style="color:#c084fc;">{len(matched_skills)} / {len(jd_skills)}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                st.progress(int(ats_score))

                # Verdict Banner
                if ats_score >= 75:
                    st.success("STRONG ALIGNMENT: Candidate profile exceeds ATS threshold for this role.")
                elif ats_score >= 50:
                    st.warning("MODERATE ALIGNMENT: Core competencies detected, but adding missing keywords will optimize ranking.")
                else:
                    st.error("LOW ALIGNMENT: Significant skill gap detected between resume and target job description.")

                # Skill Gap Pills
                c1, c2 = st.columns(2, gap="large")
                with c1:
                    st.markdown('<div class="glass-card"><h4>✅ Matched Competencies</h4>', unsafe_allow_html=True)
                    if matched_skills:
                        pills_html = "".join([f'<span class="skill-match">{s.upper()}</span>' for s in matched_skills])
                        st.markdown(pills_html, unsafe_allow_html=True)
                    else:
                        st.caption("No direct keyword overlap detected.")
                    st.markdown('</div>', unsafe_allow_html=True)

                with c2:
                    st.markdown('<div class="glass-card"><h4>⚠️ Missing Keywords (Optimization Targets)</h4>', unsafe_allow_html=True)
                    if missing_skills:
                        miss_html = "".join([f'<span class="skill-miss">{s.upper()}</span>' for s in missing_skills])
                        st.markdown(miss_html, unsafe_allow_html=True)
                    else:
                        st.caption("All target keywords are present in the resume!")
                    st.markdown('</div>', unsafe_allow_html=True)

                with st.expander("🧬 Inspect Processed NLP Token Stream"):
                    st.code(clean_resume[:1200] + ("..." if len(clean_resume) > 1200 else ""), language="text")
    else:
        st.warning("Input Required: Please upload a PDF resume and enter a Job Description before initiating scan.")