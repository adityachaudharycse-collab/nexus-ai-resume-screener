# 🚀 NEXUS AI: Automated Resume Screener & ATS Score Matcher

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.38%2B-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-NLP-F7931E?logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-green)

**NEXUS AI** is an automated, NLP-powered **Resume Screener and Applicant Tracking System (ATS) Score Matcher** built for students, job seekers, and recruiters. It evaluates semantic compatibility between a candidate's PDF resume and a target Job Description (JD) in under 1.5 seconds, providing an interpretable ATS score alongside matched and missing skill recommendations.

> **Academic Project:** B.Tech (CSE – Artificial Intelligence & Machine Learning) Mini Project  
> **Institution:** KCC Institute of Technology and Management (CC 492), Greater Noida  

---

## ✨ Key Features

- **📄 Multi-Page PDF Parsing:** Extracts clean text streams from uploaded PDF resumes using `PyPDF` with graceful error handling for scanned image-only PDFs.
- **🧹 Intelligent NLP Preprocessing:** Normalizes text using lowercase conversion, regular expressions (preserving domain symbols like `C++` and `C#`), and `NLTK` English stopword filtering.
- **📊 Hybrid ATS Scoring Engine:** Combines **TF-IDF Cosine Similarity (60% weight)** with **Domain Skill Match Ratio (40% weight)** to eliminate document-length bias and deliver a calibrated `0–99%` ATS match score.
- **🎯 Skill Gap Diagnostics:** Automatically highlights **Matched Competencies** (green pills) and **Missing Target Keywords** (red pills) to help candidates optimize their resumes.
- **🔒 Privacy-First Architecture:** Resumes are processed strictly in volatile memory (RAM) with zero disk storage.

---

## 🏗️ System Architecture & Pipeline

```text
[Input: Resume PDF + Job Description Text]
                    │
                    ▼
[Stage 1: PyPDF Multi-Page Text Extraction]
                    │
                    ▼
[Stage 2: NLP Sanitization (Regex + NLTK Stopwords)]
            /                       \
           ▼                         ▼
[Stage 3A: TF-IDF Vectorizer]   [Stage 3B: Regex Skill Extractor]
           │                         │
           ▼                         ▼
[Cosine Similarity (60%)]       [Skill Match Ratio (40%)]
            \                       /
             ▼                     ▼
   [Stage 4: Hybrid ATS Score & Streamlit Dashboard]
```

---

## 📐 Mathematical Formulation

1. **Cosine Similarity:** Measures the angular alignment between Resume vector $A$ and Job Description vector $B$:
   $$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

2. **Hybrid ATS Score:**
   $$\text{ATS Score (\%)} = \min\Big(\text{round}\big((0.60 \times \text{Cosine\_Sim} + 0.40 \times \text{Skill\_Match\_Ratio}) \times 135,\, 2\big),\, 99.0\Big)$$

---

## 🛠️ Tech Stack

- **Frontend UI:** Streamlit (Custom Dark-Glass Telemetry Theme)
- **Language:** Python 3.10+
- **Machine Learning / Vectorization:** Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`)
- **Natural Language Processing:** NLTK (`stopwords`), Python `re`
- **PDF Extraction:** `pypdf`

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/adityachaudharycse-collab/nexus-ai-resume-screener.git
   cd nexus-ai-resume-screener
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

