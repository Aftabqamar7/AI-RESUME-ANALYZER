import streamlit as st
import json
from resume_analyzer import ResumeAnalyzerEngine

st.set_page_config(
    page_title="AI Resume Analyzer & ATS Simulator",
    page_icon="📄",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 AI Resume Analyzer & ATS Matcher")
st.caption("Upload your CV and Job Description to get deterministic scoring and Groq AI insights.")

# API Key Handling (Reads from st.secrets or sidebar input)
api_key = st.secrets.get("GROQ_API_KEY", None)
if not api_key:
    api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
    st.sidebar.info("Get your API key from https://console.groq.com")

if not api_key:
    st.warning("Please provide a valid Groq API Key in `.streamlit/secrets.toml` or via sidebar to continue.")
    st.stop()

analyzer = ResumeAnalyzerEngine(api_key=api_key)

# Input Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader(
        "Supported formats: PDF, PNG, JPG, JPEG", 
        type=["pdf", "png", "jpg", "jpeg"]
    )

with col2:
    st.subheader("2. Job Description")
    job_description = st.text_area(
        "Paste the Job Description here...", 
        height=180
    )

# Execution Action
if st.button("🚀 Analyze Resume Match", type="primary", use_container_width=True):
    if not uploaded_file or not job_description.strip():
        st.error("Please provide both a Resume file and a Job Description.")
    else:
        with st.spinner("Processing document and running analysis..."):
            try:
                # Step 1: Text Extraction & Parsing
                raw_resume_text = analyzer.extract_text_from_file(uploaded_file)
                parsed_resume = analyzer.parse_resume(raw_resume_text)
                parsed_jd = analyzer.parse_job_description(job_description)

                # Step 2: Scoring Engine
                results = analyzer.calculate_matches_and_scores(
                    parsed_resume, 
                    parsed_jd, 
                    raw_resume_text
                )

                # Step 3: AI Recommendations
                recommendations = analyzer.generate_recommendations(
                    results["missing_skills"], 
                    results
                )

                st.divider()

                # Dashboard Output
                st.header("📊 Match Analysis Dashboard")

                # Main Score Banner
                score_col1, score_col2 = st.columns([1, 2])
                with score_col1:
                    score = results["overall_score"]
                    st.metric(label="Overall Match Score", value=f"{score} / 100")
                    if score >= 80:
                        st.success("Verdict: Strong Match")
                    elif score >= 60:
                        st.warning("Verdict: Moderate Match")
                    else:
                        st.error("Verdict: Low Match")

                with score_col2:
                    st.write("**Score Breakdown**")
                    st.progress(results["skill_score"] / 100, text=f"Skills Match: {results['skill_score']}%")
                    st.progress(results["exp_score"] / 100, text=f"Experience Match: {results['exp_score']}%")
                    st.progress(results["keyword_score"] / 100, text=f"Keyword Coverage: {results['keyword_score']}%")
                    st.progress(results["ats_score"] / 100, text=f"ATS Compatibility: {results['ats_score']}%")

                st.divider()

                # Detailed Metrics Split
                d_col1, d_col2 = st.columns(2)

                with d_col1:
                    st.subheader("🎯 Skills Breakdown")
                    st.write("**Matched Skills:**")
                    if results["matched_skills"]:
                        st.write(", ".join([f"`{s.title()}`" for s in results["matched_skills"]]))
                    else:
                        st.caption("No direct skills matched.")

                    st.write("**Missing Required Skills:**")
                    if results["missing_skills"]:
                        for ms in results["missing_skills"]:
                            st.markdown(f"- ❌ `{ms.title()}`")
                    else:
                        st.caption("No missing required skills!")

                with d_col2:
                    st.subheader("💼 Experience & ATS Flags")
                    st.write(f"- **Detected Work Experience:** {results['detected_experience']} years")
                    st.write(f"- **JD Required Experience:** {results['required_experience']} years")
                    
                    st.write("**ATS Compatibility Checks:**")
                    for check, passed in results["ats_checks"].items():
                        status = "✓" if passed else "⚠"
                        st.write(f"- {status} {check.replace('_', ' ').title()}")

                st.divider()

                # Recommendations
                st.subheader("💡 Top AI Recommendations")
                for idx, rec in enumerate(recommendations.get("recommendations", []), 1):
                    st.write(f"**{idx}.** {rec}")

                # JSON Export Option
                st.divider()
                export_data = {
                    "parsed_resume": parsed_resume,
                    "parsed_jd": parsed_jd,
                    "results": results,
                    "recommendations": recommendations
                }
                st.download_button(
                    label="📥 Download Analysis Report (JSON)",
                    data=json.dumps(export_data, indent=2),
                    file_name="resume_analysis_report.json",
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"An error occurred during analysis: {str(e)}")