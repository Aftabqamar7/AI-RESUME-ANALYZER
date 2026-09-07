import os
import re
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from pypdf import PdfReader
from PIL import Image
import pytesseract
from groq import Groq

# ---------------------------------------------------------------------------
# Pydantic Schemas for Groq Structured Output
# ---------------------------------------------------------------------------

class ResumeSchema(BaseModel):
    candidate_name: Optional[str] = Field(description="Full name of candidate")
    skills: List[str] = Field(description="List of detected technical and soft skills")
    total_experience_years: float = Field(description="Total years of work experience detected")
    education_level: Optional[str] = Field(description="Highest degree attained e.g. Bachelor, Master, PhD")
    section_headings: List[str] = Field(description="Headings found in CV e.g. Experience, Education")
    has_contact_info: bool = Field(description="True if phone/email/linkedin is present")

class JobDescriptionSchema(BaseModel):
    job_title: Optional[str] = Field(description="Job title")
    required_skills: List[str] = Field(description="Skills required for the job")
    preferred_skills: List[str] = Field(description="Nice to have or preferred skills")
    required_experience_years: float = Field(description="Minimum required experience years")
    required_education_level: Optional[str] = Field(description="Required education degree level")
    keywords: List[str] = Field(description="Important domain keywords and buzzwords")

class AI AdviceSchema(BaseModel):
    recommendations: List[str] = Field(description="Top actionable improvements for the CV")
    rewritten_bullet_suggestions: List[str] = Field(description="Suggested improved bullet points")

# ---------------------------------------------------------------------------
# Core Analyzer Class
# ---------------------------------------------------------------------------

class ResumeAnalyzerEngine:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def extract_text_from_file(self, uploaded_file) -> str:
        file_type = uploaded_file.type
        text = ""

        if "pdf" in file_type:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            
            # OCR Fallback for scanned/image PDFs
            if len(text.strip()) < 50:
                uploaded_file.seek(0)
                # Fallback message if OCR fails or is empty
                text = "[OCR Processing required for scanned PDF content]"
                
        elif "image" in file_type or file_type in ["image/png", "image/jpeg", "image/jpg"]:
            image = Image.open(uploaded_file)
            text = pytesseract.image_to_string(image)

        return text.strip()

    def parse_resume(self, text: str) -> Dict[str, Any]:
        prompt = f"Extract structured data from the following resume text:\n\n{text}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)

    def parse_job_description(self, text: str) -> Dict[str, Any]:
        prompt = f"Extract structured requirements from this job description:\n\n{text}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)

    def generate_recommendations(self, missing_skills: List[str], score_breakdown: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
        Based on the evaluation metrics below, generate top 4 actionable resume improvements.
        Missing Skills: {missing_skills}
        Scores: {score_breakdown}
        
        Provide response as JSON with key 'recommendations' (list of strings).
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        return json.loads(response.choices[0].message.content)

    def calculate_matches_and_scores(self, resume_data: Dict[str, Any], jd_data: Dict[str, Any], raw_cv_text: str) -> Dict[str, Any]:
        # Normalization and Skills Match
        cv_skills = set(s.lower() for s in resume_data.get("skills", []))
        jd_req_skills = set(s.lower() for s in jd_data.get("required_skills", []))
        
        matched_skills = list(cv_skills.intersection(jd_req_skills))
        missing_skills = list(jd_req_skills - cv_skills)
        
        skill_score = (len(matched_skills) / len(jd_req_skills) * 100) if jd_req_skills else 100.0

        # Experience Match Calculation
        cv_exp = float(resume_data.get("total_experience_years", 0))
        jd_exp = float(jd_data.get("required_experience_years", 0))
        if jd_exp == 0:
            exp_score = 100.0
        else:
            exp_score = min(100.0, (cv_exp / jd_exp) * 100)

        # Keyword Match
        jd_keywords = set(k.lower() for k in jd_data.get("keywords", []))
        cv_text_lower = raw_cv_text.lower()
        matched_keywords = [k for k in jd_keywords if k in cv_text_lower]
        keyword_score = (len(matched_keywords) / len(jd_keywords) * 100) if jd_keywords else 100.0

        # Deterministic ATS Compatibility Evaluation
        ats_checks = {
            "has_contact_info": bool(resume_data.get("has_contact_info", True)),
            "standard_sections": len(resume_data.get("section_headings", [])) >= 3,
            "readable_length": 100 < len(raw_cv_text) < 15000,
            "no_unreadable_chars": not bool(re.search(r'[^\x00-\x7F]+', raw_cv_text[:500]))
        }
        ats_score = (sum(ats_checks.values()) / len(ats_checks)) * 100

        # Weighted Overall Score Formula
        overall_score = round(
            (skill_score * 0.35) +
            (exp_score * 0.25) +
            (keyword_score * 0.20) +
            (ats_score * 0.20),
            1
        )

        return {
            "overall_score": overall_score,
            "skill_score": round(skill_score, 1),
            "exp_score": round(exp_score, 1),
            "keyword_score": round(keyword_score, 1),
            "ats_score": round(ats_score, 1),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "detected_experience": cv_exp,
            "required_experience": jd_exp,
            "ats_checks": ats_checks
        }