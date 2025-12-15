# %%
# Import Required Libraries
from anthropic import Anthropic
import json
import yaml
import re
from datetime import datetime
import os
from dotenv import load_dotenv
import textwrap

# Data validation & handling
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator
from pathlib import Path

# Word Document
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_PARAGRAPH_ALIGNMENT, WD_BREAK
from docx.enum.text import WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

# PDF
from docx2pdf import convert
import pythoncom

# BrightData script (for job scraping)
from src.integrations.brightdata import extract_job

# Vector store and embeddings
from src.core.embeddings import OpenAIEmbeddings
from src.core.vector_store import QdrantVectorStore
from collections import defaultdict

# %%
# Load Environment Variables
load_dotenv()

key = os.getenv("CLAUDE_API_KEY")
if not key:
    raise ValueError("CLAUDE_API_KEY environment variable is required.")

model_name = "claude-haiku-4-5"

# %%
# Job Analysis Functions
def create_analysis_prompt(job_title: str, company: str, job_description: str) -> str:
    prompt_analysis = f"""
    You are an experienced hiring manager tasked with analyzing job descriptions to extract key information. Your goal is to provide accurate and relevant data that can be used in the recruitment process.

    Here is the job information you need to analyze:

    <job_description>
    {job_description}
    </job_description>

    <job_title>
    {job_title}
    </job_title>

    <company>
    {company}
    </company>

    Your task is to extract the following information from the job details provided:

    1. technical_skills: An array of technical skills required or preferred for the role. Look for specific technologies, programming languages, tools, or technical methodologies mentioned.
    2. soft_skills: An array of soft skills or personal attributes required or preferred for the role. Identify personal attributes, interpersonal skills, or work style preferences described.
    3. keywords: An array of keywords relevant to the role, including industry-specific terms, tools, or methodologies. Extract key terms that are frequently mentioned or seem particularly important to the role or industry.

    Output the information into a JSON object only with the following structure:

    - "technical_skills": ["skill1", "skill2", "skill3"],
    - "soft_skills": ["skill1", "skill2", "skill3"],
    - "keywords": ["keyword1", "keyword2", "keyword3"]

    Each array should contain at least 3 items if possible and a maximum of 10, but don't include irrelevant information just to meet this number.

    Remember to focus on accuracy and relevance in your analysis and output.
    """
    return prompt_analysis

# Job Analysis Response Data Validation
class JobAnalysisResponse(BaseModel):
    """Model for the job analysis response structure."""
    technical_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    
    @model_validator(mode='before')
    @classmethod
    def extract_from_text(cls, data):
        """Extract JSON from text if the input is not already a dict."""
        if isinstance(data, dict):
            return data
            
        if isinstance(data, str):
            # Try direct JSON parsing first
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                # Extract JSON using regex if direct parsing fails
                try:
                    json_pattern = r'(\{[\s\S]*\})'
                    match = re.search(json_pattern, data)
                    if match:
                        return json.loads(match.group(1))
                except:
                    pass
                
                # If we can't extract valid JSON, try to create a structured response
                # by looking for keyword patterns
                skills_pattern = {
                    "technical_skills": r'"?technical_skills"?\s*:\s*\[(.*?)\]',
                    "soft_skills": r'"?soft_skills"?\s*:\s*\[(.*?)\]',
                    "keywords": r'"?keywords"?\s*:\s*\[(.*?)\]'
                }
                
                extracted_data = {}
                for key, pattern in skills_pattern.items():
                    match = re.search(pattern, data, re.DOTALL)
                    if match:
                        # Extract the items in the array
                        items_text = match.group(1)
                        # Parse the items, handling quoted strings
                        items = []
                        for item in re.findall(r'"([^"]*)"', items_text):
                            items.append(item)
                        if items:
                            extracted_data[key] = items
                
                if extracted_data:
                    return extracted_data
                    
                # If all else fails, print a warning and return empty lists
                print(f"Failed to parse analysis response: {data[:100]}...")
                return {
                    "technical_skills": [],
                    "soft_skills": [],
                    "keywords": []
                }
        
        return data
    

def claude_analysis(api_key: str, prompt_analysis: str) -> Dict[str, Any]:
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model_name,
        max_tokens=2048,
        temperature=0.5,
        messages=[{"role": "user", "content": prompt_analysis}]
    )
    # Use Pydantic model to validate and extract the response
    response_text = response.content[0].text
    try:
        analysis_data = JobAnalysisResponse.model_validate(response_text)
        return analysis_data.model_dump()
    except Exception as e:
        print(f"Error parsing response with Pydantic: {e}")
        # Fallback to simpler parsing if Pydantic validation fails
        try:
            return json.loads(response_text)
        except:
            print(f"JSON parsing failed. Raw response: {response_text[:150]}...")
            # Last resort: return empty data structure
            return {
                "technical_skills": [],
                "soft_skills": [],
                "keywords": []
            }

def format_claude_analysis_response(skills_dict):
    formatted_text = ""    
    for category, items in skills_dict.items():
        # Convert category from snake_case to Title Case
        category_title = ' '.join(word.capitalize() for word in category.split('_'))
        # Section header
        formatted_text += f"## {category_title}\n"
        # Add bullet points for each item
        for item in items:
            formatted_text += f"• {item}\n"
        # Add extra line between sections
        formatted_text += "\n"
    return formatted_text

# %%
# Resume Generation Functions

# def load_resume_yaml(file_path: str) -> Dict[Any, Any]:
#     with open(file_path, 'r') as file:
#         resume_data = yaml.safe_load(file)
#     return resume_data


def retrieve_resume_context(
    job_title: Optional[str] = None, 
    company: Optional[str] = None, 
    job_description: Optional[str] = None, 
    top_k_achievements: int = 10,
    top_k_jobs: int = 3
) -> Dict[str, Any]:
    """
    Retrieve relevant resume context using RAG from vector database.
    
    This function queries the Qdrant vector store to retrieve the most relevant
    resume information based on job requirements. It supports two modes:
    1. Full retrieval (when job info is None): Returns all resume data
    2. Filtered retrieval (when job info is provided): Returns only relevant chunks
    
    Strategy:
    - Retrieves a pool of relevant achievements (top_k_achievements)
    - Groups achievements by job (company + position + dates)
    - Ranks jobs by average similarity score
    - Selects top N most relevant jobs (top_k_jobs)
    - Returns ALL achievements for the selected jobs
    
    Parameters
    ----------
    job_title : str, optional
        Job title to search for relevant experience.
    company : str, optional
        Company name (context for search).
    job_description : str, optional
        Full job description text for semantic search.
    top_k_achievements : int, optional
        Number of achievements to retrieve for ranking (default: 10).
    top_k_jobs : int, optional
        Number of most relevant jobs to include (default: 3).
    
    Returns
    -------
    Dict[str, Any]
        Reconstructed resume data matching YAML structure:
        {
            "personal_information": {...},
            "professional_summary": {...},
            "work_experience": [{company, position, dates, achievements}, ...],
            "education": [{degree, institution, dates}, ...],
            "continuing_studies": [{name, institution, completion_date}, ...],
            "skills": {programming_languages: [...], ...}
        }
    
    Notes
    -----
    - Ranks jobs by average similarity score of their achievements
    - Returns ALL achievements for top-ranked jobs (no filtering)
    - Uses semantic search on full job information for better matching
    
    Examples
    --------
    >>> # Full retrieval (initial pass)
    >>> resume = retrieve_resume_context()
    >>> 
    >>> # Filtered retrieval using job information
    >>> resume = retrieve_resume_context(
    ...     job_title="Data Scientist",
    ...     company="Acme Corp",
    ...     job_description="We need Python expertise for ETL pipelines...",
    ...     top_k_achievements=10,
    ...     top_k_jobs=3
    ... )
    """
    # Debug: Print parameters received
    print(f"\n🔍 DEBUG - retrieve_resume_context called with:")
    print(f"   job_title: {job_title}")
    print(f"   company: {company}")
    print(f"   job_description: {job_description[:100] if job_description else None}...")
    print()
    
    # Initialize embeddings and vector store
    embedder = OpenAIEmbeddings()
    store = QdrantVectorStore()
    
    # Construct query for filtered retrieval using full job information
    if job_title or company or job_description:
        query_parts = []
        if job_title:
            query_parts.append(job_title)
        if company:
            query_parts.append(company)
        if job_description:
            query_parts.append(job_description)
        
        query_text = ' '.join(query_parts)
        query_vector = embedder.embed_query(query_text)
        use_filtering = True
    else:
        # For full retrieval, use a generic query
        query_text = "data science analytics machine learning"
        query_vector = embedder.embed_query(query_text)
        use_filtering = False
    
    # Retrieve different sections with appropriate limits
    
    # Work experience achievements (filtered by relevance)
    work_results = store.search(
        query_vector=query_vector,
        top_k=top_k_achievements if use_filtering else 100,  # Get all if no filter
        section_filter="work_experience"
    )
    
    # Education (get all)
    education_results = store.search(
        query_vector=query_vector,
        top_k=10,
        section_filter="education"
    )
    
    # Skills (get all)
    skills_results = store.search(
        query_vector=query_vector,
        top_k=20,
        section_filter="skills"
    )
    
    # Continuing studies
    continuing_results = store.search(
        query_vector=query_vector,
        top_k=10 if use_filtering else 100,
        section_filter="continuing_studies"
    )
    
    # Personal info (get the one entry)
    personal_results = store.search(
        query_vector=query_vector,
        top_k=1,
        section_filter="personal_info"
    )
    
    # Professional summary (get the one entry)
    summary_results = store.search(
        query_vector=query_vector,
        top_k=1,
        section_filter="professional_summary"
    )
    
    # Reconstruct resume structure
    resume_data = {
        "personal_information": {},
        "professional_summary": {"personality_traits": []},
        "work_experience": [],
        "education": {"degrees": [], "continuing_studies": []},
        "skills": {}
    }
    
    # Parse personal information
    if personal_results:
        # Extract from content (would need parsing, for now use metadata)
        content = personal_results[0]['content']
        lines = content.split('\n')
        for line in lines:
            if 'Email' in line:
                email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', line)
                if email_match:
                    resume_data['personal_information']['email'] = email_match.group(1)
            if 'GitHub' in line:
                github_match = re.search(r'https?://github\.com/[^\s|]+', line)
                if github_match:
                    resume_data['personal_information']['github'] = github_match.group(0)
            if 'LinkedIn' in line:
                linkedin_match = re.search(r'https?://[^\s|]+linkedin[^\s|]+', line)
                if linkedin_match:
                    resume_data['personal_information']['linkedin'] = linkedin_match.group(0)
            if 'Languages' in line:
                lang_match = re.search(r'Languages\*\*: (.+)', line)
                if lang_match:
                    resume_data['personal_information']['languages'] = lang_match.group(1).strip()
            if 'Industry Experience' in line:
                ind_match = re.search(r'Industry Experience\*\*: (.+)', line)
                if ind_match:
                    resume_data['personal_information']['Industry Experience'] = ind_match.group(1).strip()
        
        # Extract name from first line
        name_match = re.match(r'# (.+)', lines[0])
        if name_match:
            resume_data['personal_information']['full_name'] = name_match.group(1).strip()
    
    # Professional summary
    if summary_results:
        content = summary_results[0]['content']
        # Split into traits (sentences)
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        resume_data['professional_summary']['personality_traits'] = sentences
    
    # Group work experience achievements by job
    job_groups = defaultdict(lambda: {"achievements": [], "scores": [], "metadata": {}})
    for result in work_results:
        metadata = result['metadata']
        job_key = (
            metadata.get('company', ''),
            metadata.get('position', ''),
            metadata.get('start_date', ''),
            metadata.get('end_date', '')
        )
        
        # Use original achievement text from metadata (backward compatible fallback to content)
        achievement_text = result['metadata'].get('achievement_text', result['content'])
        job_groups[job_key]["achievements"].append(achievement_text)
        job_groups[job_key]["scores"].append(result['score'])
        job_groups[job_key]["metadata"] = {
            "location": metadata.get('location', ''),
            "industry": metadata.get('industry', '')
        }
    
    # Rank jobs by average similarity score
    ranked_jobs = []
    for job_key, job_data in job_groups.items():
        avg_score = sum(job_data["scores"]) / len(job_data["scores"]) if job_data["scores"] else 0
        ranked_jobs.append({
            "job_key": job_key,
            "avg_score": avg_score,
            "data": job_data
        })
    
    # Sort by average score (descending)
    ranked_jobs.sort(key=lambda x: x["avg_score"], reverse=True)
    
    # Print ranked jobs to terminal
    if ranked_jobs:
        print("\n" + "="*80)
        print("📊 RANKED JOBS BY RELEVANCE")
        print("="*80)
        for idx, job_info in enumerate(ranked_jobs, 1):
            company, position, start_date, end_date = job_info["job_key"]
            avg_score = job_info["avg_score"]
            num_achievements = len(job_info["data"]["achievements"])
            print(f"\n{idx}. {position} at {company}")
            print(f"   Period: {start_date} - {end_date}")
            print(f"   Relevance Score: {avg_score:.4f}")
            print(f"   Achievements: {num_achievements}")
        print("\n" + "="*80 + "\n")
    
    # Select top N jobs if filtering is enabled
    if use_filtering:
        selected_jobs = ranked_jobs[:top_k_jobs]
        print(f"✅ Selected top {len(selected_jobs)} most relevant job(s) for resume generation\n")
    else:
        selected_jobs = ranked_jobs
        print(f"✅ Retrieved all {len(selected_jobs)} job(s) (full resume context)\n")  # Include all jobs for full retrieval
    
    # Build work experience from selected jobs
    for job_info in selected_jobs:
        company, position, start_date, end_date = job_info["job_key"]
        job_data = job_info["data"]
        
        resume_data['work_experience'].append({
            "position": position,
            "company": company,
            "start_date": start_date,
            "end_date": end_date,
            "location": job_data["metadata"]["location"],
            "industry": job_data["metadata"]["industry"],
            "achievements": job_data["achievements"]
        })
    
    # Sort jobs by start date (most recent first)
    resume_data['work_experience'].sort(
        key=lambda x: x['start_date'],
        reverse=True
    )
    
    # Education
    for result in education_results:
        metadata = result['metadata']
        resume_data['education']['degrees'].append({
            "degree": metadata.get('degree', ''),
            "institution": metadata.get('institution', ''),
            "dates": metadata.get('dates', '')
        })
    
    # Continuing studies
    for result in continuing_results:
        metadata = result['metadata']
        resume_data['education']['continuing_studies'].append({
            "name": metadata.get('name', ''),
            "institution": metadata.get('institution', ''),
            "completion_date": metadata.get('completion_date', '')
        })
    
    # Skills (reconstruct by category)
    for result in skills_results:
        metadata = result['metadata']
        category = metadata.get('category', 'other')
        # Parse skills from content
        content = result['content']
        if ':' in content:
            skills_text = content.split(':', 1)[1].strip()
            skills_list = [s.strip() for s in skills_text.split(',')]
            
            # Convert category to snake_case key
            category_key = category.lower().replace(' ', '_')
            resume_data['skills'][category_key] = skills_list
    
    # Save retrieved resume context to JSON file
    output_dir = Path("output/retrieved_resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename based on job_title and company
    if job_title and company:
        # Sanitize job_title and company for filename
        def sanitize_filename(text: str) -> str:
            # Replace spaces with underscores and remove special characters
            text = re.sub(r'[^\w\s-]', '', text)
            text = re.sub(r'[\s-]+', '_', text)
            return text.strip('_')
        
        sanitized_title = sanitize_filename(job_title)
        sanitized_company = sanitize_filename(company)
        filename = f"Alejandro_Leiva_{sanitized_title}_{sanitized_company}.json"
    else:
        # For full retrieval, use timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"Alejandro_Leiva_full_resume_{timestamp}.json"
    
    output_path = output_dir / filename
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(resume_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Retrieved resume context saved to: {output_path}\n")
    
    return resume_data


def retrieve_personality_traits(job_analysis: Dict[str, Any], top_k: int = 5) -> str:
    """
    Retrieve relevant personality traits for cover letter enhancement.
    
    Parameters
    ----------
    job_analysis : Dict[str, Any]
        Job analysis containing soft_skills and keywords.
    top_k : int, optional
        Number of personality traits to retrieve (default: 5).
    
    Returns
    -------
    str
        Concatenated personality traits text.
    
    Examples
    --------
    >>> job_analysis = {
    ...     'soft_skills': ['collaboration', 'problem-solving', 'communication'],
    ...     'keywords': ['team player', 'analytical']
    ... }
    >>> traits = retrieve_personality_traits(job_analysis, top_k=5)
    >>> print(traits)
    """
    # Initialize embeddings and vector store
    embedder = OpenAIEmbeddings()
    store = QdrantVectorStore()
    
    # Construct query from soft skills
    query_parts = []
    if job_analysis.get('soft_skills'):
        query_parts.extend(job_analysis['soft_skills'])
    if job_analysis.get('keywords'):
        query_parts.extend(job_analysis['keywords'])
    
    query_text = ' '.join(query_parts) if query_parts else "personality traits"
    query_vector = embedder.embed_query(query_text)
    
    # Retrieve personality and strength traits
    results = store.search(
        query_vector=query_vector,
        top_k=top_k,
        section_filter=None  # Will manually filter
    )
    
    # Filter for personality/strength sections
    personality_texts = []
    for result in results:
        if result['section_type'] in ['personality', 'strength']:
            personality_texts.append(result['content'])
    
    return '\n\n'.join(personality_texts[:top_k])

def create_resume_prompt(resume_data: Dict[str, Any], job_analysis: Dict[str, Any], job_title: str, company: str, job_description: str) -> str:
    # Here the resume_data is from the RAG retrieval
    prompt_resume = f"""
    You are an expert resume writer tasked with creating a highly targeted, achievement-based resume that aligns precisely with given job requirements while accurately representing a candidate's experience.

    First, review the following information carefully:

    Job Title
    <job_title>
    {job_title}
    </job_title>

    Company
    <company>
    {company}
    </company>

    Job Description
    <job_description>
    {job_description}
    </job_description>

    Technical Skills
    <technical_skills>
    {json.dumps(job_analysis['technical_skills'])}
    </technical_skills>

    Soft Skills
    <soft_skills>
    {json.dumps(job_analysis['soft_skills'])}
    </soft_skills>

    Keywords
    <keywords>
    {json.dumps(job_analysis['keywords'])}
    </keywords>

    Current Resume Data:
    <resume_data>
    {json.dumps(resume_data, indent=2)}
    </resume_data>

    Think through these steps internally (do not output your analysis):

    1. Professional Summary: 
       CRITICAL - You must ONLY use terminology, skills, and technologies that explicitly appear in the resume data.
       
       Step 1a: Extract ONLY the technologies, tools, and skills explicitly mentioned in the candidate's work experience and continuing studies.
       Step 1b: From the job description, identify which required skills/technologies match those found in Step 1a.
       Step 1c: Calculate years of experience ONLY from the resume's work_experience dates for the matching role/field.
       Step 1d: Draft a 50-word summary using ONLY the intersection from Step 1b.
       
       Constraints:
       - Do NOT introduce terminology not present in resume data
       - Do NOT paraphrase resume skills to match job requirements
       - Do NOT use generic terms like "expert", "skilled", "seasoned", "proficient"
       - Do NOT claim experience with tools/technologies not in resume
       - DO use exact terminology from resume when possible
       - DO focus on demonstrable experience and concrete technologies
       - DO incorporate keywords from job ONLY if they appear in resume data
       
       If a job requirement has no match in resume, omit it from summary.
    
    2. Work Experience: Select the four most relevant work experiences from resume data including the most recent. Create 3 bullet points for each experience (2 bullet points for quality engineer and quality assurance roles) using strong action verbs and quantified achievements. Ensure each bullet point is relevant to the job requirements.
    
    3. Education: List all educational qualifications from resume data. Note any specific educational requirements from job description. Ensure consistent formatting for all entries.
    
    4. Continuing Studies: List all certifications and continuing education courses from resume data. Identify the four most relevant to the job requirements.

    CRITICAL: Output ONLY a valid JSON object with this exact structure. No text before or after:

    {{
        "professional_summary": "string",
        "work_experience": [
            {{
                "company": "string",
                "title": "string",
                "dates": "string",
                "bullet_points": ["string", "string", "string"]
            }}
        ],
        "education": [
            {{
                "degree": "string",
                "institution": "string",
                "year_completed": "string"
            }}
        ],
        "continuing_studies": [
            {{
                "name": "string",
                "institution": "string",
                "completion_date": "string"
            }}
        ]
    }}

    Don't use "Present" if the end date is before April 2025. Output pure JSON only.
    """
    return prompt_resume

# Resume Generation Data Validation
class ResumeResponse(BaseModel):
    """Model for the resume response structure."""
    professional_summary: str = ""
    work_experience: List[Dict[str, Any]] = Field(default_factory=list)
    education: List[Dict[str, Any]] = Field(default_factory=list)
    continuing_studies: List[Dict[str, Any]] = Field(default_factory=list)
    
    @model_validator(mode='before')
    @classmethod
    def extract_from_text(cls, data):
        """Extract JSON from text if the input is not already a dict."""
        if isinstance(data, dict):
            return data
            
        if isinstance(data, str):
            # Try direct JSON parsing first
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                # Extract JSON using regex if direct parsing fails
                try:
                    json_pattern = r'(\{[\s\S]*\})'
                    match = re.search(json_pattern, data)
                    if match:
                        return json.loads(match.group(1))
                except:
                    pass
                
                # If all else fails, print a warning and return empty structure
                print(f"Failed to parse resume response: {data[:100]}...")
                return {
                    "professional_summary": "",
                    "work_experience": [],
                    "education": [],
                    "continuing_studies": []
                }
        
        return data
    

def claude_resume(api_key: str, prompt_resume: str) -> Dict[str, Any]:
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model_name,
        max_tokens=2048,
        temperature=0.6,
        messages=[{"role": "user", "content": prompt_resume}]
    )
    # Use Pydantic model to validate and extract the response
    response_text = response.content[0].text
    try:
        resume_data = ResumeResponse.model_validate(response_text)
        return resume_data.model_dump()
    except Exception as e:
        print(f"Error parsing resume response with Pydantic: {e}")
        # Fallback to simpler parsing if Pydantic validation fails
        try:
            return json.loads(response_text)
        except:
            print(f"JSON parsing failed. Raw response: {response_text[:150]}...")
            # Last resort: return empty data structure
            return {
                "professional_summary": "",
                "work_experience": [],
                "education": [],
                "continuing_studies": []
            }

print("Resume generation functions defined!")

# %%
# Cover Letter Generation Functions

def create_cover_letter_prompt(resume_data: Dict[str, Any], job_analysis: Dict[str, Any], job_title: str, company: str, job_description: str, personality_traits: str = "") -> str:
    prompt_cover_letter = f"""
    You are an expert cover letter writer specializing in achievement-based writing and keyword optimization. 

    Here is the job description:
    <job_description>
    {job_description}
    </job_description>

    Here is the candidate's resume data:
    <resume_data>
    {json.dumps(resume_data, indent=2)}
    </resume_data>

    Here is the job title:
    <job_title>
    {job_title}
    </job_title>

    Here is the company name:
    <company>
    {company}
    </company>

    Here are the candidate's relevant personality traits:
    <personality_traits>
    {personality_traits if personality_traits else "No specific personality traits provided."}
    </personality_traits>

    Here are the relevant soft skills:
    <soft_skills>
    {json.dumps(job_analysis['soft_skills'])}
    </soft_skills>

    Here are the important keywords:
    <keywords>
    {json.dumps(job_analysis['keywords'])}
    </keywords>

    Think through your analysis internally (do not output this thinking):
    - Identify top 5 most important job requirements
    - Match soft skills to job needs and rank by importance
    - List relevant achievements from resume matching job responsibilities
    - Create skills-requirements matrix rating matches 1-5
    - Identify 3-5 current industry challenges the candidate can address
    - Review work experience timeline for accuracy
    - Find compelling career narrative or growth story
    - Brainstorm attention-grabbing opening hooks

    Now craft a cover letter following these guidelines:

    1. Structure: 3 short paragraphs (opening, body, closing) under 250 words total
    2. Tone: Casual but professional, avoid formal language or clichés
    3. Tense: Present tense (except for past experiences)
    4. Language: Avoid "expert", "skilled", "seasoned", "excited"
    5. Opening: Attention-grabbing hook
    6. Body: Highlight soft skills matching job requirements, naturally weave in relevant personality traits to demonstrate fit
    7. Closing: Explain how skills solve industry challenges
    8. Keywords: Naturally incorporate throughout
    9. Personality: Use provided personality traits to strengthen the narrative and show cultural/role fit
    10. Relevance: Only use information from resume matching job requirements
    11. Timeline: Accurately represent work experience dates

    CRITICAL: Output ONLY a valid JSON object with this exact structure. No text before or after:

    {{
        "opening_paragraph": "string (single paragraph)",
        "body_paragraph": "string (single paragraph)",
        "closing_paragraph": "string (single paragraph)"
    }}

    Do not include analysis tags, explanations, or any other text. Output pure JSON only.
    """
    return prompt_cover_letter

# Cover Letter Generation Data Validation
class CoverLetterResponse(BaseModel):
    """Model for the cover letter response structure."""
    opening_paragraph: str = "Unable to generate opening paragraph."
    body_paragraph: str = "Unable to generate body paragraph."
    closing_paragraph: str = "Unable to generate closing paragraph."
    
    @model_validator(mode='before')
    @classmethod
    def extract_from_text(cls, data):
        """Extract JSON from text if the input is not already a dict."""
        if isinstance(data, dict):
            return data
            
        if isinstance(data, str):
            # Try direct JSON parsing first
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                # Extract JSON using regex if direct parsing fails
                try:
                    json_pattern = r'(\{[\s\S]*\})'
                    match = re.search(json_pattern, data)
                    if match:
                        return json.loads(match.group(1))
                except:
                    pass
                
                # If all else fails, return default content
                return {
                    "opening_paragraph": "Unable to generate opening paragraph.",
                    "body_paragraph": "Unable to generate body paragraph.",
                    "closing_paragraph": "Unable to generate closing paragraph."
                }
        
        return data

def claude_cover_letter(api_key: str, prompt_cover_letter: str) -> Dict[str, Any]:
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model_name,
        max_tokens=4096,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt_cover_letter}]
    )
    # Use Pydantic model to validate and extract the response
    response_text = response.content[0].text
    try:
        cover_letter_data = CoverLetterResponse.model_validate(response_text)
        return cover_letter_data.model_dump()
    except Exception as e:
        print(f"Error parsing response with Pydantic: {e}")
        # Fallback to simpler parsing if Pydantic validation fails
        try:
            return json.loads(response_text)
        except:
            print(f"JSON parsing failed. Raw response: {response_text[:150]}...")
            # Last resort: return empty data structure
            return {
                "opening_paragraph": "Unable to generate opening paragraph.",
                "body_paragraph": "Unable to generate body paragraph.",
                "closing_paragraph": "Unable to generate closing paragraph."
            }

print("Cover letter generation functions defined!")

# %%
# Document Creation Functions

def _setup_document() -> Document:
    """
    Initialize and configure a Word document with standard settings.
    
    This helper function creates a new document and applies global formatting
    settings including margins, fonts, footer with page numbers, and header spacing.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    Document
        A configured python-docx Document object with standard formatting applied.
    
    Notes
    -----
    - Sets Helvetica font as default
    - Margins: Top=0.3", Bottom=0.2", Left/Right=0.5"
    - Adds right-aligned page number in footer
    - Footer distance: 0.3", Header distance: 0.3"
    
    Examples
    --------
    >>> doc = _setup_document()
    >>> # Document is ready for content addition
    """
    doc = Document()
    
    # Add footer with page number
    section = doc.sections[0]
    footer = section.footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    
    run = paragraph.add_run()
    
    # Begin field char
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    run._element.append(fldChar1)
    
    # Field instruction text
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = "PAGE"
    run._element.append(instrText)
    
    # End field char
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'end')
    run._element.append(fldChar2)

    # Set footer margins
    section.footer_distance = Inches(0.3)
    
    # Set header margins
    section.header_distance = Inches(0.3)

    # GLOBAL DOCUMENT SETTINGS
    style = doc.styles['Normal']
    style.font.name = 'Helvetica'

    # Set margin sizes
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.3)
        section.bottom_margin = Inches(0.2)
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)
    
    return doc


def _add_resume_section(doc: Document, resume: Dict[str, Any], resume_ale: Dict[str, Any]) -> None:
    """
    Add resume content section to a Word document.
    
    This helper function adds a complete resume section including header with contact
    information, professional summary, work experience, continuing studies, and education.
    
    Parameters
    ----------
    doc : Document
        The python-docx Document object to add content to.
    resume : Dict[str, Any]
        Generated resume content dictionary with keys:
        - professional_summary: str
        - work_experience: List[Dict] with company, title, dates, bullet_points
        - continuing_studies: List[Dict] with name, institution, completion_date
        - education: List[Dict] with degree, institution, year_completed
    resume_ale : Dict[str, Any]
        Candidate's base resume data from YAML file containing personal_information.
    
    Returns
    -------
    None
        Modifies the document in-place.
    
    Notes
    -----
    - Header includes name (22pt, bold, centered) and contact info (10pt, blue)
    - Professional summary is justified text
    - Work experience uses tab stops for right-aligned dates
    - Bullet points are indented 0.15 inches
    - Section headers are centered and bold
    - Phone number: 778.223.8536
    
    Examples
    --------
    >>> doc = _setup_document()
    >>> _add_resume_section(doc, resume_data, base_data)
    """
    # Name
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run(resume_ale['personal_information']['full_name'])
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title.paragraph_format.space_after = Pt(1.15)
    
    # Contact info
    contact = doc.add_paragraph()
    contact.alignment = WD_ALIGN_PARAGRAPH.CENTER
    email_run = contact.add_run(resume_ale['personal_information']['email'])
    email_run.font.size = Pt(10)
    email_run.font.color.rgb = RGBColor(23, 54, 93)
    separator_run = contact.add_run(' | ')
    linkedin_run = contact.add_run(resume_ale['personal_information']['linkedin'])
    linkedin_run.font.size = Pt(10)
    linkedin_run.font.color.rgb = RGBColor(23, 54, 93)
    separator_run = contact.add_run(' | ')
    github_run = contact.add_run(resume_ale['personal_information']['github'])
    github_run.font.size = Pt(10)
    github_run.font.color.rgb = RGBColor(23, 54, 93)
    separator_run = contact.add_run(' | ')
    phone_run = contact.add_run("778.223.8536")
    phone_run.font.size = Pt(10)
    phone_run.font.color.rgb = RGBColor(23, 54, 93)
    
    # Professional summary
    summary = doc.add_paragraph()
    summary.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    summary.add_run(resume['professional_summary'])
    summary.paragraph_format.space_after = Pt(10)
    
    # Experience
    exp_header = doc.add_paragraph()
    exp_header.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
    exp_header.add_run("EXPERIENCE").bold = True
    exp_header.paragraph_format.space_after = Pt(0)
    
    # Set up the tab stop for right-aligned dates
    tab_stops = doc.styles['Normal'].paragraph_format.tab_stops
    tab_stops.add_tab_stop(Inches(7.6), WD_TAB_ALIGNMENT.RIGHT)
    
    # Add work experience with right-aligned dates
    for job in resume['work_experience']:
        # Create paragraph for job header
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        
        # Add job title and company
        title_run = p.add_run(f"{job['title']} | ")
        title_run.bold = True
        company_run = p.add_run(job['company'])
        company_run.italic = True
        
        # Add right-aligned date using tab
        date_run = p.add_run('\t' + job['dates'])
        date_run.italic = True
        date_run.bold = True
        
        # Add bullet points
        for point in job['bullet_points']:
            bullet = doc.add_paragraph()
            bullet.alignment = WD_PARAGRAPH_ALIGNMENT.JUSTIFY
            bullet.paragraph_format.left_indent = Inches(0.15)
            bullet.add_run('• ' + point)
            bullet.paragraph_format.space_after = Pt(6)

    # Add Continuing Studies section
    cont_header = doc.add_paragraph()
    cont_header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cont_header.add_run("CONTINUING STUDIES").bold = True
    cont_header.paragraph_format.space_before = Pt(10)
    cont_header.paragraph_format.space_after = Pt(0)

    # Add continuing studies entries
    for cont in resume['continuing_studies']:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        name_run = p.add_run(f"{cont['name']} | ")
        name_run.bold = True
        inst_run = p.add_run(cont['institution'])
        inst_run.italic = True
        
        # Add completion date (right-aligned)
        completion_date = p.add_run('\t' + cont['completion_date'])
        completion_date.italic = True
        completion_date.bold = True
    
    # Add Education section
    edu_header = doc.add_paragraph()
    edu_header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    edu_header.add_run("EDUCATION").bold = True
    edu_header.paragraph_format.space_before = Pt(10)
    edu_header.paragraph_format.space_after = Pt(0)
    
    # Add education entries
    for edu in resume['education']:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        degree_run = p.add_run(f"{edu['degree']} | ")
        degree_run.bold = True
        inst_run = p.add_run(edu['institution'])
        inst_run.italic = True
        
        # Add year (right-aligned)
        completion_date = p.add_run('\t' + edu['year_completed'])
        completion_date.italic = True
        completion_date.bold = True


def _add_cover_letter_section(doc: Document, cover_letter: Dict[str, Any], resume_ale: Dict[str, Any], company: str) -> None:
    """
    Add cover letter content section to a Word document.
    
    This helper function adds a complete cover letter section including header with
    contact information, date, company name, greeting, and three content paragraphs.
    
    Parameters
    ----------
    doc : Document
        The python-docx Document object to add content to.
    cover_letter : Dict[str, Any]
        Generated cover letter content dictionary with keys:
        - opening_paragraph: str
        - body_paragraph: str
        - closing_paragraph: str
    resume_ale : Dict[str, Any]
        Candidate's base resume data from YAML file containing personal_information.
    company : str
        Company name for the cover letter recipient.
    
    Returns
    -------
    None
        Modifies the document in-place.
    
    Notes
    -----
    - Header includes name (22pt, bold, centered) and contact info (10pt, blue)
    - Date is right-aligned in "Month DD, YYYY" format
    - Greeting uses "Dear Hiring Manager:"
    - Three paragraphs: opening, body, closing (12pt spacing between)
    - Signature includes "Sincerely," and candidate name
    - Phone number: 778.223.8536
    
    Examples
    --------
    >>> doc = _setup_document()
    >>> _add_cover_letter_section(doc, cover_letter_data, base_data, "Tech Corp")
    """
    # Name
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title.add_run(resume_ale['personal_information']['full_name'])
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title.paragraph_format.space_after = Pt(1.15)
    
    # Contact info
    contact = doc.add_paragraph()
    contact.alignment = WD_ALIGN_PARAGRAPH.CENTER
    email_run = contact.add_run(resume_ale['personal_information']['email'])
    email_run.font.size = Pt(10)
    email_run.font.color.rgb = RGBColor(23, 54, 93)
    separator_run = contact.add_run(' | ')
    linkedin_run = contact.add_run(resume_ale['personal_information']['linkedin'])
    linkedin_run.font.size = Pt(10)
    linkedin_run.font.color.rgb = RGBColor(23, 54, 93)
    separator_run = contact.add_run(' | ')
    github_run = contact.add_run(resume_ale['personal_information']['github'])
    github_run.font.size = Pt(10)
    github_run.font.color.rgb = RGBColor(23, 54, 93)
    separator_run = contact.add_run(' | ')
    phone_run = contact.add_run("778.223.8536")
    phone_run.font.size = Pt(10)
    phone_run.font.color.rgb = RGBColor(23, 54, 93)

    # Current Date
    date = doc.add_paragraph()
    date.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    date.add_run(datetime.now().strftime('%B %d, %Y'))

    # Add space after the date
    date.paragraph_format.space_after = Pt(24)
    
    # Recipient info (Company name)    
    recipient = doc.add_paragraph()
    recipient.alignment = WD_ALIGN_PARAGRAPH.LEFT
    recipient.add_run(company)
    recipient.paragraph_format.space_after = Pt(24)
    
    # Greeting
    greeting = doc.add_paragraph()
    greeting.alignment = WD_ALIGN_PARAGRAPH.LEFT
    greeting.add_run('Dear Hiring Manager:')
    greeting.paragraph_format.space_after = Pt(12)
    
    # Opening paragraph
    opening = doc.add_paragraph()
    opening.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    opening.add_run(cover_letter['opening_paragraph'])
    opening.paragraph_format.space_after = Pt(12)
    
    # Body paragraph
    body = doc.add_paragraph()
    body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    body.add_run(cover_letter['body_paragraph'])
    body.paragraph_format.space_after = Pt(12)
    
    # Closing paragraph
    closing = doc.add_paragraph()
    closing.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    closing.add_run(cover_letter['closing_paragraph'])
    closing.paragraph_format.space_after = Pt(24)
    
    # Signature
    signature = doc.add_paragraph()
    signature.alignment = WD_ALIGN_PARAGRAPH.LEFT
    signature.add_run('Sincerely,\n\nAlejandro Leiva')


def create_resume_document(resume: Dict[str, Any], resume_ale: Dict[str, Any]) -> Document:
    """
    Create a Word document containing only a resume.
    
    This function generates a single-page Word document with the candidate's resume,
    including professional summary, work experience, continuing studies, and education.
    
    Parameters
    ----------
    resume : Dict[str, Any]
        Generated resume content dictionary with keys:
        - professional_summary: str
        - work_experience: List[Dict] with company, title, dates, bullet_points
        - continuing_studies: List[Dict] with name, institution, completion_date
        - education: List[Dict] with degree, institution, year_completed
    resume_ale : Dict[str, Any]
        Candidate's base resume data from YAML file containing personal_information.
    
    Returns
    -------
    Document
        A python-docx Document object containing the formatted resume.
    
    External Files Required
    -----------------------
    None. All data is passed as parameters.
    
    Examples
    --------
    Create a resume document:
    
    >>> doc = create_resume_document(resume_data, base_data)
    >>> doc.save('resume.docx')
    """
    doc = _setup_document()
    _add_resume_section(doc, resume, resume_ale)
    return doc


def create_cover_letter_document(cover_letter: Dict[str, Any], resume_ale: Dict[str, Any], company: str) -> Document:
    """
    Create a Word document containing only a cover letter.
    
    This function generates a single-page Word document with a tailored cover letter,
    including header, date, company address, greeting, and content paragraphs.
    
    Parameters
    ----------
    cover_letter : Dict[str, Any]
        Generated cover letter content dictionary with keys:
        - opening_paragraph: str
        - body_paragraph: str
        - closing_paragraph: str
    resume_ale : Dict[str, Any]
        Candidate's base resume data from YAML file containing personal_information.
    company : str
        Company name for the cover letter recipient.
    
    Returns
    -------
    Document
        A python-docx Document object containing the formatted cover letter.
    
    External Files Required
    -----------------------
    None. All data is passed as parameters.
    
    Examples
    --------
    Create a cover letter document:
    
    >>> doc = create_cover_letter_document(cover_letter_data, base_data, "Tech Corp")
    >>> doc.save('cover_letter.docx')
    """
    doc = _setup_document()
    _add_cover_letter_section(doc, cover_letter, resume_ale, company)
    return doc


def create_resume_coverletter(resume: Dict[str, Any], resume_ale: Dict[str, Any], cover_letter: Dict[str, Any], company: str) -> Document:
    """
    Create a Word document containing both resume and cover letter.
    
    This function generates a two-page Word document with the candidate's resume
    on the first page and a tailored cover letter on the second page.
    
    Parameters
    ----------
    resume : Dict[str, Any]
        Generated resume content dictionary with keys:
        - professional_summary: str
        - work_experience: List[Dict] with company, title, dates, bullet_points
        - continuing_studies: List[Dict] with name, institution, completion_date
        - education: List[Dict] with degree, institution, year_completed
    resume_ale : Dict[str, Any]
        Candidate's base resume data from YAML file containing personal_information.
    cover_letter : Dict[str, Any]
        Generated cover letter content dictionary with keys:
        - opening_paragraph: str
        - body_paragraph: str
        - closing_paragraph: str
    company : str
        Company name for the cover letter recipient.
    
    Returns
    -------
    Document
        A python-docx Document object containing both the resume and cover letter.
    
    External Files Required
    -----------------------
    None. All data is passed as parameters.
    
    Notes
    -----
    This function maintains backward compatibility with existing code that expects
    a combined resume and cover letter document.
    
    Examples
    --------
    Create a complete application package:
    
    >>> doc = create_resume_coverletter(resume_data, base_data, cover_letter_data, "Tech Corp")
    >>> doc.save('application.docx')
    """
    # Setup document with standard formatting
    doc = _setup_document()
    
    # Add resume section
    _add_resume_section(doc, resume, resume_ale)
    
    # Add page break before cover letter
    if doc.paragraphs:
        last_para = doc.paragraphs[-1]
        last_para.add_run().add_break(WD_BREAK.PAGE)
    else:
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    
    # Add cover letter section
    _add_cover_letter_section(doc, cover_letter, resume_ale, company)
    
    return doc

def convert_word_to_pdf(file_path: str) -> str:
    pdf_path = file_path.replace('.docx', '.pdf')
    pythoncom.CoInitialize()  # Initialize COM
    convert(file_path, pdf_path)
    pythoncom.CoUninitialize()  # Uninitialize COM
    return pdf_path

print("Document creation functions defined!")

# %%
# Orchestrating Function
def create_resume(job_url: str, brightdata_api_key: str, resume_yaml_path: str = 'resume_ale.yaml') -> tuple[str, str, Dict[str, Any]]:
    """
    Orchestrate the complete resume and cover letter generation process.
    
    This function serves as the main entry point for generating tailored resumes
    and cover letters from job postings. It extracts job information, analyzes
    requirements, generates customized content using AI, and creates professional
    documents.
    
    Parameters
    ----------
    job_url : str
        URL of the job posting (LinkedIn or Indeed supported).
    brightdata_api_key : str
        BrightData API key for job extraction.
    resume_yaml_path : str, optional
        Path to the YAML file containing resume data. Default is 'resume_ale.yaml'.
    
    Returns
    -------
    tuple[str, str, Dict[str, Any]]
        A tuple containing the paths to the generated Word document and PDF, and the job dictionary.
        Format: (word_document_path, pdf_document_path, job_data_dict)
    
    Raises
    ------
    ValueError
        If required environment variables are missing or job extraction fails.
    FileNotFoundError
        If the resume YAML file is not found.
    Exception
        For any other errors during the process.
    
    Examples
    --------
    Generate documents for a LinkedIn job:
    
    >>> doc_path, pdf_path = main("https://www.linkedin.com/jobs/view/12345")
    >>> print(f"Generated: {doc_path}, {pdf_path}")
    Generated: Data_Scientist_Company_2025_09_02.docx, Data_Scientist_Company_2025_09_02.pdf
    
    Use a custom resume file:
    
    >>> doc_path, pdf_path = main("https://ca.indeed.com/viewjob?jk=67890", "my_resume.yaml")
    """
    try:
        # Step 1: Extract job information
        print("Extracting job information from URL")
        job_dict, job_df = extract_job(job_url, brightdata_api_key)
        job_title = job_dict.get('job_title', 'Unknown Title')
        company = job_dict.get('company_name', 'Unknown Company')
        job_description = job_dict.get('job_summary', '') or job_dict.get('description_text', 'No description available')
        print(f"Extracted: {job_title} at {company}")
        
        # Step 2: Load resume data
        print("Loading resume data")
        resume_data = load_resume_yaml(resume_yaml_path)
        
        # Step 3: Analyze job description
        print("Analyzing job description")
        prompt_analysis = create_analysis_prompt(job_title, company, job_description)
        if not key:
            raise ValueError("CLAUDE_API_KEY environment variable is required.")
        job_analysis = claude_analysis(key, prompt_analysis)
        
        # Step 4: Generate tailored resume
        print("Generating tailored resume")
        prompt_resume = create_resume_prompt(resume_data, job_analysis, job_title, company, job_description)
        resume = claude_resume(key, prompt_resume)
        
        # Step 5: Generate tailored cover letter
        print("Generating tailored cover letter")
        prompt_cover_letter = create_cover_letter_prompt(resume, job_analysis, job_title, company, job_description)
        cover_letter = claude_cover_letter(key, prompt_cover_letter)
        
        # Step 6: Create and save documents
        print("Creating and saving documents")
        today = datetime.today().strftime('%Y_%m_%d')
        doc = create_resume_coverletter(resume, resume_data, cover_letter, company)
        doc_path = f'Alejandro_Leiva_{job_title}_{company}_{today}.docx'
        doc.save(doc_path)
        pdf_path = convert_word_to_pdf(doc_path)
        
        return doc_path, pdf_path, job_dict
        
    except Exception as e:
        print(f"❌ Error in main process: {e}")
        raise


