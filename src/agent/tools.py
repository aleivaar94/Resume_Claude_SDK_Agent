import os
import json
from datetime import datetime
from typing import Dict, Any, List
from claude_agent_sdk import tool
from src.integrations.brightdata import extract_job
from src.core.resume_generator import (
    claude_analysis, 
    create_analysis_prompt,
    claude_resume, 
    create_resume_prompt,
    claude_cover_letter, 
    create_cover_letter_prompt,
    create_resume_coverletter,
    create_resume_document,
    create_cover_letter_document,
    convert_word_to_pdf,
    # load_resume_yaml,
    retrieve_resume_context,
    retrieve_personality_traits
)

# Helper to get API keys
def get_brightdata_key():
    """
    Retrieve BrightData API key from environment variables.
    
    Returns
    -------
    str
        BrightData API key.
        
    Raises
    ------
    ValueError
        If BRIGHTDATA_API_KEY is not found in environment.
    """
    key = os.getenv("BRIGHTDATA_API_KEY")
    if not key:
        raise ValueError("BRIGHTDATA_API_KEY not found in environment variables")
    return key

def get_claude_key():
    """
    Retrieve Claude API key from environment variables.
    
    Returns
    -------
    str
        Claude API key.
        
    Raises
    ------
    ValueError
        If CLAUDE_API_KEY is not found in environment.
    """
    key = os.getenv("CLAUDE_API_KEY")
    if not key:
        raise ValueError("CLAUDE_API_KEY not found in environment variables")
    return key

@tool(
    "scrape_job", 
    "Extract job information from a LinkedIn or Indeed URL. Returns job_title, company_name, job_summary/description_text, and other metadata.", 
    {"url": str}
)
async def scrape_job_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scrapes job details from LinkedIn or Indeed URLs using BrightData API.
    
    Parameters
    ----------
    args : Dict[str, Any]
        Dictionary containing:
        - url : str
            The job posting URL (LinkedIn or Indeed).
    
    Returns
    -------
    Dict[str, Any]
        MCP tool response containing job information as JSON text.
    
    Examples
    --------
    Input:
    {"url": "https://www.linkedin.com/jobs/view/123456"}
    
    Output:
    {
        "content": [{
            "type": "text",
            "text": "{\"job_title\": \"Data Scientist\", \"company_name\": \"Acme Corp\", ...}"
        }]
    }
    """
    url = args["url"]
    api_key = get_brightdata_key()
    
    try:
        print(f"[scrape_job_tool] Extracting job from: {url}")
        # Here the "_" is the a pandas dataframe of the job data (not used)
        job_dict, _ = extract_job(url, api_key)
        print(f"[scrape_job_tool] Success: {job_dict.get('job_title', 'N/A')} at {job_dict.get('company_name', 'N/A')}")
        return {
            "content": [
                {"type": "text", "text": json.dumps(job_dict, indent=2)}
            ]
        }
    except Exception as e:
        print(f"[scrape_job_tool] Error: {str(e)}")
        return {
            "content": [
                {"type": "text", "text": f"Error scraping job: {str(e)}"}
            ],
            "is_error": True
        }

@tool(
    "get_candidate_profile", 
    "Retrieve the candidate's resume profile data using RAG from vector database. Contains personal_information, work_experience, education, skills, etc. Returns full resume data initially.", 
    {}
)
async def get_candidate_profile_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Loads the candidate's base resume information using RAG retrieval.
    
    This tool performs initial full retrieval from the vector database,
    returning all resume data without filtering.
    
    Parameters
    ----------
    args : Dict[str, Any]
        Empty dictionary (no parameters required).
    
    Returns
    -------
    Dict[str, Any]
        MCP tool response containing resume data as JSON text.
        
    Notes
    -----
    Uses RAG to retrieve from Qdrant vector database.
    Returns all resume data for initial context.
    
    Examples
    --------
    Output structure:
    {
        "content": [{
            "type": "text",
            "text": "{\"personal_information\": {...}, \"work_experience\": [...], ...}"
        }]
    }
    """
    try:
        print("[get_candidate_profile_tool] Retrieving resume data from vector DB (full retrieval)...")
        # Full retrieval without job filtering - call with no arguments
        resume_data = retrieve_resume_context()
        print(f"[get_candidate_profile_tool] Success - Retrieved {len(resume_data.get('work_experience', []))} jobs")
        return {
            "content": [
                {"type": "text", "text": json.dumps(resume_data, indent=2)}
            ]
        }
    except Exception as e:
        print(f"[get_candidate_profile_tool] Error: {str(e)}")
        return {
            "content": [
                {"type": "text", "text": f"Error loading resume profile: {str(e)}"}
            ],
            "is_error": True
        }

@tool(
    "get_personality_traits",
    "Retrieve relevant personality traits based on job soft skills to enhance cover letter. Pass job_analysis as JSON string.",
    {"job_analysis_json": str}
)
async def get_personality_traits_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieves personality traits relevant to job requirements using RAG.
    
    Parameters
    ----------
    args : Dict[str, Any]
        Dictionary containing:
        - job_analysis_json : str
            JSON string of job analysis from analyze_job tool.
    
    Returns
    -------
    Dict[str, Any]
        MCP tool response containing personality traits text.
    
    Examples
    --------
    Input:
    {"job_analysis_json": "{\"soft_skills\": [\"collaboration\", \"problem-solving\"]}"}
    
    Output:
    {
        "content": [{
            "type": "text",
            "text": "Innovative Mindset: My ability to see possibilities...\\n\\nIndependent Worker: ..."
        }]
    }
    """
    try:
        print("[get_personality_traits_tool] Retrieving relevant personality traits...")
        job_analysis = json.loads(args["job_analysis_json"])
        personality_text = retrieve_personality_traits(job_analysis, top_k=5)
        print(f"[get_personality_traits_tool] Success - Retrieved {len(personality_text.split('\\n\\n'))} traits")
        return {
            "content": [
                {"type": "text", "text": personality_text}
            ]
        }
    except json.JSONDecodeError as e:
        print(f"[get_personality_traits_tool] JSON Parse Error: {str(e)}")
        return {
            "content": [
                {"type": "text", "text": f"Error parsing JSON: {str(e)}"}
            ],
            "is_error": True
        }
    except Exception as e:
        print(f"[get_personality_traits_tool] Error: {str(e)}")
        return {
            "content": [
                {"type": "text", "text": f"Error retrieving personality traits: {str(e)}"}
            ],
            "is_error": True
        }

@tool(
    "analyze_job", 
    "Analyze job description using Claude to extract technical_skills, soft_skills, and keywords. Pass the job info as strings.", 
    {"job_title": str, "company": str, "job_description": str}
)
async def analyze_job_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes the job description using Claude AI to extract relevant skills and keywords.
    
    Parameters
    ----------
    args : Dict[str, Any]
        Dictionary containing:
        - job_title : str
            The job title.
        - company : str
            The company name.
        - job_description : str
            Full text of the job description.
    
    Returns
    -------
    Dict[str, Any]
        MCP tool response containing analysis results as JSON text.
    
    Examples
    --------
    Input:
    {
        "job_title": "Data Scientist",
        "company": "Acme Corp",
        "job_description": "We are looking for a Data Scientist with Python experience..."
    }
    
    Output:
    {
        "content": [{
            "type": "text",
            "text": "{\"technical_skills\": [...], \"soft_skills\": [...], \"keywords\": [...]}"
        }]
    }
    """
    try:
        print(f"[analyze_job_tool] Analyzing: {args['job_title']} at {args['company']}")
        prompt = create_analysis_prompt(
            args['job_title'], 
            args['company'], 
            args['job_description']
        )
        result = claude_analysis(get_claude_key(), prompt)
        print(f"[analyze_job_tool] Success - found {len(result.get('technical_skills', []))} technical skills")
        return {
            "content": [
                {"type": "text", "text": json.dumps(result, indent=2)}
            ]
        }
    except Exception as e:
        print(f"[analyze_job_tool] Error: {str(e)}")
        return {
            "content": [
                {"type": "text", "text": f"Error analyzing job: {str(e)}"}
            ],
            "is_error": True
        }

@tool(
    "generate_resume_content", 
    "Generate tailored resume content. Pass resume_data and job_analysis as JSON strings (use the exact output from previous tools).", 
    {
        "resume_data_json": str, 
        "job_analysis_json": str, 
        "job_title": str, 
        "company": str, 
        "job_description": str
    }
)
async def generate_resume_content_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates tailored resume content using Claude AI based on job requirements.
    
    Parameters
    ----------
    args : Dict[str, Any]
        Dictionary containing:
        - resume_data_json : str
            JSON string of resume data from get_candidate_profile.
        - job_analysis_json : str
            JSON string of job analysis from analyze_job.
        - job_title : str
            The job title.
        - company : str
            The company name.
        - job_description : str
            Full job description text.
    
    Returns
    -------
    Dict[str, Any]
        MCP tool response containing generated resume sections as JSON text.
    
    Examples
    --------
    The output contains:
    {
        "professional_summary": "...",
        "work_experience": [...],
        "education": [...],
        "continuing_studies": [...]
    }
    """
    try:
        print(f"[generate_resume_content_tool] Generating resume for: {args['job_title']}")
        print(f"[generate_resume_content_tool] Target company: {args['company']}")
        
        # Parse job_analysis JSON string
        job_analysis = json.loads(args['job_analysis_json'])
        
        # Retrieve filtered resume data using semantic search on full job info
        print(f"[generate_resume_content_tool] Retrieving relevant resume data via semantic search...")
        resume_data = retrieve_resume_context(
            job_title=args['job_title'],
            company=args['company'],
            job_description=args['job_description'],
            top_k_achievements=10,
            top_k_jobs=3
        )
        print(f"[generate_resume_content_tool] Retrieved {len(resume_data.get('work_experience', []))} relevant jobs")
        
        # Generate resume using filtered data
        prompt = create_resume_prompt(
            resume_data,
            job_analysis,
            args['job_title'],
            args['company'],
            args['job_description']
        )
        result = claude_resume(get_claude_key(), prompt)
        print(f"[generate_resume_content_tool] Success - {len(result.get('work_experience', []))} experiences")
        return {
            "content": [
                {"type": "text", "text": json.dumps(result, indent=2)}
            ]
        }
    except json.JSONDecodeError as e:
        print(f"[generate_resume_content_tool] JSON Parse Error: {str(e)}")
        return {
            "content": [
                {"type": "text", "text": f"Error parsing JSON inputs: {str(e)}. Make sure to pass the exact JSON output from previous tools."}
            ],
            "is_error": True
        }
    except Exception as e:
        print(f"[generate_resume_content_tool] Error: {str(e)}")
        return {
            "content": [
                {"type": "text", "text": f"Error generating resume content: {str(e)}"}
            ],
            "is_error": True
        }

@tool(
    "generate_cover_letter_content", 
    "Generate tailored cover letter. Pass resume_generated, job_analysis, and personality_traits as JSON strings from previous tool outputs.", 
    {
        "resume_generated_json": str, 
        "job_analysis_json": str, 
        "personality_traits": str,
        "job_title": str, 
        "company": str, 
        "job_description": str
    }
)
async def generate_cover_letter_content_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates tailored cover letter content using Claude AI.
    
    Parameters
    ----------
    args : Dict[str, Any]
        Dictionary containing:
        - resume_generated_json : str
            JSON string of generated resume from generate_resume_content.
        - job_analysis_json : str
            JSON string of job analysis from analyze_job.
        - personality_traits : str
            Text string of personality traits from get_personality_traits.
        - job_title : str
            The job title.
        - company : str
            The company name.
        - job_description : str
            Full job description text.
    
    Returns
    -------
    Dict[str, Any]
        MCP tool response containing cover letter paragraphs as JSON text.
    
    Examples
    --------
    The output contains:
    {
        "opening_paragraph": "...",
        "body_paragraph": "...",
        "closing_paragraph": "..."
    }
    """
    try:
        print(f"[generate_cover_letter_content_tool] Generating cover letter for: {args['company']}")
        
        # Parse JSON strings to dicts
        resume_generated = json.loads(args['resume_generated_json'])
        job_analysis = json.loads(args['job_analysis_json'])
        personality_traits = args.get('personality_traits', '')
        
        prompt = create_cover_letter_prompt(
            resume_generated,
            job_analysis,
            args['job_title'],
            args['company'],
            args['job_description'],
            personality_traits
        )
        result = claude_cover_letter(get_claude_key(), prompt)
        print("[generate_cover_letter_content_tool] Success")
        return {
            "content": [
                {"type": "text", "text": json.dumps(result, indent=2)}
            ]
        }
    except json.JSONDecodeError as e:
        print(f"[generate_cover_letter_content_tool] JSON Parse Error: {str(e)}")
        return {
            "content": [
                {"type": "text", "text": f"Error parsing JSON inputs: {str(e)}. Make sure to pass the exact JSON output from previous tools."}
            ],
            "is_error": True
        }
    except Exception as e:
        print(f"[generate_cover_letter_content_tool] Error: {str(e)}")
        return {
            "content": [
                {"type": "text", "text": f"Error generating cover letter content: {str(e)}"}
            ],
            "is_error": True
        }

@tool(
    "create_documents", 
    "Create Word and PDF documents. Supports 'both', 'resume_only', or 'cover_letter_only'. Pass all JSON data as strings. Returns absolute file paths.", 
    {
        "resume_generated_json": str, 
        "resume_original_json": str, 
        "cover_letter_generated_json": str, 
        "company": str, 
        "job_title": str,
        "document_type": str
    }
)
async def create_documents_tool(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates Word and PDF documents containing resume and/or cover letter.
    
    Parameters
    ----------
    args : Dict[str, Any]
        Dictionary containing:
        - resume_generated_json : str
            JSON string of generated resume content.
        - resume_original_json : str
            JSON string of original resume data (for personal info).
        - cover_letter_generated_json : str
            JSON string of generated cover letter (optional for resume_only).
        - company : str
            Company name for filename.
        - job_title : str
            Job title for filename.
        - document_type : str
            Type of document to create. Options:
            - "both" (default): Create resume + cover letter (2 pages)
            - "resume_only": Create only resume (1 page)
            - "cover_letter_only": Create only cover letter (1 page)
    
    Returns
    -------
    Dict[str, Any]
        MCP tool response containing file paths as JSON text.
    
    Notes
    -----
    Creates files in the 'output/' directory.
    Requires docx2pdf for PDF conversion (Windows only).
    File naming includes document type suffix for single-document outputs.
    
    Examples
    --------
    Output for resume_only:
    {
        "message": "Resume document created successfully.",
        "word_path": "C:\\...\\output\\Alejandro_Leiva_Data_Scientist_Acme_Resume_2025_11_22.docx",
        "pdf_path": "C:\\...\\output\\Alejandro_Leiva_Data_Scientist_Acme_Resume_2025_11_22.pdf"
    }
    """
    try:
        # Get document type (default to 'both')
        document_type = args.get('document_type', 'both').lower()
        
        print(f"[create_documents_tool] Creating {document_type} for: {args['job_title']} at {args['company']}")
        
        # Parse JSON strings to dicts
        resume_generated = json.loads(args['resume_generated_json'])
        resume_original = json.loads(args['resume_original_json'])
        
        # Cover letter is optional for resume_only
        cover_letter_generated = None
        if document_type in ['both', 'cover_letter_only']:
            cover_letter_generated = json.loads(args['cover_letter_generated_json'])
        
        company = args['company']
        job_title = args['job_title']
        today = datetime.today().strftime('%Y_%m_%d')
        
        # Create output directory if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Sanitize filename
        safe_title = "".join([c for c in job_title if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')
        safe_company = "".join([c for c in company if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')
        
        # Add document type suffix to filename
        type_suffix = ""
        if document_type == "resume_only":
            type_suffix = "_Resume"
        elif document_type == "cover_letter_only":
            type_suffix = "_CoverLetter"
        
        filename_base = f"Alejandro_Leiva_{safe_title}_{safe_company}{type_suffix}_{today}"
        doc_path = os.path.join(output_dir, f"{filename_base}.docx")
        
        # Create document based on type
        if document_type == "resume_only":
            doc = create_resume_document(resume_generated, resume_original)
            message = "Resume document created successfully."
        elif document_type == "cover_letter_only":
            doc = create_cover_letter_document(cover_letter_generated, resume_original, company)
            message = "Cover letter document created successfully."
        else:  # both
            doc = create_resume_coverletter(
                resume_generated, 
                resume_original, 
                cover_letter_generated, 
                company
            )
            message = "Resume and cover letter documents created successfully."
        
        doc.save(doc_path)
        print(f"[create_documents_tool] Word document saved: {doc_path}")
        
        pdf_path = convert_word_to_pdf(doc_path)
        print(f"[create_documents_tool] PDF created: {pdf_path}")
        
        return {
            "content": [
                {"type": "text", "text": json.dumps({
                    "message": message,
                    "document_type": document_type,
                    "word_path": os.path.abspath(doc_path),
                    "pdf_path": os.path.abspath(pdf_path)
                }, indent=2)}
            ]
        }
    except json.JSONDecodeError as e:
        print(f"[create_documents_tool] JSON Parse Error: {str(e)}")
        return {
            "content": [
                {"type": "text", "text": f"Error parsing JSON inputs: {str(e)}. Make sure to pass the exact JSON output from previous tools."}
            ],
            "is_error": True
        }
    except Exception as e:
        print(f"[create_documents_tool] Error: {str(e)}")
        return {
            "content": [
                {"type": "text", "text": f"Error creating documents: {str(e)}"}
            ],
            "is_error": True
        }
