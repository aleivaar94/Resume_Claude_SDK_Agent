import os
import json
import re
import chainlit as cl
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, create_sdk_mcp_server, AssistantMessage, TextBlock
from tools import (
    scrape_job_tool, 
    get_candidate_profile_tool, 
    analyze_job_tool, 
    generate_resume_content_tool, 
    generate_cover_letter_content_tool, 
    create_documents_tool
)

# Ensure API key is set for the SDK
if not os.getenv("ANTHROPIC_API_KEY") and os.getenv("CLAUDE_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("CLAUDE_API_KEY")

# Create the MCP server
resume_server = create_sdk_mcp_server(
    name="resume_agent",
    version="1.0.0",
    tools=[
        scrape_job_tool, 
        get_candidate_profile_tool, 
        analyze_job_tool, 
        generate_resume_content_tool, 
        generate_cover_letter_content_tool, 
        create_documents_tool
    ]
)

SYSTEM_PROMPT = """
You are an expert Resume AI Agent that generates tailored resumes and cover letters based on job postings.

## AVAILABLE TOOLS

1. **scrape_job** - Extract job details from LinkedIn/Indeed URL
   Returns: job_title, company_name, job_summary/description_text

2. **get_candidate_profile** - Load candidate's resume data from resume_ale.yaml
   Returns: personal_information, work_experience, education, skills

3. **analyze_job** - Analyze job description to extract skills and keywords
   Input: job_title, company, job_description
   Returns: technical_skills, soft_skills, keywords

4. **generate_resume_content** - Generate tailored resume content
   Input: resume_data_json, job_analysis_json, job_title, company, job_description
   Returns: professional_summary, work_experience, education, continuing_studies

5. **generate_cover_letter_content** - Generate tailored cover letter
   Input: resume_generated_json, job_analysis_json, job_title, company, job_description
   Returns: opening_paragraph, body_paragraph, closing_paragraph

6. **create_documents** - Create Word and PDF documents
   Input: resume_generated_json, resume_original_json, cover_letter_generated_json, company, job_title, document_type
   document_type options: "both" (default), "resume_only", "cover_letter_only"
   Returns: file paths to generated documents

## TYPICAL WORKFLOW

When a user provides a job URL:
1. Use `scrape_job` to extract job information
2. Use `get_candidate_profile` to load resume data
3. Use `analyze_job` to identify key requirements
4. Use `generate_resume_content` to create tailored resume
5. Use `generate_cover_letter_content` to create tailored cover letter
6. Use `create_documents` to create Word/PDF files

## FLEXIBLE BEHAVIOR - ADAPT TO USER REQUESTS

**Default behavior (just URL provided):**
- Generate both resume AND cover letter
- Use document_type="both" in create_documents
- Create 2-page document (resume page 1, cover letter page 2)

**When user says "only resume" or "resume without cover letter":**
- Skip generate_cover_letter_content tool
- Use document_type="resume_only" in create_documents
- Create 1-page document with only resume

**When user says "only cover letter":**
- Generate resume content first (needed for cover letter generation)
- Generate cover letter content
- Use document_type="cover_letter_only" in create_documents
- Create 1-page document with only cover letter

**When user says "markdown only" or "text only" or "show in chat":**
- Generate content as normal
- Display the content in the chat
- Do NOT call create_documents
- Format nicely for readability

**When user specifies format preferences:**
- Listen for keywords: "Word only", "PDF only", "just show me"
- Adjust your output accordingly

## CRITICAL DATA PASSING RULES

⚠️ Tools expect JSON STRINGS, not Python dicts!

**CORRECT way to pass data between tools:**
```
Tool output: {"key": "value"}
Next tool input: "{\"key\": \"value\"}"  ← Pass as STRING
```

**How to handle tool outputs:**
1. Save COMPLETE JSON output from each tool (don't summarize)
2. Pass the ENTIRE JSON string to the next tool
3. Never paraphrase or modify the JSON structure
4. Copy JSON exactly as received

**Example flow:**
- Step 2: get_candidate_profile returns: `{"personal_information": {...}, "work_experience": [...]}`
- Step 4: Pass to generate_resume_content as: `resume_data_json = "{\"personal_information\": {...}, \"work_experience\": [...]}"`

## OUTPUT FORMATTING

After generating documents:
- Explicitly state the document type created
- Provide the full file paths
- Explain what the user can do next

Example responses:
- "I've created a complete application package with resume and cover letter..."
- "I've generated a resume-only document as requested..."
- "Here's the cover letter content in markdown format..."

## ADAPTABILITY

You are flexible and intelligent:
- Understand user intent even with varied phrasing
- Ask clarifying questions if the request is ambiguous
- Provide helpful suggestions based on the context
- Adjust your workflow based on what the user actually needs
"""

@cl.on_chat_start
async def start():
    """
    Initialize the Chainlit application and Claude SDK client.
    
    This function runs when a user starts a new chat session. It sets up
    the Claude agent with the resume generation tools and stores the client
    in the user session.
    
    Notes
    -----
    Requires ANTHROPIC_API_KEY or CLAUDE_API_KEY in environment variables.
    """
    options = ClaudeAgentOptions(
        model="claude-haiku-4-5",
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"resume_tools": resume_server},
        allowed_tools=[
            "mcp__resume_tools__scrape_job", 
            "mcp__resume_tools__get_candidate_profile",
            "mcp__resume_tools__analyze_job",
            "mcp__resume_tools__generate_resume_content",
            "mcp__resume_tools__generate_cover_letter_content",
            "mcp__resume_tools__create_documents"
        ]
    )
    client = ClaudeSDKClient(options=options)
    await client.__aenter__()
    cl.user_session.set("client", client)
    
    await cl.Message(content="Hello! I'm your Resume AI Agent. Send me a LinkedIn or Indeed job URL to get started, or ask me to generate content in a specific format (e.g., 'only show the resume in markdown').").send()

@cl.on_chat_end
async def end():
    """
    Clean up resources when the chat session ends.
    
    This function properly closes the Claude SDK client and releases
    any associated resources.
    """
    client = cl.user_session.get("client")
    if client:
        await client.__aexit__(None, None, None)

@cl.on_message
async def main(message: cl.Message):
    """
    Handle incoming messages from the user.
    
    This function processes user messages, sends them to the Claude agent,
    streams the response back to the UI, and automatically detects generated
    file paths to provide download buttons.
    
    Parameters
    ----------
    message : cl.Message
        The incoming message from the user containing the job URL or request.
    
    Notes
    -----
    The function will automatically create download buttons for any .docx
    or .pdf files mentioned in the agent's response.
    """
    client = cl.user_session.get("client")
    
    msg = cl.Message(content="")
    await msg.send()
    
    full_response = ""

    try:
        # Send the query
        await client.query(message.content)
        
        # Iterate over responses
        async for response in client.receive_response():
            if isinstance(response, AssistantMessage):
                for block in response.content:
                    if isinstance(block, TextBlock):
                        text = block.text
                        await msg.stream_token(text)
                        full_response += text
    except Exception as e:
        await msg.stream_token(f"\\n\\nError: {str(e)}")
    
    await msg.update()

    # Check for file paths in the response to create download buttons
    elements = []
    
    # Find .docx paths
    docx_matches = re.findall(r'(?:[a-zA-Z]:\\[\w\-. \\]+\.docx|output[/\\][\w\-. ]+\.docx)', full_response)
    for path in docx_matches:
        clean_path = path.strip().replace('\\\\', '\\')
        if os.path.exists(clean_path):
            if not any(e.path == clean_path for e in elements):
                elements.append(cl.File(path=clean_path, name=os.path.basename(clean_path), display="inline"))

    # Find .pdf paths
    pdf_matches = re.findall(r'(?:[a-zA-Z]:\\[\w\-. \\]+\.pdf|output[/\\][\w\-. ]+\.pdf)', full_response)
    for path in pdf_matches:
        clean_path = path.strip().replace('\\\\', '\\')
        if os.path.exists(clean_path):
            if not any(e.path == clean_path for e in elements):
                elements.append(cl.File(path=clean_path, name=os.path.basename(clean_path), display="inline"))
            
    if elements:
        await cl.Message(content="Here are your generated documents:", elements=elements).send()