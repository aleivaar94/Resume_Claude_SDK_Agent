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
You are an expert Resume AI Agent. Your goal is to help users generate tailored resumes and cover letters.

CRITICAL WORKFLOW - Follow these steps in order:

1. Use `scrape_job` with the URL to get job details (job_title, company_name, job_summary/description_text)
   - Save the ENTIRE JSON output as "job_data"

2. Use `get_candidate_profile` to get resume data
   - Save the ENTIRE JSON output as "resume_data"

3. Use `analyze_job` with:
   - job_title: from job_data
   - company: from job_data (company_name field)
   - job_description: from job_data (job_summary OR description_text field)
   - Save the ENTIRE JSON output as "job_analysis"

4. Use `generate_resume_content` with:
   - resume_data_json: the COMPLETE JSON string from step 2
   - job_analysis_json: the COMPLETE JSON string from step 3
   - job_title: from job_data
   - company: from job_data
   - job_description: from job_data
   - Save the ENTIRE JSON output as "resume_generated"

5. Use `generate_cover_letter_content` with:
   - resume_generated_json: the COMPLETE JSON string from step 4
   - job_analysis_json: the COMPLETE JSON string from step 3
   - job_title: from job_data
   - company: from job_data
   - job_description: from job_data
   - Save the ENTIRE JSON output as "cover_letter_generated"

6. Check user's request:
   - If user asks for "markdown only" or "text only": Display the resume/cover letter content in the chat. STOP HERE.
   - Otherwise, proceed to step 7.

7. Use `create_documents` with:
   - resume_generated_json: the COMPLETE JSON string from step 4
   - resume_original_json: the COMPLETE JSON string from step 2
   - cover_letter_generated_json: the COMPLETE JSON string from step 5
   - company: from job_data
   - job_title: from job_data

IMPORTANT RULES:
- Always pass the COMPLETE JSON output from one tool to the next
- Never summarize or paraphrase tool outputs
- When a tool returns JSON, copy it exactly as-is to the next tool
- The tools expect JSON strings, not dictionaries
- After creating documents, explicitly state the file paths in your response

Example of correct data passing:
```
Step 2 output: {"personal_information": {...}, "work_experience": [...]}
Step 4 input resume_data_json: "{\"personal_information\": {...}, \"work_experience\": [...]}"
```
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
        model="claude-sonnet-4-5",
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
    
    await cl.Message(content="Hello! I'm your Resume AI Agent. Send me a LinkedIn or Indeed job URL to get started.").send()

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