import os
import json
import re
import asyncio
import chainlit as cl
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, create_sdk_mcp_server, AssistantMessage, TextBlock
from src.agent.tools import (
    scrape_job_tool, 
    get_personality_traits_tool,
    get_portfolio_projects_tool,
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
        get_personality_traits_tool,
        get_portfolio_projects_tool,
        analyze_job_tool, 
        generate_resume_content_tool, 
        generate_cover_letter_content_tool, 
        create_documents_tool
    ]
)

SYSTEM_PROMPT = """
You are an expert Resume AI Agent that generates tailored resumes and cover letters based on job postings.

## AVAILABLE TOOLS

1. **scrape_job** - Extract job details from LinkedIn/Indeed URL or retrieve from existing snapshot ID
   Input (URL): {"url": "https://www.linkedin.com/jobs/view/123456"}
   Input (Snapshot): {"url": "s_mjepdfj94zb4miakd", "platform": "linkedin"}
   Note: When using snapshot ID (starts with 's_'), platform parameter is REQUIRED ('linkedin' or 'indeed')
   Returns: job_title, company_name, job_summary/description_text

2. **analyze_job** - Analyze job description to extract skills and keywords
   Input: job_title, company, job_description
   Returns: technical_skills, soft_skills, keywords

3. **get_personality_traits** - Retrieve relevant personality traits for cover letter
   Input: job_analysis_json
   Returns: personality traits text matching job soft skills

4. **generate_resume_content** - Generate tailored resume content
   Input: job_analysis_json, job_title, company, job_description
   Returns: {resume_generated: {...}, resume_original: {...}}
   Note: Returns BOTH tailored content and original resume data (includes personal_information)

5. **generate_cover_letter_content** - Generate tailored cover letter
   Input: resume_generated_json (pass the entire output from generate_resume_content), job_analysis_json, job_title, company, personality_traits, portfolio_projects_json
   Returns: {"opening_paragraph": "...", "body_paragraph": "...", "closing_paragraph": "..."}
   Note: Returns ONLY the cover letter paragraphs (flat structure, no nested data)

6. **create_documents** - Create Word and PDF documents
   Input: resume_generated_json (from generate_resume_content), cover_letter_generated_json (from generate_cover_letter_content), portfolio_projects_json (from get_portfolio_projects), company, job_title, document_type, hiring_manager_greeting (optional)
   document_type options: "both" (default), "resume_only", "cover_letter_only"
   hiring_manager_greeting (optional): Custom greeting for cover letter (e.g., "Dear Alice Rid:"), defaults to "Dear Hiring Manager:". Extract from user message if hiring manager name is mentioned.
   Returns: file paths to generated documents
   Note: Pass cover_letter_generated_json and portfolio_projects_json as SEPARATE parameters

7. **get_portfolio_projects** - Retrieve relevant portfolio projects
   Input: job_analysis_json (from analyze_job)
   Returns: {
     "projects_for_prompt": [{"title": "...", "content": "...", "url": "..."}],
     "projects_for_list": [{"title": "...", "url": "...", "tech_stack": [...]}] 
   }
   - projects_for_prompt: has "content" field (no tech_stack)
   - projects_for_list: has "tech_stack" field (no content)

   
## TYPICAL WORKFLOW

When a user provides a job URL or snapshot ID, you MUST:
1. Use `scrape_job` to extract job information
2. Use `analyze_job` to identify key requirements
3. Use `get_personality_traits` to retrieve relevant personality traits
4. Use `get_portfolio_projects` to retrieve relevant portfolio projects
5. Use `generate_resume_content` to create tailored resume
6. Use `generate_cover_letter_content` to create tailored cover letter
7. Use `create_documents` to create Word/PDF files

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
- Step 2: analyze_job returns: `{"technical_skills": [...], "soft_skills": [...], "keywords": [...]}`
- Step 4: get_portfolio_projects returns: `{"projects_for_prompt": [...], "projects_for_list": [...]}`
- Step 6: generate_cover_letter_content returns: `{"opening_paragraph": "...", "body_paragraph": "...", "closing_paragraph": "..."}`
- Step 7: create_documents receives THREE separate JSON inputs:
  - resume_generated_json from generate_resume_content
  - cover_letter_generated_json from generate_cover_letter_content
  - portfolio_projects_json from get_portfolio_projects

## OUTPUT FORMATTING - CRITICAL

**After successfully calling create_documents:**
1. State: "✅ I've successfully created your complete application package for the [job_title] position at [company]!"
2. Do NOT add anything else


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
    The client is NOT context-managed here to avoid task mismatch errors.
    """
    options = ClaudeAgentOptions(
        model="claude-haiku-4-5",
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"resume_tools": resume_server},
        allowed_tools=[
            "mcp__resume_tools__scrape_job", 
            "mcp__resume_tools__get_personality_traits",
            "mcp__resume_tools__analyze_job",
            "mcp__resume_tools__get_portfolio_projects",
            "mcp__resume_tools__generate_resume_content",
            "mcp__resume_tools__generate_cover_letter_content",
            "mcp__resume_tools__create_documents"
        ]
    )
    # Create client WITHOUT context manager to avoid task mismatch
    client = ClaudeSDKClient(options=options)
    cl.user_session.set("client", client)
    cl.user_session.set("client_initialized", False)
    cl.user_session.set("stop_requested", False)
    cl.user_session.set("response_generator", None)
    
    await cl.Message(content="Hello! I'm your Resume AI Agent. Send me a LinkedIn or Indeed job URL or a snapshot ID. Make sure to specify the platform (e.g., LinkedIn, Indeed) if you send a snapshot ID.").send()

@cl.on_stop
def on_stop():
    """
    Handle user stop button clicks.
    
    Sets a flag to indicate the user wants to stop the current operation,
    allowing graceful shutdown of the async generator.
    
    Notes
    -----
    This is a synchronous function as required by Chainlit's @cl.on_stop.
    """
    cl.user_session.set("stop_requested", True)
    print("User requested stop - flagging for graceful shutdown")

@cl.on_chat_end
async def end():
    """
    Clean up resources when the chat session ends.
    
    This function properly closes the Claude SDK client and releases
    any associated resources without causing task mismatch errors.
    
    Notes
    -----
    Silently ignores cleanup errors to prevent error messages during shutdown.
    """
    # Close any pending response generator
    response_gen = cl.user_session.get("response_generator")
    if response_gen:
        try:
            await response_gen.aclose()
        except Exception:
            pass
        cl.user_session.set("response_generator", None)
    
    client = cl.user_session.get("client")
    if client:
        try:
            # Only cleanup if initialized
            if cl.user_session.get("client_initialized"):
                # Use close() method if available, otherwise just clear reference
                if hasattr(client, 'close'):
                    await client.close()
        except Exception:
            # Silently ignore cleanup errors - they don't affect functionality
            pass
        finally:
            cl.user_session.set("client", None)
            cl.user_session.set("client_initialized", False)

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
    Uses proper context management per message to avoid task mismatch.
    Properly handles GeneratorExit to prevent async generator cleanup errors.
    """
    client = cl.user_session.get("client")
    
    # Reset stop flag for new message
    cl.user_session.set("stop_requested", False)
    
    # Clear any previous file paths from session
    cl.user_session.set("generated_word_path", None)
    cl.user_session.set("generated_pdf_path", None)
    
    msg = cl.Message(content="")
    await msg.send()
    
    full_response = ""
    response_gen = None

    try:
        # Initialize client with context manager ONLY for this message
        if not cl.user_session.get("client_initialized"):
            await client.__aenter__()
            cl.user_session.set("client_initialized", True)
        
        # Send the query
        await client.query(message.content)
        
        # Get the response generator and store it for potential cleanup
        response_gen = client.receive_response()
        cl.user_session.set("response_generator", response_gen)
        
        # Iterate over responses with proper exception handling
        async for response in response_gen:
            # Check if stop was requested
            if cl.user_session.get("stop_requested"):
                await msg.stream_token("\n\n⚠️ Stopped by user.")
                break
            
            if isinstance(response, AssistantMessage):
                for block in response.content:
                    if isinstance(block, TextBlock):
                        text = block.text
                        await msg.stream_token(text)
                        full_response += text
                        
    except GeneratorExit:
        # Handle generator cleanup gracefully - this is expected during shutdown
        pass
    except asyncio.CancelledError:
        # Handle task cancellation gracefully
        try:
            await msg.stream_token("\n\n⚠️ Request was cancelled.")
        except Exception:
            pass
        raise
    except Exception as e:
        try:
            await msg.stream_token(f"\n\n❌ Error: {str(e)}")
        except Exception:
            pass
    finally:
        # Clear the stored generator reference
        cl.user_session.set("response_generator", None)
        
        # Properly close the async generator if it exists
        if response_gen is not None:
            try:
                await response_gen.aclose()
            except Exception:
                # Ignore errors during generator cleanup
                pass
        
        # Update the message
        try:
            await msg.update()
        except Exception:
            pass

    # Check for file paths in session to create download buttons
    elements = []
    
    # Retrieve file paths from session (stored by create_documents_tool)
    word_path = cl.user_session.get("generated_word_path")
    pdf_path = cl.user_session.get("generated_pdf_path")
    
    # Add Word file if path exists in session and file is accessible
    if word_path and os.path.exists(word_path):
        elements.append(cl.File(
            path=word_path,
            name=os.path.basename(word_path),
            display="inline"
        ))
        print(f"[main] Added Word file to download: {word_path}")
    
    # Add PDF file if path exists in session and file is accessible
    if pdf_path and os.path.exists(pdf_path):
        elements.append(cl.File(
            path=pdf_path,
            name=os.path.basename(pdf_path),
            display="inline"
        ))
        print(f"[main] Added PDF file to download: {pdf_path}")
    
    # Display download message with file buttons if files were created
    if elements:
        await cl.Message(content="📥 **Download your documents:**", elements=elements).send()
    else:
        print("[main] No file paths found in session - download buttons not displayed")