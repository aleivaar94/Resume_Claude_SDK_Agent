![Claude](https://img.shields.io/badge/Claude-D97757?style=for-the-badge&logo=claude&logoColor=white) ![Chainlit](https://img.shields.io/badge/Chainlit-ff0059?style=for-the-badge&logo=chainlit&logoColor=white) ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

# Reflekt

/riˈflɛkt/

pronounced ree-FLEKT

Reflekt is an AI agent that generates tailored resumes and cover letters by analyzing job postings from LinkedIn and Indeed. It uses RAG (Retrieval-Augmented Generation) with Claude Agent SDK, OpenAI embeddings, and Qdrant vector database, to match your resume,  personality traits and portfolio projects, with the job requirements.

See the LinkedIn post [here](https://www.linkedin.com/posts/activity-7411627532262969344-HfWO?utm_source=share&utm_medium=member_desktop&rcm=ACoAAA2BJpgBEZ_SUWo4zpVHkx9hYA8UBl3Wfew)

<figure>
  <img src="assets/images/Reflekt_1.0.gif" alt="Alt text" width="700"/>
  <figcaption>Reflekt: AI Agent Resume and Cover Letter Generator</figcaption>
</figure>

<br>

This project builds upon the first version of the Resume generator I made (below), which was a simple AI automation with a Streamlit interface. You can find the LinkedIn posts [here](https://www.linkedin.com/posts/activity-7318293543297458176-UPuD?utm_source=share&utm_medium=member_desktop&rcm=ACoAAA2BJpgBEZ_SUWo4zpVHkx9hYA8UBl3Wfew) and [here](https://www.linkedin.com/posts/activity-7325897273341542400-2kUd?utm_source=share&utm_medium=member_desktop&rcm=ACoAAA2BJpgBEZ_SUWo4zpVHkx9hYA8UBl3Wfew)

<figure>
  <img src="assets/images/Resume_AI_1.0.gif" alt="Alt text" width="500"/>
  <figcaption>First version of the Resume generator with a Streamlit interface</figcaption>
</figure>

## Features

- **Job Scraping**: Extract job details from LinkedIn and Indeed URLs using [BrightData API.](https://brightdata.com/)
- **Intelligent Job Analysis**: Analyze job descriptions to identify technical skills, soft skills, and keywords using LLMs.
- **RAG-Based Resume Generation**: Retrieve relevant experience from vector database and generate tailored resumes.
- **RAG-Based Cover Letter Generation:**
    - **Personality Matching**: Match personality traits to job soft skills for authentic cover letters.
    - **Portfolio Integration**: Automatically select and include relevant portfolio projects.
- **Multi-Format Output**: Generate both Word (.docx) and PDF documents.
- **Interactive UI**: Chat-based interface powered by Chainlit.
- **Flexible Document Creation**: Generate resume only, cover letter only, or both.

### Example Interactions

**Generate both resume and cover letter:**
```
User: https://www.linkedin.com/jobs/view/123456
Agent: ✅ I've successfully created your complete application package!
```

**Resume only:**
```
User: https://ca.indeed.com/viewjob?jk=67890 - resume only
Agent: ✅ I've successfully created your resume!
```

**Using snapshot ID:**
```
User: s_mjepdfj94zb4miakd for LinkedIn
Agent: ✅ [Processes and generates documents]
```

**Custom hiring manager:**
```
User: https://www.linkedin.com/jobs/view/123456 
      The hiring manager is Sarah Johnson
Agent: ✅ [Generates cover letter with "Dear Sarah Johnson:"]
```

## Repository Structure

```
Resume_Claude_SDK_Agent/
├── app.py                           # Chainlit UI entry point
├── src/
│   ├── agent/
│   │   └── tools.py                 # MCP tool definitions for Claude agent
│   ├── core/
│   │   ├── embeddings.py            # OpenAI embeddings wrapper
│   │   ├── resume_generator.py      # Resume/cover letter generation logic
│   │   └── vector_store.py          # Qdrant vector database manager
│   └── integrations/
│       └── brightdata.py            # Job scraping via BrightData API
├── scripts/
│   ├── convert_yaml_to_md.py        # Convert YAML resume to markdown
│   └── create_embeddings.py         # Process markdown files and create embeddings
├── assets/
│   └── images/                      # Project images and GIFs
├── data/                            # Resume, personality, and portfolio data
├── docs/
│   └── Documentation.md             # Comprehensive project documentation
├── notebooks/
│   ├── get_snapshot_brighdata.ipynb # BrightData API interaction notebook
│   └── query_vector_store.ipynb     # Vector store query exploration notebook
├── vector_db/                       # Qdrant local storage
└── output/                          # Generated documents
```

## Set-up Requirements

- Python 3.12.5
- uv package manager
- API keys for:
  - Anthropic Claude API
  - OpenAI API
  - BrightData API

**Environment variables**
   
   Create a `.env` file in the project root:
   ```env
   CLAUDE_API_KEY=your_claude_key_here
   OPENAI_API_KEY=your_openai_key_here
   BRIGHTDATA_API_KEY=your_brightdata_key_here
   ```
   See `.env.example` for reference.

**Prepare your resume data**
   
Add your resume data in exactly the same format as `resume_ale.md`

**Prepare your personality traits data**
Add your personality traits data in exactly the same format as `personalities_16.md`

**Prepare your portfolio projects data**
Add your portfolio projects data in exactly the same format as `portfolio_projects.md`

**Create embeddings**
```bash
# Process resume data
python scripts/create_embeddings.py --file data/resume_ale.md

# Process personality traits
python scripts/create_embeddings.py --file data/personalities_16.md

# Process portfolio projects
python scripts/create_embeddings.py --file data/portfolio_projects.md
```

### Run the Application

Start the Chainlit UI:
```bash
chainlit run app.py
```

The application will open in your browser at `http://localhost:8000`.



## Project Components
For detailed documentation see `docs/Documentation.md`

### Core Modules

#### `src/core/embeddings.py`
OpenAI embeddings wrapper using `text-embedding-3-small` model (1536 dimensions).

**Key Functions:**
- `embed_texts(texts: List[str])` - Batch embed multiple texts
- `embed_query(text: str)` - Embed single query

#### `src/core/vector_store.py`
Qdrant vector database manager for local persistent storage.

**Key Functions:**
- `add_documents(chunks, embeddings)` - Store document chunks with embeddings
- `search(query_vector, top_k, section_filter)` - Semantic search with filtering
- `reset_database()` - Reset entire database

**Collections:**
- `resume_data`: Work experience, education, skills, personal info
- `personality`: Personality traits from `personalities_16.md`
- `projects`: Portfolio projects from `portfolio_projects.md`

#### `src/core/resume_generator.py`
Core resume and cover letter generation logic with RAG retrieval.

**Key Functions:**
- `create_analysis_prompt()` - Create job analysis prompt
- `claude_analysis()` - Extract skills and keywords from job description
- `retrieve_resume_context()` - RAG-based resume data retrieval
- `retrieve_personality_traits()` - Match personality to job requirements
- `retrieve_portfolio_projects_hierarchical()` - Two-step portfolio search
- `create_resume_prompt()` - Create tailored resume prompt
- `claude_resume()` - Generate resume content with Claude
- `create_cover_letter_prompt()` - Create cover letter prompt
- `claude_cover_letter()` - Generate cover letter with Claude
- `create_resume_document()` - Generate Word document for resume
- `create_cover_letter_document()` - Generate Word document for cover letter
- `convert_word_to_pdf()` - Convert Word to PDF

### Agent Tools

#### `src/agent/tools.py`
MCP (Model Context Protocol) tools that Claude can invoke:

1. **scrape_job** - Extract job info from URL or snapshot ID (requires BrightData API)
2. **analyze_job** - Analyze job description using Claude
3. **get_personality_traits** - Retrieve relevant personality traits via RAG
4. **get_portfolio_projects** - Retrieve relevant portfolio projects via RAG
5. **generate_resume_content** - Generate tailored resume content
6. **generate_cover_letter_content** - Generate tailored cover letter
7. **create_documents** - Create Word and PDF files

### Integrations

#### `src/integrations/brightdata.py`
BrightData API integration for job scraping.

**Key Functions:**
- `get_brightdata_snapshot_linkedin()` - Trigger LinkedIn scrape
- `get_brightdata_snapshot_indeed()` - Trigger Indeed scrape
- `get_snapshot_output()` - Retrieve scrape results with retry logic
- `extract_job()` - Main function supporting URLs and snapshot IDs

### Utility Scripts


#### `scripts/create_embeddings.py`
Processes markdown files, chunks them, generates embeddings, and stores in Qdrant.

**Usage:**
```bash
# Delete specific collection
python scripts/create_embeddings.py --delete_collection personality

# Reset entire database
python scripts/create_embeddings.py --reset
```

**Chunking Strategies:**
- `personalities_16.md`: Fixed-size chunks (400 chars, 100 overlap)
- `portfolio_projects.md`: Hierarchical chunking (project-level)
- `resume_ale.md`: Header-based chunking (section-level)


**Retrieval Process:**
1. **Query Analysis**: Extract relevant job requirements
2. **Semantic Search**: Find most similar chunks in vector DB
3. **Filtering**: Apply metadata filters (e.g., section type)
4. **Ranking**: Score results by relevance
5. **Context Assembly**: Combine retrieved chunks for prompt