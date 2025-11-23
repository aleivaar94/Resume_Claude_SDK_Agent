# Repository Structure

This document describes the reorganized structure of the Resume_Claude_SDK_Agent repository.

## Directory Structure

```
Resume_Claude_SDK_Agent/
├── app.py                           # Chainlit UI entry point (stays at root for easy execution)
├── src/                             # Source code directory
│   ├── __init__.py                  # Package marker
│   ├── agent/                       # Agent tools and orchestration
│   │   ├── __init__.py
│   │   └── tools.py                 # MCP tool definitions for Claude agent
│   ├── core/                        # Core business logic
│   │   ├── __init__.py
│   │   └── resume_generator.py     # Resume/cover letter generation logic
│   └── integrations/                # External service integrations
│       ├── __init__.py
│       └── brightdata.py            # Job scraping via BrightData API
├── data/                            # Data files
│   └── resume_ale.yaml              # Candidate resume data
├── output/                          # Generated documents (gitignored)
│   └── .gitkeep
├── tests/                           # Test suite (for future tests)
│   └── __init__.py
├── docs/                            # Documentation (for future docs)
├── .env                             # Environment variables (gitignored)
├── .env.example                     # Environment variable template
├── .gitignore                       # Git ignore rules
├── pyproject.toml                   # Project dependencies
├── README.md                        # Project overview
└── chainlit.md                      # Chainlit welcome screen
```

## File Descriptions

### Root Level
- **`app.py`** - Main entry point for the Chainlit chat UI. Kept at root for easy execution with `chainlit run app.py`

### src/agent/
- **`tools.py`** - Defines 6 MCP tools that the Claude agent can use:
  - `scrape_job_tool` - Extract job information from LinkedIn/Indeed
  - `get_candidate_profile_tool` - Load resume data from YAML
  - `analyze_job_tool` - Analyze job description for skills/keywords
  - `generate_resume_content_tool` - Generate tailored resume
  - `generate_cover_letter_content_tool` - Generate tailored cover letter
  - `create_documents_tool` - Create Word and PDF documents

### src/core/
- **`resume_generator.py`** - Core resume generation logic including:
  - Prompt creation functions
  - Claude API calls with Pydantic validation
  - Word document creation with python-docx
  - PDF conversion functionality
  - Orchestration function for the complete workflow

### src/integrations/
- **`brightdata.py`** - Job scraping functionality:
  - Detects platform (LinkedIn/Indeed) from URL
  - Triggers BrightData dataset collection
  - Polls for snapshot completion
  - Extracts job information into dict and DataFrame

### data/
- **`resume_ale.yaml`** - Candidate's resume data including personal information, work experience, education, skills, etc.

### output/
- Generated Word and PDF documents are saved here
- Excluded from version control via `.gitignore`

### tests/
- Placeholder for future unit and integration tests

### docs/
- Placeholder for future documentation (architecture diagrams, API docs, user guides)

## Import Structure

The new structure uses absolute imports from the `src/` package:

```python
# In app.py
from src.agent.tools import scrape_job_tool, analyze_job_tool, ...

# In src/agent/tools.py
from src.integrations.brightdata import extract_job
from src.core.resume_generator import claude_analysis, create_resume_prompt, ...

# In src/core/resume_generator.py
from src.integrations.brightdata import extract_job
```

## Why This Structure Follows Best Practices

### 1. Separation of Concerns
- **Agent tools** (`src/agent/`) are separated from **core logic** (`src/core/`) and **external integrations** (`src/integrations/`)
- Each module has a single, clear responsibility

### 2. Scalability
- Easy to add new integrations (e.g., `src/integrations/indeed_api.py`)
- Easy to add new tools to the agent without touching core logic
- Easy to add tests without cluttering the root directory

### 3. Maintainability
- Clear module boundaries make it easy to locate code
- Configuration files (`data/`) separated from source code
- Generated files (`output/`) separated from source code

### 4. Security
- `.env` file is gitignored to protect API keys
- `.env.example` provides a template for new developers
- Personal resume data can be excluded from version control if needed

### 5. Developer Experience
- Entry point (`app.py`) remains at root for easy execution
- Standard Python package structure with `__init__.py` files
- Clear separation between code, data, and output

### 6. Version Control
- Generated files excluded via `.gitignore`
- Output directory only contains `.gitkeep` for Git tracking
- Environment variables properly templated

## Running the Application

```bash
# Install dependencies
uv sync

# Set up environment variables (copy .env.example to .env and fill in values)
cp .env.example .env

# Run the Chainlit application
chainlit run app.py
```

## Environment Variables Required

See `.env.example` for all required and optional environment variables:

- **CLAUDE_API_KEY** (required) - For resume generation
- **BRIGHTDATA_API_KEY** (required) - For job scraping
- **OPEN_AI_KEY** (optional) - For OpenAI features
- **FIRECRAWL_API_KEY** (optional) - For web scraping
