# Repository Structure

This document describes the reorganized structure of the Resume_Claude_SDK_Agent repository.

## Directory Structure

```
Resume_Claude_SDK_Agent/
├── app.py                           # Chainlit UI entry point (stays at root for easy execution)
├── chainlit.md                      # Chainlit welcome screen
├── pyproject.toml                   # Project dependencies
├── README.md                        # Project overview
├── src/                             # Source code directory
│   ├── __init__.py                  # Package marker
│   ├── agent/                       # Agent tools and orchestration
│   │   ├── __init__.py
│   │   ├── tools.py                 # MCP tool definitions for Claude agent
│   │   └── __pycache__/             # Python bytecode cache
│   ├── core/                        # Core business logic
│   │   ├── __init__.py
│   │   ├── embeddings.py            # OpenAI embeddings wrapper
│   │   ├── resume_generator.py      # Resume/cover letter generation logic
│   │   ├── vector_store.py          # Qdrant vector database manager
│   │   └── __pycache__/             # Python bytecode cache
│   └── integrations/                # External service integrations
│       ├── __init__.py
│       ├── brightdata.py            # Job scraping via BrightData API
│       └── __pycache__/             # Python bytecode cache
├── scripts/                         # Utility scripts
│   ├── convert_yaml_to_md.py        # Convert YAML resume to markdown
│   └── create_embeddings.py         # Process markdown files and create embeddings
├── data/                            # Data files
│   ├── personalities_16.md          # Personality traits markdown
│   ├── portfolio_projects.md        # Portfolio projects markdown
│   ├── resume_ale.md                # Resume in markdown format
│   └── resume_ale.yaml              # Candidate resume data in YAML
├── notebooks/                       # Jupyter notebooks for experimentation
├── vector_db/                       # Vector database storage
│   └── qdrant_storage/              # Qdrant local storage directory
│       ├── meta.json                # Qdrant metadata
│       └── collection/              # Vector collections
│           └── resume_data/         # Resume data collection
├── output/                          # Generated documents (gitignored)
│   └── .gitkeep
├── archive/                         # Archived files
│   └── resume_ale_V1.yaml           # Previous version of resume YAML
├── docs/                            # Documentation
│   ├── Documentation.md             # This file
│   ├── JSON_Parse_Error.md          # JSON parsing error documentation
│   └── notes.md                     # Project notes
├── __pycache__/                     # Python bytecode cache (root level)
├── .env                             # Environment variables (gitignored)
├── .env.example                     # Environment variable template
└── .gitignore                       # Git ignore rules
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
- **`embeddings.py`** - OpenAI embeddings wrapper for generating text embeddings using OpenAI's text-embedding-3-small model
- **`resume_generator.py`** - Core resume generation logic including:
  - Prompt creation functions
  - Claude API calls with Pydantic validation
  - Word document creation with python-docx
  - PDF conversion functionality
  - Orchestration function for the complete workflow
- **`vector_store.py`** - Qdrant vector database manager for storing and retrieving embeddings with semantic search capabilities

### src/integrations/
- **`brightdata.py`** - Job scraping functionality:
  - Detects platform (LinkedIn/Indeed) from URL
  - Triggers BrightData dataset collection
  - Polls for snapshot completion
  - Extracts job information into dict and DataFrame

### scripts/
- **`convert_yaml_to_md.py`** - Utility script to convert structured YAML resume data to human-readable markdown format for RAG processing
- **`create_embeddings.py`** - Script to process markdown files, chunk them into sections, generate OpenAI embeddings, and store in Qdrant vector database for retrieval-augmented generation

### data/
- **`personalities_16.md`** - Markdown file containing personality traits and characteristics
- **`portfolio_projects.md`** - Markdown file describing portfolio projects and achievements
- **`resume_ale.md`** - Resume content in markdown format (generated from YAML)
- **`resume_ale.yaml`** - Candidate resume data in structured YAML format including personal information, work experience, education, skills, etc.

### vector_db/
- **`qdrant_storage/`** - Local Qdrant vector database storage directory containing metadata and vector collections for persistent storage of embeddings

### notebooks/
- Placeholder for Jupyter notebooks used for experimentation, data analysis, and prototyping

### archive/
- **`resume_ale_V1.yaml`** - Archived previous version of the resume YAML file

## Embedding and Vector Store System

This project implements a Retrieval-Augmented Generation (RAG) system using OpenAI embeddings and Qdrant vector database for intelligent resume processing and job matching.

### Overview
The embedding system converts text documents (resumes, personality profiles, portfolio projects) into numerical vectors that capture semantic meaning. These vectors are stored in a vector database for efficient similarity search, enabling the AI agent to retrieve relevant information when generating tailored resumes and cover letters.

### Components

#### 1. OpenAI Embeddings (`src/core/embeddings.py`)
- **Purpose**: Generates high-dimensional vector representations of text using OpenAI's `text-embedding-3-small` model
- **Key Features**:
  - Batch processing for multiple texts (`embed_texts()`)
  - Single query embedding generation (`embed_query()`)
  - 1536-dimensional vectors for rich semantic representation
  - Environment variable-based API key management
- **Usage**: Converts raw text chunks into embeddings before storage

#### 2. Qdrant Vector Store (`src/core/vector_store.py`)
- **Purpose**: Manages persistent storage and retrieval of embeddings with metadata
- **Key Features**:
  - Local file-based storage (no external database required)
  - Cosine similarity search for semantic matching
  - Metadata filtering (e.g., by section type: work_experience, education)
  - Collection management (create, delete, count documents)
  - Batch document insertion with embeddings
- **Storage Structure**: Each document chunk includes content, source file, section type, and custom metadata

#### 3. Data Processing Pipeline (`scripts/create_embeddings.py`)
- **Purpose**: Orchestrates the complete pipeline from raw markdown to stored embeddings
- **Key Features**:
  - Markdown parsing with intelligent chunking strategies
  - Different parsing logic for resumes vs. personality profiles
  - Batch embedding generation and storage
  - Command-line interface for processing specific files or resetting the database
- **Chunking Strategy**:
  - Resume: Personal info (1 chunk), Professional summary (1 chunk), Work experience (per achievement bullet), Education (per degree), Skills (per category)
  - Personality: Main sections (## headers) and subsections (### headers) as separate chunks

### Data Flow

```
Raw Data (YAML/Markdown) → Parsing & Chunking → Embedding Generation → Vector Storage → Semantic Search
     ↓                           ↓                      ↓                   ↓                ↓
resume_ale.yaml            MarkdownParser         OpenAI API      QdrantVectorStore   Query Vectors
personalities_16.md        create_embeddings.py   embeddings.py    vector_store.py     RAG Retrieval
```

### Usage Examples

#### Processing Resume Data
```bash
# Convert YAML to markdown (one-time setup)
python scripts/convert_yaml_to_md.py

# Generate embeddings and store in vector database
python scripts/create_embeddings.py --file data/resume_ale.md --type markdown
python scripts/create_embeddings.py --file data/personalities_16.md --type markdown
```

#### Programmatic Usage
```python
from src.core.embeddings import OpenAIEmbeddings
from src.core.vector_store import QdrantVectorStore

# Initialize components
embedder = OpenAIEmbeddings()
store = QdrantVectorStore()

# Generate query embedding
query_vector = embedder.embed_query("Python machine learning experience")

# Search for relevant resume sections
results = store.search(query_vector, top_k=5, section_filter="work_experience")

# Results include content, metadata, and similarity scores
for result in results:
    print(f"Company: {result['metadata']['company']}")
    print(f"Content: {result['content'][:100]}...")
    print(f"Similarity: {result['score']:.3f}")
```

### Environment Variables
- **`OPENAI_API_KEY`**: Required for generating embeddings via OpenAI API

### Dependencies
- `openai`: For embedding API calls
- `qdrant-client`: For vector database operations
- `python-dotenv`: For environment variable loading

### Performance Considerations
- Embeddings are 1536-dimensional floats (approximately 6KB per vector)
- Cosine similarity provides efficient semantic search
- Local Qdrant storage ensures data privacy and offline operation
- Batch processing optimizes API usage and reduces latency

## Import Structure

The new structure uses absolute imports from the `src/` package:

```python
# In app.py
from src.agent.tools import scrape_job_tool, analyze_job_tool, ...

# In src/agent/tools.py
from src.integrations.brightdata import extract_job
from src.core.resume_generator import claude_analysis, create_resume_prompt, ...
from src.core.embeddings import OpenAIEmbeddings
from src.core.vector_store import QdrantVectorStore

# In src/core/resume_generator.py
from src.integrations.brightdata import extract_job

# In scripts/create_embeddings.py
from src.core.embeddings import OpenAIEmbeddings
from src.core.vector_store import QdrantVectorStore
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

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- uv package manager (recommended) or pip
- Git for version control

### Installation Steps
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Resume_Claude_SDK_Agent
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Prepare data**:
   ```bash
   # Convert YAML resume to markdown
   python scripts/convert_yaml_to_md.py

   # Generate embeddings
   python scripts/create_embeddings.py --file data/resume_ale.md --type markdown
   python scripts/create_embeddings.py --file data/personalities_16.md --type markdown
   ```

## Usage Guide

### Basic Workflow
1. **Data Preparation**: Convert structured data to markdown and generate embeddings
2. **Job Scraping**: Use the Chainlit UI to scrape job postings
3. **Resume Generation**: The AI agent analyzes jobs and generates tailored resumes
4. **Document Creation**: Output Word and PDF documents

### Command Line Tools
- **Convert YAML to Markdown**:
  ```bash
  python scripts/convert_yaml_to_md.py
  ```

- **Create Embeddings**:
  ```bash
  # Process specific file
  python scripts/create_embeddings.py --file data/resume_ale.md --type markdown

  # Reset and rebuild entire database
  python scripts/create_embeddings.py --reset
  ```

### Chainlit Application
```bash
chainlit run app.py
```
Access the web interface to interact with the AI agent for resume generation.

## Flow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Input    │───▶│  Chainlit UI     │───▶│   AI Agent      │
│  (Job URL)      │    │  (app.py)        │    │  (tools.py)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Job Scraping    │───▶│  Data Analysis   │───▶│ Resume Gen      │
│ (BrightData)    │    │  (Claude API)    │    │ (Claude API)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Vector Search   │◀───│  Embeddings      │◀───│ Document Output │
│ (Qdrant)        │    │  (OpenAI)        │    │ (Word/PDF)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

**Data Processing Flow**:
```
YAML Resume ──▶ Markdown ──▶ Chunks ──▶ Embeddings ──▶ Vector DB ──▶ RAG Search
     │               │           │           │             │            │
     └─ Structured   └─ Human    └─ Sections └─ OpenAI    └─ Qdrant    └─ Semantic
        Data            Readable    by Type     API         Local DB      Retrieval
```

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
- **OPENAI_API_KEY** (required) - For generating text embeddings
- **OPEN_AI_KEY** (optional) - For additional OpenAI features
- **FIRECRAWL_API_KEY** (optional) - For web scraping
