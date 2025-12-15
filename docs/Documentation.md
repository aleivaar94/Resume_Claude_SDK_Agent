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

# Reset and rebuild entire vector database (use when source data or chunking logic changes)
python scripts/create_embeddings.py --reset
```

#### When to Reset the Vector Database
**Why Reset?** The vector database must be reset when:
- **Chunking logic changes** (e.g., fixing date parsing, modifying section extraction)
- **Source data is updated** (e.g., adding new work experience, skills)
- **Metadata structure changes** (e.g., adding new fields to chunks)
- **Corrupted or incorrect embeddings** are stored

**What Happens During Reset:**
1. Deletes the entire `resume_data` collection from Qdrant
2. Re-parses all markdown files with the latest chunking logic
3. Generates fresh embeddings for all chunks
4. Stores corrected data with proper metadata in the vector database

**Example Scenario:** If dates are stored incorrectly (e.g., `"March"` instead of `"March-2025"`), you must reset the database after fixing the parsing code. Simply re-running the script without `--reset` will create duplicate entries with the old corrupted data still present.

```bash
# After fixing chunking logic or updating source data:
python scripts/create_embeddings.py --reset
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

## RAG Process in Resume Generation

This project implements Retrieval-Augmented Generation (RAG) for creating tailored resumes and cover letters. RAG consists of three distinct phases: **Retrieval**, **Augmentation**, and **Generation**.

### What is RAG?

RAG stands for **Retrieval-Augmented Generation**, which consists of three distinct phases:

1. **Retrieval**: Query vector database to find relevant information
2. **Augmented**: Combine retrieved information with the query/prompt
3. **Generation**: Use LLM (Claude) to generate final output using augmented context

### RAG in Resume Generation: Step-by-Step Process

#### Phase 1: RETRIEVAL (Semantic Search - No Claude)
**Function**: [`retrieve_resume_context()`](src/core/resume_generator.py )

**What happens:**
```python
# Step 1a: Create query embedding (OpenAI)
query_text = f"{job_title} {company} {job_description}"
query_vector = embedder.embed_query(query_text)

# Step 1b: Retrieve a large pool of achievements from the vector store
# Parameters controlling retrieval:
# - top_k_achievement_pool: size of the achievement pool to fetch (e.g. 100)
# - top_k_achievements_per_job: maximum achievements to keep per job (e.g. 3)
# - top_k_jobs: how many jobs to return to the prompt (e.g. 4)

work_results = store.search(
    query_vector=query_vector,
    top_k=top_k_achievement_pool,          # retrieve a broad set of candidate achievements
    section_filter="work_experience"
)

# Step 1c: Group by job and reduce to top N achievements per job
# - Group retrieved achievements by (company, position, start_date, end_date)
# - For each job: sort its achievements by similarity score and keep only
#   the top `top_k_achievements_per_job` (jobs with fewer achievements keep all)
# - Calculate an average relevance score per job using those top achievements
# - Rank jobs by that average score and select the top `top_k_jobs` for generation
```

**Output**: Filtered resume data containing the selected jobs and their top achievements
```json
{
  "work_experience": [
    {"company": "CFIA", "position": "Data Scientist", "achievements": ["ach1","ach2","ach3"]}
  ],
  "education": [...],
  "skills": {...}
}
```

**Key Point**: No Claude is used in retrieval — this is pure **vector similarity search** (OpenAI embeddings + Qdrant) followed by deterministic post-processing that ensures a maximum number of achievements per job before the prompt is assembled.

Additional detail: what gets passed into the resume prompt
------------------------------------------------------

- For each job selected (the top `top_k_jobs` after ranking), the system keeps only the top N achievements where N is `top_k_achievements_per_job` (default: 3). Jobs with fewer than N retrieved achievements keep all available achievements.
- The resulting structure — i.e. the selected jobs and their up-to-N achievements — is what is embedded directly into the `Current Resume Data` section of the resume prompt passed to `claude_resume()` / `create_resume_prompt()`.

Example of the exact `resume_data['work_experience']` structure that is serialized into the Claude prompt:

```python
resume_data = {
  "work_experience": [
    {
      "company": "Canadian Food Inspection Agency",
      "position": "Data Scientist II",
      "start_date": "March-2025",
      "end_date": "November-2025",
      "achievements": [
        "Built ETL pipeline to process 1M records/day...",
        "Designed ML model improving detection accuracy by 12%...",
        "Optimized data ingestion reducing latency by 40%..."
      ]  # <= up to top_k_achievements_per_job (default 3)
    },
    {
      "company": "Rubicon Organics",
      "position": "Data Analyst",
      "start_date": "March-2023",
      "end_date": "December-2023",
      "achievements": [
        "Automated weekly sales reports...",
        "Built dashboards to track KPIs...",
        "Improved ETL reliability by 25%..."
      ]
    }
  ],
  "education": [...],
  "skills": {...}
}

# The prompt assembly then inserts this JSON under the "Current Resume Data" block:
prompt = create_resume_prompt(resume_data, job_analysis, job_title, company, job_description)
# Then claude_resume(api_key, prompt) is called to generate the tailored resume
```

This guarantees the resume prompt receives a concise, focused set of achievements per job (max 3 by default), preventing overly long context while preserving the most relevant evidence for each position.

---

#### Phase 2: AUGMENTATION (Combine Context - No LLM)
**Function**: [`create_resume_prompt()`](src/core/resume_generator.py )

**What happens:**
```python
prompt = f"""
You are an expert resume writer...

Job Title: {job_title}
Company: {company}
Job Description: {job_description}
Technical Skills: {job_analysis['technical_skills']}

Current Resume Data:  # ← RAG-RETRIEVED DATA
{json.dumps(resume_data, indent=2)}

Create a tailored resume using ONLY the resume data provided...
"""
```

**Output**: A prompt string that combines:
- Job requirements (context)
- RAG-retrieved resume chunks (retrieved data)
- Instructions for Claude (generation rules)

**Key Point**: This is the **augmentation** step where retrieved context enriches the prompt. No LLM is called yet.

---

#### Phase 3: GENERATION (Claude LLM Creates Content)
**Function**: [`claude_resume()`](src/core/resume_generator.py )

**What happens:**
```python
client = Anthropic(api_key=api_key)
response = client.messages.create(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": prompt_resume}]  # Augmented prompt
)
```

**Output**: Tailored resume generated by Claude using the augmented context
```json
{
  "professional_summary": "Data Scientist with 5+ years...",
  "work_experience": [
    {
      "company": "CFIA",
      "bullet_points": ["Developed Python ETL...", ...]
    }
  ]
}
```

**Key Point**: Claude generates the final content using the **augmented prompt** (retrieved data + job context).

### RAG in Cover Letter Generation: Step-by-Step Process

#### Phase 1: RETRIEVAL (Personality Traits)
**Function**: [`retrieve_personality_traits()`](src/core/resume_generator.py )

**What happens:**
```python
# Search vector store for personality traits matching job soft skills
personality_traits = retrieve_personality_traits(job_analysis, top_k=5)
```

**Output**: Relevant personality traits from vector store
```json
["analytical problem-solver", "collaborative team player", "innovative thinker"]
```

#### Phase 2: AUGMENTATION (Combine Context)
**Function**: [`create_cover_letter_prompt()`](src/core/resume_generator.py )

**What happens:**
```python
prompt = f"""
Job Description: {job_description}
Resume Data: {resume_data}
Personality Traits: {personality_traits}  # ← RAG-RETRIEVED
Soft Skills: {job_analysis['soft_skills']}

Create a cover letter...
"""
```

#### Phase 3: GENERATION (Claude Creates Cover Letter)
**Function**: [`claude_cover_letter()`](src/core/resume_generator.py )

**What happens:**
```python
response = client.messages.create(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": prompt_cover_letter}]
)
```

**Output**: Three-paragraph cover letter
```json
{
  "opening_paragraph": "Dear Hiring Manager, I am excited to apply...",
  "body_paragraph": "With my experience in Python and SQL...",
  "closing_paragraph": "I would welcome the opportunity to discuss..."
}
```

### Why Prompts Cannot Be Used in Retrieval

**Retrieval uses mathematical vector similarity**, not natural language understanding.

**Vector Search Process:**
```python
# This is what happens in retrieval:
query_vector = [0.234, -0.567, 0.891, ...]  # 1536 numbers
resume_vectors = [
    [0.245, -0.523, 0.876, ...],  # Achievement 1
    [0.123, -0.789, 0.234, ...],  # Achievement 2
]

# Calculate cosine similarity (mathematical operation)
similarities = cosine_similarity(query_vector, resume_vectors)
# Returns: [0.95, 0.72] - no prompt involved
```

**Prompts are for LLMs**, not for vector databases. The retrieval phase uses:
1. **Embeddings** (OpenAI's text-embedding-3-small)
2. **Vector similarity** (Qdrant's cosine similarity)
3. **No natural language processing**

### Sample Scenario: Data Scientist Job

**Input**: Data Scientist job at TechCorp requiring Python, SQL

#### RAG for Resume:

**1. RETRIEVAL** (No prompts, just math):
```python
query = "Data Scientist TechCorp Python SQL machine learning..."
query_vector = [0.234, -0.567, ...]  # 1536 dimensions

# Qdrant finds most similar vectors
retrieved_chunks = [
    "Developed Python ETL pipelines" (similarity: 0.92),
    "Built SQL databases" (similarity: 0.89),
    "Led team meetings" (similarity: 0.45)  # ← Filtered out
]
```

**2. AUGMENTATION** (Combine context):
```python
prompt = f"""
Job: Data Scientist at TechCorp
Requirements: Python, SQL

Retrieved Resume:  # ← From retrieval
- Developed Python ETL pipelines
- Built SQL databases

Create tailored resume...
"""
```

**3. GENERATION** (Claude creates content):
```python
claude_resume(prompt) → {
    "professional_summary": "Data Scientist with Python and SQL expertise...",
    "work_experience": [
        {"bullet_points": ["Developed Python ETL pipelines for 500K records..."]}
    ]
}
```

#### RAG for Cover Letter:

**1. RETRIEVAL** (Personality traits):
```python
traits = retrieve_personality_traits(
    {"soft_skills": ["problem-solving", "collaboration"]}, 
    top_k=5
) → ["analytical problem-solver", "collaborative team player"]
```

**2. AUGMENTATION**:
```python
prompt = f"""
Job: Data Scientist at TechCorp
Traits: analytical problem-solver, collaborative team player
Resume: [filtered resume data]

Write cover letter...
"""
```

**3. GENERATION**:
```python
claude_cover_letter(prompt) → {
    "opening_paragraph": "Dear Hiring Manager, As an analytical problem-solver...",
    "body_paragraph": "My collaborative approach to data science...",
    "closing_paragraph": "I look forward to contributing my expertise..."
}
```

### RAG Process Summary

| **Phase** | **Resume Function** | **Cover Letter Function** | **Technology** | **Purpose** |
|-----------|-------------------|---------------------------|----------------|-------------|
| **Retrieval** | [`retrieve_resume_context()`](src/core/resume_generator.py ) | [`retrieve_personality_traits()`](src/core/resume_generator.py ) | OpenAI Embeddings + Qdrant | Find relevant content via vector similarity |
| **Augmentation** | [`create_resume_prompt()`](src/core/resume_generator.py ) | [`create_cover_letter_prompt()`](src/core/resume_generator.py ) | String formatting | Combine retrieved data with job context |
| **Generation** | [`claude_resume()`](src/core/resume_generator.py ) | [`claude_cover_letter()`](src/core/resume_generator.py ) | Claude Haiku | Generate tailored content using augmented prompt |

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
