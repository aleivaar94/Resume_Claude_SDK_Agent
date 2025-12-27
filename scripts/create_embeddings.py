"""
Create embeddings from markdown files and store in Qdrant vector database.

This script processes resume and personality markdown files, chunks them appropriately,
generates OpenAI embeddings, and stores them in a local Qdrant database for RAG retrieval.

Usage
-----
python scripts/create_embeddings.py --file data/resume_ale.md
python scripts/create_embeddings.py --file data/personalities_16.md
python scripts/create_embeddings.py --reset  # Reset and rebuild entire database
"""

import argparse
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.embeddings import OpenAIEmbeddings
from src.core.vector_store import QdrantVectorStore


class MarkdownParser:
    """
    Parser for markdown resume and personality files.
    
    Extracts structured chunks with metadata for RAG storage.
    """
    
    @staticmethod
    def parse_resume_markdown(content: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Parse resume markdown file into chunks.
        
        Parameters
        ----------
        content : str
            Full markdown content.
        source_file : str
            Name of the source file.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of document chunks with metadata.
        
        Notes
        -----
        Chunking strategy:
        - Personal info: 1 chunk
        - Professional summary: 1 chunk
        - Work experience: 1 chunk per achievement bullet
        - Education: 1 chunk per degree
        - Continuing studies: 1 chunk per certification
        - Skills: 1 chunk per skill category
        """
        chunks = []
        lines = content.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Personal Information (header section)
            if line.startswith('# ') and not line.startswith('## '):
                name = line[2:].strip()
                personal_content = [line]
                i += 1
                
                # Collect until next ## header
                while i < len(lines) and not lines[i].strip().startswith('## '):
                    personal_content.append(lines[i])
                    i += 1
                
                chunks.append({
                    "content": '\n'.join(personal_content).strip(),
                    "source_file": source_file,
                    "section_type": "personal_info",
                    "metadata": {"name": name}
                })
                continue
            
            # Professional Summary
            if line == "## Professional Summary":
                i += 1
                summary_lines = []
                while i < len(lines) and not lines[i].strip().startswith('## '):
                    if lines[i].strip():
                        summary_lines.append(lines[i].strip())
                    i += 1
                
                chunks.append({
                    "content": ' '.join(summary_lines),
                    "source_file": source_file,
                    "section_type": "professional_summary",
                    "metadata": {}
                })
                continue
            
            # Work Experience
            if line == "## Work Experience":
                i += 1
                
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Check for next section
                    if line.startswith('## ') and 'Work Experience' not in line:
                        break
                    
                    # Job header: ### Position | Company
                    if line.startswith('### '):
                        job_header = line[4:].strip()
                        parts = job_header.split('|')
                        position = parts[0].strip() if len(parts) > 0 else ""
                        company = parts[1].strip() if len(parts) > 1 else ""
                        
                        i += 1
                        
                        # Metadata line: **Dates** | Location | Industry
                        if i < len(lines) and '**' in lines[i]:
                            metadata_line = lines[i].strip()
                            # Extract dates, location, industry
                            meta_parts = metadata_line.split('|')
                            dates = meta_parts[0].strip().replace('**', '') if len(meta_parts) > 0 else ""
                            location = meta_parts[1].strip() if len(meta_parts) > 1 else ""
                            industry = meta_parts[2].strip() if len(meta_parts) > 2 else ""
                            
                            # Parse start and end dates - split on ' - ' (with spaces) to preserve Month-Year format
                            if ' - ' in dates:
                                date_parts = dates.split(' - ')
                                start_date = date_parts[0].strip()
                                end_date = date_parts[1].strip() if len(date_parts) > 1 else ""
                            else:
                                # Handle single date or "Present"
                                start_date = dates.strip()
                                end_date = ""
                            
                            i += 1
                        else:
                            dates, location, industry = "", "", ""
                            start_date, end_date = "", ""
                        
                        # Collect achievement bullets
                        achievement_index = 0
                        while i < len(lines):
                            line = lines[i].strip()
                            
                            # Stop at next job or section
                            if line.startswith('###') or (line.startswith('## ') and 'Work Experience' not in line):
                                break
                            
                            # Achievement bullet
                            if line.startswith('- '):
                                achievement = line[2:].strip()
                                
                                # Create embedded content with job position context for better similarity search
                                embedded_content = f"{position}: {achievement}"
                                
                                chunks.append({
                                    "content": embedded_content,
                                    "source_file": source_file,
                                    "section_type": "work_experience",
                                    "metadata": {
                                        "position": position,
                                        "company": company,
                                        "start_date": start_date,
                                        "end_date": end_date,
                                        "location": location,
                                        "industry": industry,
                                        "achievement_index": achievement_index,
                                        "achievement_text": achievement  # Store original for display
                                    }
                                })
                                achievement_index += 1
                            
                            i += 1
                    else:
                        i += 1
                continue
            
            # Education
            if line == "## Education":
                i += 1
                
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Check for next section
                    if line.startswith('## ') and 'Education' not in line:
                        break
                    
                    # Degree header: ### Degree | Institution
                    if line.startswith('### '):
                        degree_header = line[4:].strip()
                        parts = degree_header.split('|')
                        degree = parts[0].strip() if len(parts) > 0 else ""
                        institution = parts[1].strip() if len(parts) > 1 else ""
                        
                        i += 1
                        
                        # Dates and country line
                        dates_country = ""
                        if i < len(lines) and '**' in lines[i]:
                            dates_country = lines[i].strip().replace('**', '')
                            i += 1
                        
                        content = f"{degree} from {institution}. {dates_country}"
                        
                        chunks.append({
                            "content": content,
                            "source_file": source_file,
                            "section_type": "education",
                            "metadata": {
                                "degree": degree,
                                "institution": institution,
                                "dates": dates_country
                            }
                        })
                    else:
                        i += 1
                continue
            
            # Continuing Studies
            if line == "## Continuing Studies":
                i += 1
                
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Check for next section
                    if line.startswith('## ') and 'Continuing Studies' not in line:
                        break
                    
                    # Study header: ### Name | Institution
                    if line.startswith('### '):
                        study_header = line[4:].strip()
                        parts = study_header.split('|')
                        name = parts[0].strip() if len(parts) > 0 else ""
                        institution = parts[1].strip() if len(parts) > 1 else ""
                        
                        i += 1
                        
                        # Completion date
                        completion_date = ""
                        if i < len(lines) and 'Completed:' in lines[i]:
                            completion_date = lines[i].strip().replace('**Completed:', '').replace('**', '').strip()
                            i += 1
                        
                        content = f"{name} from {institution}. Completed: {completion_date}"
                        
                        chunks.append({
                            "content": content,
                            "source_file": source_file,
                            "section_type": "continuing_studies",
                            "metadata": {
                                "name": name,
                                "institution": institution,
                                "completion_date": completion_date
                            }
                        })
                    else:
                        i += 1
                continue
            
            # Skills
            if line == "## Skills":
                i += 1
                
                while i < len(lines):
                    line = lines[i].strip()
                    
                    # Check for next section
                    if line.startswith('## '):
                        break
                    
                    # Skill category: **Category**: skill1, skill2, ...
                    if line.startswith('**') and '**:' in line:
                        parts = line.split('**:')
                        category = parts[0].replace('**', '').strip()
                        skills = parts[1].strip() if len(parts) > 1 else ""
                        
                        chunks.append({
                            "content": f"{category}: {skills}",
                            "source_file": source_file,
                            "section_type": "skills",
                            "metadata": {"category": category}
                        })
                    
                    i += 1
                continue
            
            i += 1
        
        return chunks
    
    @staticmethod
    def parse_portfolio_projects_hierarchical(content: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Parse portfolio projects into hierarchical chunks.
        
        Creates two chunk types:
        1. Technical summary (Tech Stack + Technical Highlights) for filtering
        2. Full project (all sections) for detailed context
        
        Parameters
        ----------
        content : str
            Raw markdown content of portfolio_projects.md.
        source_file : str
            Name of source file.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of chunks (6 total: 3 technical + 3 full).
        
        Examples
        --------
        >>> chunks = MarkdownParser.parse_portfolio_projects_hierarchical(content, "portfolio_projects.md")
        >>> tech_chunks = [c for c in chunks if c["section_type"] == "project_technical"]
        >>> len(tech_chunks)
        3
        >>> full_chunks = [c for c in chunks if c["section_type"] == "project_full"]
        >>> len(full_chunks)
        3
        """
        chunks = []
        projects = content.split('\n# ')[1:]  # Split by project headers
        
        for idx, project in enumerate(projects):
            lines = project.strip().split('\n')
            project_title = lines[0].strip()
            
            # Parse sections
            sections = {}
            current_section = None
            section_content = []
            
            for line in lines[1:]:
                if line.startswith('## '):
                    if current_section:
                        sections[current_section] = '\n'.join(section_content).strip()
                    current_section = line[3:].strip()
                    section_content = []
                else:
                    section_content.append(line)
            
            if current_section:
                sections[current_section] = '\n'.join(section_content).strip()
            
            # Extract metadata
            project_url = sections.get('URL', '').strip()
            tech_stack_raw = sections.get('Tech Stack', '')
            tech_stack = [t.strip() for t in tech_stack_raw.replace('.', '').split(',') if t.strip()]
            
            project_id = f"project_{idx}"
            
            # Chunk 1: Technical Summary (Tech Stack + Technical Highlights)
            tech_content = f"Project: {project_title}\n\n"
            tech_content += f"Technologies: {sections.get('Tech Stack', 'N/A')}\n\n"
            tech_content += f"Technical Work:\n{sections.get('Technical Highlights', 'N/A')}"
            
            chunks.append({
                "content": tech_content,
                "source_file": source_file,
                "section_type": "project_technical",
                "metadata": {
                    "project_title": project_title,
                    "project_url": project_url,
                    "project_id": project_id,
                    "tech_stack": tech_stack,
                    "chunk_type": "technical_summary"
                }
            })
            
            # Chunk 2: Full Project (all sections)
            full_content = f"# {project_title}\n\n"
            for section_name in ['Purpose', 'Tech Stack', 'Technical Highlights', 'Skills Demonstrated', 'Result/Impact']:
                if section_name in sections:
                    full_content += f"## {section_name}\n{sections[section_name]}\n\n"
            
            chunks.append({
                "content": full_content.strip(),
                "source_file": source_file,
                "section_type": "project_full",
                "metadata": {
                    "project_title": project_title,
                    "project_url": project_url,
                    "project_id": project_id,
                    "tech_stack": tech_stack,
                    "chunk_type": "full_content"
                }
            })
        
        return chunks
    
    @staticmethod
    def parse_personality_fixed_chunks(
        content: str, 
        source_file: str,
        chunk_size: int = 400,
        overlap: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Parse personality markdown with fixed-size chunking and sliding window.
        
        Applies simple fixed-size chunking across entire document without section
        awareness. Uses sliding window with overlap to maintain context.
        
        Parameters
        ----------
        content : str
            Full markdown content from personalities_16.md.
        source_file : str
            Filename for tracking ("personalities_16.md").
        chunk_size : int, default=400
            Target characters per chunk.
        overlap : int, default=100
            Overlapping characters between consecutive chunks (25% overlap).
        
        Returns
        -------
        List[Dict[str, Any]]
            List of chunk dictionaries with payload structure:
            {
                "content": str,
                "source_file": str,
                "section_type": "",  # Empty for personality chunks
                "metadata": {
                    "chunk_index": int,
                    "char_start": int,
                    "char_end": int,
                    "overlap_chars": int
                }
            }
        
        Notes
        -----
        Chunking strategy:
        - Apply 400-char chunks with 100-char overlap across entire content
        - No section identification or header parsing
        - Simple sliding window approach
        
        Examples
        --------
        >>> content = Path("data/personalities_16.md").read_text()
        >>> chunks = MarkdownParser.parse_personality_fixed_chunks(content, "personalities_16.md")
        >>> len(chunks)
        10
        >>> chunks[0]["section_type"]
        ''
        """
        chunks = []
        chunk_index = 0
        start = 0
        
        # Apply fixed-size chunking across entire content
        while start < len(content):
            end = min(start + chunk_size, len(content))
            chunk_text = content[start:end]
            
            # Calculate overlap for next chunk
            overlap_chars = overlap if end < len(content) else 0
            
            chunks.append({
                "content": chunk_text.strip(),
                "source_file": source_file,
                "section_type": "",  # No section_type for personality chunks
                "metadata": {
                    "chunk_index": chunk_index,
                    "char_start": start,
                    "char_end": end,
                    "overlap_chars": overlap_chars
                }
            })
            
            # Move forward with overlap (step size = chunk_size - overlap)
            start += (chunk_size - overlap)
            chunk_index += 1
        
        return chunks


def process_file(file_path: str, file_type: str) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process a file and extract chunks.
    
    Uses auto-detection to determine chunking method:
    - personalities_16.md: Fixed-size chunking (400 chars, 100 overlap)
    - resume files: Header-based chunking
    
    Parameters
    ----------
    file_path : str
        Path to the file to process.
    file_type : str
        Type of file ('markdown').
    
    Returns
    -------
    Tuple[List[Dict[str, Any]], int]
        Tuple of (chunks list, total character count).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    source_file = Path(file_path).name
    
    # Auto-detect chunking method based on filename
    if source_file == "personalities_16.md":
        print(f"   ✂️  Using fixed-size chunking (400 chars, 100 overlap) for personalities_16.md")
        chunks = MarkdownParser.parse_personality_fixed_chunks(content, source_file)
    elif source_file == "portfolio_projects.md":
        print(f"   ✂️  Using hierarchical chunking for portfolio_projects.md")
        chunks = MarkdownParser.parse_portfolio_projects_hierarchical(content, source_file)
    elif 'resume' in source_file.lower():
        chunks = MarkdownParser.parse_resume_markdown(content, source_file)
    else:
        raise ValueError(f"Unknown file type: {source_file}. Expected 'personalities_16.md', 'portfolio_projects.md', or resume file.")
    
    return chunks, len(content)


def main():
    """
    Main function to process files and store embeddings.
    
    Examples
    --------
    $ python scripts/create_embeddings.py --file data/resume_ale.md
    $ python scripts/create_embeddings.py --file data/personalities_16.md --collection personality
    $ python scripts/create_embeddings.py --delete_collection personality
    $ python scripts/create_embeddings.py --reset
    """
    parser = argparse.ArgumentParser(description="Create embeddings from markdown files")
    parser.add_argument('--file', type=str, help='Path to markdown file')
    parser.add_argument('--collection', type=str, 
                        help='Target collection name (auto-detected if not specified: resume files -> "resume_data", personality files -> "personality")')
    parser.add_argument('--delete_collection', type=str, 
                        choices=['resume_data', 'personality', 'projects'],
                        help='Delete a specific collection (resume_data, personality, or projects)')
    parser.add_argument('--reset', action='store_true', 
                        help='Reset and rebuild entire vector database (deletes storage folder)')
    
    args = parser.parse_args()
    
    # Check for mutually exclusive flags
    if args.reset and args.delete_collection:
        print("❌ Error: Cannot use --reset and --delete_collection together")
        print("   Use --reset to delete entire database, OR --delete_collection to delete specific collection")
        return
    
    # Initialize embeddings
    print("🔧 Initializing OpenAI embeddings...")
    embedder = OpenAIEmbeddings()
    
    # Delete specific collection if requested
    if args.delete_collection:
        print(f"🗑️  Deleting collection: {args.delete_collection}")
        store = QdrantVectorStore(collection_name=args.delete_collection, auto_create=False)
        store.delete_collection()
        print(f"   💡 You can now recreate it using: python scripts/create_embeddings.py --file <file_path>")
        return
    
    # Reset database if requested
    if args.reset:
        print("🗑️  Resetting vector database (deleting all collections)...")
        # Create a single store instance to reset the entire database
        # This avoids file locking issues with multiple client instances
        store = QdrantVectorStore(collection_name="resume_data")
        store.reset_database()
        print(f"   ✅ Reset complete - all collections deleted and storage cleared")
    
    # Process file if provided
    if args.file:
        print(f"\n📄 Processing file: {args.file}")
        
        # Auto-detect collection if not specified
        if args.collection:
            collection_name = args.collection
            print(f"📦 Using specified collection: {collection_name}")
        else:
            # Auto-detect based on filename
            source_file = Path(args.file).name.lower()
            if 'personalit' in source_file:
                collection_name = "personality"
                print(f"📦 Auto-detected collection: personality (from filename)")
            elif 'portfolio' in source_file or 'project' in source_file:
                collection_name = "projects"
                print(f"📦 Auto-detected collection: projects (from filename)")
            elif 'resume' in source_file:
                collection_name = "resume_data"
                print(f"📦 Auto-detected collection: resume_data (from filename)")
            else:
                print(f"⚠️  Warning: Could not auto-detect collection from filename. Using 'resume_data'")
                collection_name = "resume_data"
        
        # Initialize vector store with target collection
        store = QdrantVectorStore(collection_name=collection_name)
        
        # Extract chunks
        chunks, char_count = process_file(args.file, 'markdown')
        print(f"✂️  Extracted {len(chunks)} chunks ({char_count} characters)")
        
        # Generate embeddings (batch)
        print("🔮 Generating embeddings...")
        texts = [chunk["content"] for chunk in chunks]
        embeddings = embedder.embed_texts(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Store in Qdrant
        print(f"💾 Storing in vector database (collection: {collection_name})...")
        store.add_documents(chunks, embeddings)
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Collection: {collection_name}")
        print(f"   Total documents in collection: {store.count_documents()}")
        print(f"   Section types: {set(c['section_type'] for c in chunks)}")
        
        # Additional details for portfolio projects
        if collection_name == "projects":
            tech_chunks = [c for c in chunks if c['section_type'] == 'project_technical']
            full_chunks = [c for c in chunks if c['section_type'] == 'project_full']
            print(f"   Technical summary chunks: {len(tech_chunks)}")
            print(f"   Full project chunks: {len(full_chunks)}")
            print(f"   Projects indexed: {[c['metadata']['project_title'] for c in tech_chunks]}")
    else:
        # Show current state for all collections
        print(f"\n📊 Current database state:")
        for collection in ["resume_data", "personality", "projects"]:
            try:
                store = QdrantVectorStore(collection_name=collection)
                count = store.count_documents()
                print(f"   {collection}: {count} documents")
            except Exception as e:
                print(f"   {collection}: Collection not found or empty")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
