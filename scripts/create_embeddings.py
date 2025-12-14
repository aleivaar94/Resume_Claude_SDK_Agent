"""
Create embeddings from markdown files and store in Qdrant vector database.

This script processes resume and personality markdown files, chunks them appropriately,
generates OpenAI embeddings, and stores them in a local Qdrant database for RAG retrieval.

Usage
-----
python scripts/create_embeddings.py --file data/resume_ale.md --type markdown
python scripts/create_embeddings.py --file data/personalities_16.md --type markdown
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
                                
                                chunks.append({
                                    "content": achievement,
                                    "source_file": source_file,
                                    "section_type": "work_experience",
                                    "metadata": {
                                        "position": position,
                                        "company": company,
                                        "start_date": start_date,
                                        "end_date": end_date,
                                        "location": location,
                                        "industry": industry,
                                        "achievement_index": achievement_index
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
    def parse_personality_markdown(content: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Parse personality markdown file into chunks.
        
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
        - Main sections (## headers): 1 chunk each
        - Subsections (### headers): 1 chunk each (strengths/weaknesses)
        """
        chunks = []
        lines = content.split('\n')
        i = 0
        
        current_section = ""
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Main section (## header)
            if line.startswith('## '):
                current_section = line[3:].strip()
                section_content = []
                i += 1
                
                # Collect content until next ## or ### header
                while i < len(lines):
                    next_line = lines[i].strip()
                    
                    # Stop at next main section
                    if next_line.startswith('## '):
                        break
                    
                    # Handle subsections (### headers)
                    if next_line.startswith('### '):
                        break
                    
                    if next_line:
                        section_content.append(next_line)
                    i += 1
                
                # Add main section if it has content
                if section_content:
                    chunks.append({
                        "content": ' '.join(section_content),
                        "source_file": source_file,
                        "section_type": "personality",
                        "metadata": {"section": current_section}
                    })
                continue
            
            # Subsection (### header) - typically strengths/weaknesses
            if line.startswith('### '):
                subsection = line[4:].strip()
                subsection_content = []
                i += 1
                
                # Collect content until next ### or ## header
                while i < len(lines):
                    next_line = lines[i].strip()
                    
                    if next_line.startswith('###') or next_line.startswith('## '):
                        break
                    
                    if next_line:
                        subsection_content.append(next_line)
                    i += 1
                
                # Determine section type (strength or weakness)
                section_type = "strength" if "Strengths" in current_section else "weakness"
                
                if subsection_content:
                    chunks.append({
                        "content": f"{subsection}: {' '.join(subsection_content)}",
                        "source_file": source_file,
                        "section_type": section_type,
                        "metadata": {
                            "trait_name": subsection,
                            "category": current_section
                        }
                    })
                continue
            
            i += 1
        
        return chunks


def process_file(file_path: str, file_type: str) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process a file and extract chunks.
    
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
    
    if 'resume' in source_file.lower():
        chunks = MarkdownParser.parse_resume_markdown(content, source_file)
    elif 'personalit' in source_file.lower():
        chunks = MarkdownParser.parse_personality_markdown(content, source_file)
    else:
        raise ValueError(f"Unknown file type: {source_file}")
    
    return chunks, len(content)


def main():
    """
    Main function to process files and store embeddings.
    
    Examples
    --------
    $ python scripts/create_embeddings.py --file data/resume_ale.md --type markdown
    $ python scripts/create_embeddings.py --file data/personalities_16.md --type markdown
    $ python scripts/create_embeddings.py --reset
    """
    parser = argparse.ArgumentParser(description="Create embeddings from markdown files")
    parser.add_argument('--file', type=str, help='Path to markdown file')
    parser.add_argument('--type', type=str, default='markdown', choices=['markdown'], 
                        help='File type (only markdown supported)')
    parser.add_argument('--reset', action='store_true', 
                        help='Reset and rebuild entire vector database (deletes storage folder)')
    
    args = parser.parse_args()
    
    # Initialize embeddings and vector store
    print("🔧 Initializing OpenAI embeddings and Qdrant vector store...")
    embedder = OpenAIEmbeddings()
    store = QdrantVectorStore()
    
    # Reset database if requested
    if args.reset:
        print("🗑️  Resetting vector database (deleting storage folder)...")
        store.reset_database()
    
    # Process file if provided
    if args.file:
        print(f"\n📄 Processing file: {args.file}")
        
        # Extract chunks
        chunks, char_count = process_file(args.file, args.type)
        print(f"✂️  Extracted {len(chunks)} chunks ({char_count} characters)")
        
        # Generate embeddings (batch)
        print("🔮 Generating embeddings...")
        texts = [chunk["content"] for chunk in chunks]
        embeddings = embedder.embed_texts(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Store in Qdrant
        print("💾 Storing in vector database...")
        store.add_documents(chunks, embeddings)
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"   Total documents in DB: {store.count_documents()}")
        print(f"   Section types: {set(c['section_type'] for c in chunks)}")
    else:
        # Show current state
        print(f"\n📊 Current database state:")
        print(f"   Total documents: {store.count_documents()}")
    
    print("\n✨ Done!")


if __name__ == "__main__":
    main()
