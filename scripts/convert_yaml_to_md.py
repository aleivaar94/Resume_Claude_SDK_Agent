"""
Convert resume_ale.yaml to resume_ale.md in markdown format.

This script performs a one-time conversion of the structured YAML resume
into a human-readable markdown format suitable for RAG chunking and embeddings.

Usage
-----
python scripts/convert_yaml_to_md.py
"""

import yaml
import os
from pathlib import Path


def load_yaml(file_path: str) -> dict:
    """
    Load YAML file and return as dictionary.
    
    Parameters
    ----------
    file_path : str
        Path to the YAML file.
    
    Returns
    -------
    dict
        Parsed YAML content.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def format_work_experience(experience: list) -> str:
    """
    Format work experience entries as markdown.
    
    Parameters
    ----------
    experience : list
        List of work experience dictionaries.
    
    Returns
    -------
    str
        Formatted markdown text.
    """
    md = "## Work Experience\n\n"
    
    for job in experience:
        # Header: Position | Company
        md += f"### {job['position']} | {job['company']}\n"
        # Metadata line: Dates | Location | Industry
        md += f"**{job['start_date']} - {job['end_date']}** | {job['location']} | {job['industry']}\n\n"
        
        # Achievements as bullet points
        for achievement in job['achievements']:
            md += f"- {achievement}\n"
        md += "\n"
    
    return md


def format_education(education: dict) -> str:
    """
    Format education entries as markdown.
    
    Parameters
    ----------
    education : dict
        Dictionary containing degrees and continuing studies.
    
    Returns
    -------
    str
        Formatted markdown text.
    """
    md = "## Education\n\n"
    
    for degree in education['degrees']:
        md += f"### {degree['degree']} | {degree['institution']}\n"
        md += f"**{degree['start_date']} - {degree['end_date']}** | {degree['country']}\n\n"
    
    if 'continuing_studies' in education and education['continuing_studies']:
        md += "## Continuing Studies\n\n"
        for study in education['continuing_studies']:
            md += f"### {study['name']} | {study['institution']}\n"
            md += f"**Completed: {study['completion_date']}**\n\n"
    
    return md


def format_skills(skills: dict) -> str:
    """
    Format skills as markdown.
    
    Parameters
    ----------
    skills : dict
        Dictionary of skill categories.
    
    Returns
    -------
    str
        Formatted markdown text.
    """
    md = "## Skills\n\n"
    
    for category, items in skills.items():
        # Convert category key to readable title
        title = category.replace('_', ' ').title()
        md += f"**{title}**: {', '.join(items)}\n\n"
    
    return md


def format_personal_info(info: dict) -> str:
    """
    Format personal information as markdown.
    
    Parameters
    ----------
    info : dict
        Personal information dictionary.
    
    Returns
    -------
    str
        Formatted markdown text.
    """
    md = f"# {info['full_name']}\n\n"
    md += f"**Email**: {info['email']} | **GitHub**: {info['github']} | **LinkedIn**: {info['linkedin']}\n\n"
    md += f"**Languages**: {info['languages']}\n\n"
    md += f"**Industry Experience**: {info['Industry Experience']}\n\n"
    
    return md


def format_professional_summary(summary: dict) -> str:
    """
    Format professional summary as markdown paragraph.
    
    Parameters
    ----------
    summary : dict
        Professional summary dictionary with personality traits.
    
    Returns
    -------
    str
        Formatted markdown text.
    """
    md = "## Professional Summary\n\n"
    
    # Combine traits into a paragraph
    traits = summary['personality_traits']
    md += ' '.join(traits) + "\n\n"
    
    return md


def convert_yaml_to_markdown(yaml_path: str, output_path: str) -> None:
    """
    Convert YAML resume to markdown format.
    
    Parameters
    ----------
    yaml_path : str
        Path to input YAML file.
    output_path : str
        Path to output markdown file.
    
    Notes
    -----
    Creates output directory if it doesn't exist.
    """
    # Load YAML
    resume = load_yaml(yaml_path)
    
    # Build markdown content
    markdown = ""
    markdown += format_personal_info(resume['personal_information'])
    markdown += format_professional_summary(resume['professional_summary'])
    markdown += format_work_experience(resume['work_experience'])
    markdown += format_education(resume['education'])
    markdown += format_skills(resume['skills'])
    
    # Write to file
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown)
    
    print(f"✅ Converted {yaml_path} → {output_path}")
    print(f"📄 Generated markdown file with {len(markdown)} characters")


if __name__ == "__main__":
    # Define paths
    project_root = Path(__file__).parent.parent
    yaml_file = project_root / "data" / "resume_ale.yaml"
    md_file = project_root / "data" / "resume_ale.md"
    
    # Convert
    convert_yaml_to_markdown(str(yaml_file), str(md_file))
