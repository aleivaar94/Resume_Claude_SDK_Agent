# %%
import requests
import json
import os
from dotenv import load_dotenv
import time
from typing import Dict, Any, Optional, Literal
import pandas as pd
from urllib.parse import urlparse

# Load environment variables
load_dotenv()

brightdata_api_key = os.getenv("BRIGHTDATA_API_KEY")
if not brightdata_api_key:
    raise ValueError("BRIGHTDATA_API_KEY environment variable is required. Please check your .env file.")

# %%
def get_brightdata_snapshot_linkedin(job_url, api_key):
    """
    Triggers a BrightData snapshot for a LinkedIn job posting.
    
    Parameters
    ----------
    job_url : str
        LinkedIn job posting URL to scrape.
    api_key : str
        BrightData API authentication key.
    
    Returns
    -------
    str
        Snapshot ID for tracking the scraping job.
    """
    api_url = "https://api.brightdata.com/datasets/v3/trigger"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    params = {
        "dataset_id": "gd_lpfll7v5hcqtkxl6l",
        "include_errors": "true",
    }
    data = {
        "input": [{"url": job_url}],
        "custom_output_fields": [
            "url",
            "job_posting_id",
            "job_title",
            "company_name",
            "company_id",
            "job_location",
            "job_summary",
            "company_url",
            "job_posted_date",
            "job_poster",
        ],
    }

    response = requests.post(api_url, headers=headers, params=params, json=data)
    response_json = response.json()
    print(f"LinkedIn Snapshot ID: {response_json['snapshot_id']}")
    return response_json['snapshot_id']


def get_brightdata_snapshot_indeed(job_url, api_key):
    """
    Triggers a BrightData snapshot for an Indeed job posting.
    
    Parameters
    ----------
    job_url : str
        Indeed job posting URL to scrape.
    api_key : str
        BrightData API authentication key.
    
    Returns
    -------
    str
        Snapshot ID for tracking the scraping job.
    """
    api_url = "https://api.brightdata.com/datasets/v3/trigger"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    params = {
        "dataset_id": "gd_l4dx9j9sscpvs7no2",
        "include_errors": "true",
    }
    data = {
        "input": [{"url": job_url}],
        "custom_output_fields": [
            "jobid",
            "company_name", 
            "date_posted_parsed",
            "job_title",
            "description_text",
            "company_link",
        ]
    }

    response = requests.post(api_url, headers=headers, params=params, json=data)
    response_json = response.json()
    print(f"Indeed Snapshot ID: {response_json['snapshot_id']}")
    return response_json['snapshot_id']

# %%
def get_snapshot_output(snapshot_id: str, api_key: str, max_retries: int = 40, wait_time: int = 5) -> dict:
    """
    Retrieves snapshot output from Bright Data API with automatic retry logic.
    
    Args:
        snapshot_id: The snapshot ID returned from triggering the dataset
        api_key: Bright Data API key
        max_retries: Maximum number of retry attempts (default: 5)
        wait_time: Wait time in seconds between retries (default: 3)

    Returns:
        dict: The extracted job data once ready
        
    Raises:
        TimeoutError: If snapshot is not ready after max_retries
        requests.RequestException: If API request fails
    """
    url = f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    params = {
        "format": "json"
    }
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Checking snapshot status...")
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            data = response.json()

            # Status values that indicate the snapshot is still processing
            processing_statuses = {"building", "running", "pending", "queued", "STATUS"}
            
            # Check if snapshot is still running
            if isinstance(data, dict) and data.get("status") in processing_statuses:
                print(f"Snapshot still processing. Waiting {wait_time} seconds...")
                if attempt < max_retries - 1:  # Don't sleep on the last attempt
                    time.sleep(wait_time)
                continue
            
            # Data is ready - return the parsed response
            print("Snapshot ready! Data retrieved successfully.")
            return json.dumps(data, indent=2, ensure_ascii=False)

        except requests.RequestException as e:
            print(f"API request failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
                continue
            raise
    
    # If we've exhausted all retries
    raise TimeoutError(f"Snapshot {snapshot_id} was not ready after {max_retries} attempts ({max_retries * wait_time} seconds)")

# %%
def extract_job_info_linkedin(json_output):
    """
    Extracts all fields from the Bright Data job snapshot JSON output, including job poster details as separate fields.
    
    Parameters
    ----------
    json_output : str or dict
        The JSON output from the Bright Data API, either as a string or a parsed dictionary.
    
    Returns
    -------
    dict
        A dictionary containing all extracted fields, including job poster details as separate fields.
    """
    # If data is a JSON string, parse it into a Python dict or list of dictionaries
    if isinstance(json_output, str):
        import json as _json
        data = _json.loads(json_output)
    else:
        data = json_output

    # If the data is a list, take the first item
    if isinstance(data, list):
        job = data[0]
    else:
        job = data

    # Extract all fields, including nested fields
    result = {
        'url': job.get('url'),
        'job_posting_id': job.get('job_posting_id'),
        'job_title': job.get('job_title'),
        'company_name': job.get('company_name'),
        'company_id': job.get('company_id'),
        'job_location': job.get('job_location'),
        'job_description': job.get('job_summary'),
        'company_url': job.get('company_url'),
        'job_posted_date': job.get('job_posted_date'),
        # Extract job poster details as separate fields
        'job_poster_name': job.get('job_poster', {}).get('name'),
        'job_poster_title': job.get('job_poster', {}).get('title'),
        'job_poster_url': job.get('job_poster', {}).get('url'),
    }
    # Convert result dictionary to DataFrame
    result_df = pd.DataFrame([result])

    return result, result_df

# %%
def extract_job_info_indeed(json_output):
    """
    Extracts all fields from the Bright Data job snapshot JSON output for Indeed jobs.
    
    Parameters
    ----------
    json_output : str or dict
        The JSON output from the Bright Data API, either as a string or a parsed dictionary.
    
    Returns
    -------
    dict
        A dictionary containing all extracted fields.
    pd.DataFrame
        A DataFrame containing the extracted fields.
    """
    # If data is a JSON string, parse it into a Python dict or list of dictionaries
    if isinstance(json_output, str):
        import json as _json
        data = _json.loads(json_output)
    else:
        data = json_output

    # If the data is a list, take the first item
    if isinstance(data, list):
        job = data[0]
    else:
        job = data

    # Extract all fields
    result = {
        'jobid': job.get('jobid'),
        'company_name': job.get('company_name'),
        'date_posted_parsed': job.get('date_posted_parsed'),
        'job_title': job.get('job_title'),
        'job_description': job.get('description_text'),
        'company_link': job.get('company_link'),
    }

    # Convert result dictionary to DataFrame
    result_df = pd.DataFrame([result])

    return result, result_df

# %%
def extract_job(
    job_url_or_snapshot_id: str, 
    api_key: str, 
    platform: Optional[Literal["linkedin", "indeed"]] = None,
    max_retries: int = 360, 
    wait_time: int = 5
) -> tuple[dict, pd.DataFrame]:
    """
    Orchestrates the full job scraping process for LinkedIn or Indeed using Bright Data API.
    
    This function automatically detects the job platform (LinkedIn or Indeed) from the URL
    or retrieves data from an existing snapshot ID. When using a snapshot ID, the platform
    parameter must be specified.
    
    Parameters
    ----------
    job_url_or_snapshot_id : str
        The URL of the job posting OR a BrightData snapshot ID (starting with 's_').
    api_key : str
        Bright Data API key.
    platform : Optional[Literal["linkedin", "indeed"]], optional
        Platform identifier. Required when using snapshot ID, optional for URLs (auto-detected).
    max_retries : int, optional
        Maximum retries for snapshot polling (default: 360).
    wait_time : int, optional
        Wait time in seconds between retries (default: 5).
    
    Returns
    -------
    tuple[dict, pd.DataFrame]
        A tuple containing the extracted job data as a dictionary and a DataFrame.
    
    Raises
    ------
    ValueError
        If the platform cannot be detected, is unsupported, or is missing when using snapshot ID.
    
    Examples
    --------
    Extract from URL (auto-detect platform):
    >>> job_dict, job_df = extract_job("https://www.linkedin.com/jobs/view/12345", api_key)
    
    Extract from snapshot ID (must specify platform):
    >>> job_dict, job_df = extract_job("s_mjepdfj94zb4miakd", api_key, platform="linkedin")
    """
    # Detect if input is snapshot ID or URL
    is_snapshot_id = job_url_or_snapshot_id.startswith("s_")
    
    if is_snapshot_id:
        # Snapshot ID flow - platform must be specified
        if not platform:
            raise ValueError(
                "Platform parameter is required when using snapshot ID. "
                "Specify platform='linkedin' or platform='indeed'."
            )
        snapshot_id = job_url_or_snapshot_id
        print(f"Using existing snapshot ID: {snapshot_id} for platform: {platform}")
    else:
        # URL flow - detect platform from URL if not provided
        if not platform:
            parsed_url = urlparse(job_url_or_snapshot_id.lower())
            domain = parsed_url.netloc
            
            if 'linkedin.com' in domain:
                platform = 'linkedin'
            elif 'indeed.com' in domain:
                platform = 'indeed'
            else:
                raise ValueError(
                    f"Unsupported platform. Only LinkedIn and Indeed are supported. "
                    f"Detected domain: {domain}"
                )
        
        # Trigger snapshot creation
        if platform == "linkedin":
            snapshot_id = get_brightdata_snapshot_linkedin(job_url_or_snapshot_id, api_key)
        elif platform == "indeed":
            snapshot_id = get_brightdata_snapshot_indeed(job_url_or_snapshot_id, api_key)
        else:
            raise ValueError(
                f"Invalid platform: {platform}. Use 'linkedin' or 'indeed'."
            )
    
    # Common flow: retrieve snapshot output and extract job info
    output = get_snapshot_output(snapshot_id, api_key, max_retries, wait_time)
    
    if platform == "linkedin":
        return extract_job_info_linkedin(output)
    elif platform == "indeed":
        return extract_job_info_indeed(output)
    else:
        raise ValueError(f"Invalid platform: {platform}. Use 'linkedin' or 'indeed'.")


