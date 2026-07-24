"""Module for extracting LinkedIn profile data."""

import time
import requests
import logging
from typing import Dict, Optional, Any

import config

logger = logging.getLogger(__name__)

# Keys to strip from the profile to reduce noise in the vector store
_UNWANTED_KEYS = {
    "similarly_named_profiles",
    "people_also_viewed",
    "certifications",
    "activities",
    "recommendations",
    "accomplishment_courses",
}


def _clean_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove empty values and unwanted keys from profile data."""
    cleaned = {}
    for key, value in data.items():
        if key in _UNWANTED_KEYS:
            continue
        if value is None or value == "" or value == [] or value == {}:
            continue
        cleaned[key] = value
    return cleaned


def extract_linkedin_profile(
    linkedin_profile_url: str,
    api_key: Optional[str] = None,
    mock: bool = False,
) -> Dict[str, Any]:
    """Extract LinkedIn profile data using ProxyCurl API or loads a premade JSON file.

    Args:
        linkedin_profile_url: The LinkedIn profile URL to extract data from.
        api_key: ProxyCurl API key. Required if mock is False.
        mock: If True, loads mock data from a premade JSON file instead of using the API.

    Returns:
        Dictionary containing the LinkedIn profile data.
    """
    if mock:
        logger.info("Loading mock LinkedIn profile from: %s", config.MOCK_DATA_URL)
        response = requests.get(config.MOCK_DATA_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
    else:
        if not api_key:
            raise ValueError("A ProxyCurl API key is required when not using mock data.")

        logger.info("Fetching LinkedIn profile via ProxyCurl: %s", linkedin_profile_url)
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"url": linkedin_profile_url}
        response = requests.get(
            "https://nubela.co/proxycurl/api/v2/linkedin",
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

    cleaned = _clean_profile(data)
    logger.info("Profile extracted and cleaned (%d top-level keys).", len(cleaned))
    return cleaned