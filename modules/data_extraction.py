"""LinkedIn profile data extraction via ProxyCurl API or mock JSON."""

import requests
import logging
from typing import Dict, Optional, Any

import config

logger = logging.getLogger(__name__)

_UNWANTED_KEYS = {
    "similarly_named_profiles",
    "people_also_viewed",
    "certifications",
    "activities",
    "recommendations",
    "accomplishment_courses",
}


def _clean_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    """Strip unwanted keys and empty values from raw profile data."""
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
    """Fetch and clean a LinkedIn profile.

    Uses mock JSON when ``mock=True``, otherwise calls the ProxyCurl API.
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