"""Inside Airbnb data discovery and download utilities."""

from __future__ import annotations

import re
from pathlib import Path

import requests

from config import INSIDE_AIRBNB_URL


def find_latest_listings_url(city: str) -> str:
    """Discover the most recent detailed-listings CSV URL for a city.

    Scrapes the Inside Airbnb *get-the-data* page for all
    ``listings.csv.gz`` links matching *city*, then returns the one
    with the newest date stamp.

    Args:
        city: City name as it appears in Inside Airbnb URLs
              (e.g. ``"london"``).

    Returns:
        Full URL to the latest ``listings.csv.gz`` file.

    Raises:
        RuntimeError: When no listing URL is found for the city.
    """
    response = requests.get(INSIDE_AIRBNB_URL, timeout=30)
    response.raise_for_status()

    pattern = re.compile(
        rf"https?://data\.insideairbnb\.com/[^\"]*/"
        rf"{re.escape(city)}/(\d{{4}}-\d{{2}}-\d{{2}})/data/listings\.csv\.gz",
        flags=re.IGNORECASE,
    )

    dated_links: list[tuple[str, str]] = [
        (match.group(1), match.group(0)) for match in pattern.finditer(response.text)
    ]

    if not dated_links:
        raise RuntimeError(
            f"No detailed-listings URL found for '{city}' " "on the Inside Airbnb page."
        )

    dated_links.sort(key=lambda item: item[0], reverse=True)
    return dated_links[0][1]


def download_file(url: str, destination: Path) -> Path:
    """Stream-download a file from *url* to *destination*.

    Parent directories are created automatically.

    Args:
        url: Remote file URL.
        destination: Local path where the file will be written.

    Returns:
        The *destination* path (for chaining).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as output_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output_file.write(chunk)

    return destination
