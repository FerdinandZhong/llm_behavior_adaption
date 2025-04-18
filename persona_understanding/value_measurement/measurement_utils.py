"""Measurement utils file
"""

import numpy as np
import pycountry
import pycountry_convert as pc
from transformers import pipeline

from persona_understanding.value_measurement.constant import (
    COUNTRY_TO_CULTURE,
    DEVELOPMENT_LEVEL,
)


def get_continent(country_name):
    """
    Maps a given country name to its corresponding continent.

    Args:
        country_name (str): The name of the country.

    Returns:
        str: The corresponding continent name or an error message if not found.
    """
    try:
        # Get country alpha-2 code
        country = pycountry.countries.lookup(country_name)
        country_code = country.alpha_2

        # Get continent code
        continent_code = pc.country_alpha2_to_continent_code(country_code)

        # Map continent codes to full names
        continent_map = {
            "AF": "Africa",
            "AS": "Asia",
            "EU": "Europe",
            "NA": "America",
            "OC": "Unknown",  # "Oceania",
            "SA": "America",
            "AN": "Unknown",  # "Antarctica",
        }

        return continent_map.get(continent_code, "Unknown")

    except LookupError:
        if country_name == "Micronesia":
            # return "Oceania"
            return "Unknown"
        elif country_name == "pitcairn islands":
            # return "Oceania"
            return "Unknown"
        return "Unknown"


def get_culture(country_name):
    """
    Maps a given country name to its corresponding continent.

    Args:
        country_name (str): The name of the country.

    Returns:
        str: The corresponding continent name or an error message if not found.
    """
    try:
        # Get country alpha-2 code
        country = pycountry.countries.lookup(country_name)
        iso3 = country.alpha_3
        return COUNTRY_TO_CULTURE.get(iso3, "Other")

    except LookupError:
        if country_name == "Micronesia":
            return "Oceania"
        elif country_name == "pitcairn islands":
            return "Oceania"

        return "Other"


def get_development_level(country_name):
    """
    Maps a given country name to its corresponding continent.

    Args:
        country_name (str): The name of the country.

    Returns:
        str: The corresponding continent name or an error message if not found.
    """
    try:
        # Get country alpha-2 code
        country = pycountry.countries.lookup(country_name)
        iso3 = country.alpha_3
        return DEVELOPMENT_LEVEL.get(iso3, "Unknown")

    except LookupError:
        return "Unknown"


class JobClassifier:
    """Zero shot classifier for job titles"""

    position_levels = [
        "Entry Level",
        "Professional",
        "Management",
        "Senior Management",
        "C-Suite",
    ]
    sector_levels = [
        "Tech",
        "Finance",
        "Sales",
        "Marketing",
        "Operations",
        "Human Resources",
        "Healthcare",
        "Design",
    ]  # TODO: to update the list

    job_categories = [
        "Healthcare & Therapy",
        "Science, Environment & Research",
        "Creative Arts & Media",
        "Education & Social Services",
        "Business, Finance & Administration",
        "Technology & Engineering",
        "Hospitality, Events & Leisure"
    ]

    def __init__(self):
        self.classifier = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli"
        )

    def get_position_level(self, job_title):
        result = self.classifier(job_title, self.job_categories)

        selected_lable = result["labels"][np.array(result["scores"]).argmax()]
        return selected_lable
