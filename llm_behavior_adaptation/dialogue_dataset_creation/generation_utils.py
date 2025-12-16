"""Generation utils file"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from jinja2 import Template

from .wvs_dataset_constants import JSON_TEMPLATE, PROFILE_KEYS


def render_json(json_input: dict) -> str:
    serialized_json_input = json.dumps(json_input, indent=2, sort_keys=True, ensure_ascii=False)
    return Template(JSON_TEMPLATE).render(json_input=serialized_json_input)


def load_json_folder(
    folder: str | Path,
    pattern: str = "*.json",
    recursive: bool = False,
    key_style: str = "stem",  # "stem", "name", or "relpath"
    on_error: str = "raise",  # "raise", "warn", or "ignore"
) -> Dict[str, Any]:
    folder = Path(folder)
    files = folder.rglob(pattern) if recursive else folder.glob(pattern)
    out: Dict[str, Any] = {}

    for f in files:
        try:
            obj = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            if on_error == "raise":
                raise
            elif on_error == "warn":
                print(f"Warning: skipping {f}: {e}")
                continue
            else:  # ignore
                continue

        if key_style == "stem":
            key = f.stem
        elif key_style == "name":
            key = f.name
        elif key_style == "relpath":
            key = str(f.relative_to(folder))
        else:
            raise ValueError("key_style must be 'stem', 'name', or 'relpath'")

        out[key] = obj
    return out


def calculate_age(dob):
    """
    Calculate the age based on the date of birth.

    Args:
        dob (str): The date of birth in the format "dd-mm-yyyy".

    Returns:
        int: The age calculated based on the current date.
    """
    dob_date = datetime.strptime(dob, "%d-%M-%Y")
    today = datetime.today()
    print(today)
    # today = datetime.strptime("02-01-2025", "%d-%M-%Y")
    # today = datetime.strptime("31-03-2025", "%d-%M-%Y")
    age = today.year - dob_date.year
    if today.month < dob_date.month:
        if today.day < dob_date.day:
            age -= 1
    return age


# Function to prepare the user profile
def retrieve_user_profile(row, profile_keys=PROFILE_KEYS):
    """
    Retrieve and format the user profile, calculating age for the 'Date of Birth' field.

    Args:
        row (dict): A dictionary containing user profile data with keys corresponding to profile fields.
        profile_keys (list, optional): A list of profile keys to retrieve from the row. Defaults to PROFILE_KEYS.

    Returns:
        dict: A dictionary with profile field names as keys and their corresponding values, including the calculated age.
    """
    # Create a dictionary with processed values
    profile_data = {}
    for key in profile_keys:
        if key == "Date of Birth":
            profile_data["Age"] = calculate_age(row[key])
        else:
            profile_data[key] = row[key]
    return profile_data


def render_template(template_str, **kwargs):
    """
    Renders a Jinja2 template with the provided context.

    Args:
        template_str (str): The Jinja2 template as a string.
        **kwargs: Arbitrary keyword arguments to be passed as context to the template.

    Returns:
        str: The rendered template with the context applied.

    Example:
        template = "Hello, {{ name }}!"
        context = {'name': 'Alice'}
        rendered = render_template(template, **context)
        print(rendered)  # Output: "Hello, Alice!"
    """
    # Create a Jinja2 Template object from the template string
    template = Template(template_str)
    # Render the template with the provided keyword arguments (context)
    rendered_profile = template.render(**kwargs)

    return rendered_profile


def retrieve_user_profile_wvs(row, profile_keys=PROFILE_KEYS):
    """
    Retrieve and format the user profile, calculating age for the 'Date of Birth' field.

    Args:
        row (dict): A dictionary containing user profile data with keys corresponding to profile fields.
        profile_keys (list, optional): A list of profile keys to retrieve from the row. Defaults to PROFILE_KEYS.

    Returns:
        dict: A dictionary with profile field names as keys and their corresponding values, including the calculated age.
    """
    # Create a dictionary with processed values
    profile_data = {}
    for key in profile_keys:
        profile_data[key] = row[key]
    return profile_data
