"""Generation utils file
"""

from datetime import datetime

from jinja2 import Template

from .constant import PROFILE_KEYS


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
    age = (
        today.year
        - dob_date.year
        - ((today.month, today.day) < (dob_date.month, dob_date.day))
    )
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
