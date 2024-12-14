from os import path

PROFILE_KEYS = [
    "Gender",
    "Date of Birth",
    "Country",
    "Education Level",
    "Years of Experience",
    "Job Title",
]

PROFILE_TEMPLATE = """
{% for key, value in profile_data.items() %}
{{ key }}: {{ value }};
{% endfor %}
"""

LINE_BREAK = "*" * 50 + "\n"

OPTIONS_TEMPLATE = """
{% for option in option_list %}
{{ option }};
{% endfor %}
"""

DIRECT_VALUE_SELECTION_PROMPT = [
    {
        "role": "system",
        "content": "You are a chatbot designed to provide precise and personalized answers to questions based on the given user profile.  Analyze the question carefully and tailor your response to match the user's context.",
    },
    {
        "role": "user",
        "content": (
            "Here are the details of the user profile:\n"
            + LINE_BREAK
            + "{user_details}\n"
            + LINE_BREAK
        ),
    },
    {
        "role": "user",
        "content": (
            "Below is the question:\n" + LINE_BREAK + "{question}\n" "{option_list}"
        ),
    },
]

CONVERSATION_HISTORY_PROMPT = [
    {
        "role": "user",
        "content": "Help me find the most sutiable answer to the following question.",
    },
    {
        "role": "user",
        "content": (
            "Below is the question:\n" + LINE_BREAK + "{question}\n" "{option_list}"
        ),
    },
]


main_dir = path.abspath(path.join(path.dirname(__file__), "..", ".."))

DIRECT_VALUE_QUESTIONS_CSV = path.join(
    main_dir,
    "datasets",
    "values_dataset",
    "vsm_direct_profile_questions_en_2024_Dec.csv",
)
DIALOGUE_CONTINUE_VALUE_QUESTIONS_CSV = path.join(
    main_dir,
    "datasets",
    "values_dataset",
    "vsm_dialogue_continue_questions_en_2024_Dec.csv",
)

DEFAULT_OPTION_IDS = [1, 2, 3, 4, 5]
