"""
Constants for dataset generation
"""

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

USER_SIMULATOR_INITIAL_PROMPT_MESSAGES = [
    {
        "role": "system",
        "content": "You are role-playing as a user seeking career advice from a chatbot. Always respond using the JSON format.",
    },
    {
        "role": "user",
        "content": (
            "Here are the details of the user you are simulating:\n"
            + LINE_BREAK
            + "{user_details}\n"
            + LINE_BREAK
            + "You aim to engage with a chatbot to explore career guidance in the following areas:\n"
            "1. Career direction in the next 5 years\n"
            "2. Career direction in the next 10 years\n"
            "3. Essential skills for career growth\n"
            "4. Relevant certifications to obtain\n"
            "Start the conversation by asking for short-term career suggestions, either explicitly mentioning your age and job title or subtly hinting at them. As the discussion evolves, progressively share more personal details to obtain tailored advice and deeper insights.\n\n"
            'Always respond using the following JSON format: {{ "proposed_question": ...}}'
        ),
    },
]

USER_SIMULATOR_SUBSEQUENT_PROMPT_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are role-playing as a user seeking career advice from a chatbot. "
            "Review the conversation history between you (the user) and the chatbot, then continue the role-play. "
            "Always respond in JSON format."
        ),
    },
    {"role": "user", "content": "Conversation history: \n{conversation_history}\n"},
    {
        "role": "user",
        "content": (
            "Here are the details of the user you are simulating:\n"
            + LINE_BREAK
            + "{user_details}\n"
            + LINE_BREAK
            + "Through the conversation, your goal is to seek career guidance in the following areas:\n"
            "1. Career direction in the next 5 years\n"
            "2. Career direction in the next 10 years\n"
            "3. Essential skills for career growth\n"
            "4. Relevant certifications to obtain\n"
            "If all four areas are addressed, conclude the conversation and leave the proposed question as the empty string. Otherwise, generate a follow-up question to gather more insights.\n"
            "Feel free to share additional details about yourself in your follow-up questions, either based on the chatbot’s responses or as needed for more tailored advice.\n\n"
            'Always respond using the following JSON format: {{ "proposed_question": ...,"end_conversation": true/false}}'
        ),
    },
]

CONVERSATION_TEMPLATE_STRING = """
{{ line_break }}
{% for user_msg, chatbot_msg in conversation_history %}
User: {{ user_msg }}
Chatbot: {{ chatbot_msg }}
{{ line_break }}
{% endfor %}
"""

DIALOGUE_RUNS_THRESHOLD = 10
