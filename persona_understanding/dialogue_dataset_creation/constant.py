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

CHATBOT_SYSTEM_PROMPT = {"role": "system", "content": "Answer the question concisely"}

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
            "If all four areas are addressed, conclude the conversation. Otherwise, generate a follow-up question to gather more insights.\n"
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

DIALOGUE_RUNS_THRESHOLD = 5

LLM_BASED_OOC_DETECTION_PROMPT = [
    {
        "role": "system",
        "content": (
            "You are an out-of-context detector. Compare the details in a user-provided question with the user profile. Ensure the question meets the following criteria:\n"
            "1. Reflects only information present in the profile. \n"
            "2. Is written in the first-person perspective and seeks career suggestions. \n"
            'If the question fails to meet these criteria, rewrite it accordingly. \nRespond in JSON format: {"has_out_of_context": true/false, "updated_question": "..."}. Leave "updated_question" empty if no discrepancies are found.'
        ),
    },
    {
        "role": "user",
        "content": (
            "User profile: \n"
            + LINE_BREAK
            + "{user_details}\n"
            + LINE_BREAK
            + "Question: \n"
            + LINE_BREAK
            + "{question}\n"
            + LINE_BREAK
        ),
    },
]


# LLM as Judge
# Question 1
JUDGE_QUESTIONS_NUM_ATTRIBUTES = [
    {
        "role": "system",
        "content": """You will be given a a set of questions and a persona attributes.
Your task is to provide a 'total rating' scoring how many attributes mentioned in the questions. The score should be given based on the exact number of mentioned attributes.
Give your answer as a float on a scale of 0 to 5, you don't need to care about the correctness of the attribute values.

Scoring details:
* no attributes mentioned --> 0
* mentioning 1 attribute --> 1
* mentioning 2 attributes --> 2
* mentioning 3 attributes --> 3
* mentioning 4 attributes --> 4
* mentioning 5 attributes --> 5
        
Always respond using the following JSON format: {{ "rating": int, "reason": str}}'
""",
    },
    {
        "role": "user",
        "content": (
            "Here are the details of the persona attributes:\n"
            + LINE_BREAK
            + "{user_details}\n"
            + LINE_BREAK
        ),
    },
    {
        "role": "user",
        "content": (
            "Here are the questions:\n" + LINE_BREAK + "{question_str}\n" + LINE_BREAK
        ),
    },
]

JUDGE_QUESTIONS_NUM_CORRECT_ATTRIBUTES = [
    {
        "role": "system",
        "content": """You will be given a a set of questions and a persona attributes.
Your task is to provide a 'total rating' scoring if attributes values are correctly mentioned in the questions.
Initial score is 5, for every wrong attribute value detected in the questions, deducting one point.
You don't need to deduct point for missing attributes.
Give your answer as a float on a scale of 0 to 5.

Scoring details:
* all mentioned attributes are having the correct values --> 5
* one attribute value is wrong --> 4
* two attribute values are wrong --> 3
* three attribute values are wrong --> 2
* four attribute values are wrong --> 1
* five attribute values are wrong --> 0
        
Always respond using the following JSON format: {{ "rating": int, "reason": str}}'

Write your reason concisely.
""",
    },
    {
        "role": "user",
        "content": (
            "Here are the details of the persona attributes:\n"
            + LINE_BREAK
            + "{user_details}\n"
            + LINE_BREAK
        ),
    },
    {
        "role": "user",
        "content": (
            "Here are the questions:\n" + LINE_BREAK + "{question_str}\n" + LINE_BREAK
        ),
    },
]

JUDGE_QUESTIONS_NUM_QUESTIONS_REPEATED = [
    {
        "role": "system",
        "content": """You will be given a a set of questions and a persona attributes.
Your task is to provide a 'total rating' scoring if questions are repeating the similar content, more repetitions, lower the score.
Please give a score based on the number of unique questions.
Give your answer as a float on a scale of 1 to 5.

Scoring details:
* all five questions are having the unique contents --> 5
* 4 unique contents, and 1 repeated question --> 4
* 3 unique contents and 2 repeated questions --> 3
* only 2 unique contents among 5 questions, 3 questions are having the repeated contents --> 2
* all five questions are having the same content --> 1
        
Always respond using the following JSON format: {{ "rating": int, "reason": str}}'

Write your reason concisely.
""",
    },
    {
        "role": "user",
        "content": (
            "Here are the details of the persona attributes:\n"
            + LINE_BREAK
            + "{user_details}\n"
            + LINE_BREAK
        ),
    },
    {
        "role": "user",
        "content": (
            "Here are the questions:\n" + LINE_BREAK + "{question_str}\n" + LINE_BREAK
        ),
    },
]


JUDGE_QUESTIONS_QUESTIONS_QUALITY = [
    {
        "role": "system",
        "content": """You will be given a a set of questions and a persona attributes.
Your task is to provide a 'total rating' for the readability of the questions.
All questions are about seeking the career advice from the chatbot.

Please give a score based on the readability of each question.
Give your answer as a float on a scale of 1 to 5.

Scoring details:
* Start with the score as 3.
* If there's any reactions to the chatbot's response in the questions --> plus 1 to the score.
* If the langauge is natural and the questions are organized in a progressive manner --> plus 1 to the score.
* If the language is not natural and the following questions keep repeating the user's self introduction --> minus 1 from the score.
* If the language is too polite and sounds like seeking the advice from seniors --> minus 1 from the socre.

Always respond using the following JSON format: {{ "rating": int, "reason": str}}'

Write your reason concisely.
""",
    },
    {
        "role": "user",
        "content": (
            "Here are the details of the persona attributes:\n"
            + LINE_BREAK
            + "{user_details}\n"
            + LINE_BREAK
        ),
    },
    {
        "role": "user",
        "content": (
            "Here are the questions:\n" + LINE_BREAK + "{question_str}\n" + LINE_BREAK
        ),
    },
]

LLM_JUDGE_DICT = {
    "judge_dim_1": JUDGE_QUESTIONS_NUM_ATTRIBUTES,
    "judge_dim_2": JUDGE_QUESTIONS_NUM_CORRECT_ATTRIBUTES,
    "judge_dim_3": JUDGE_QUESTIONS_NUM_QUESTIONS_REPEATED,
    "judge_dim_4": JUDGE_QUESTIONS_QUESTIONS_QUALITY
}
