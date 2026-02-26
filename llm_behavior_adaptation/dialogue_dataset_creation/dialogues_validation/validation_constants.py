"""
Constants and prompt templates for dialogue validation.

Scoring Rubric:
1. Attribute Coverage (0-5): Count of demographic attributes mentioned
2. Attribute Correctness (0-5): Count of correctly referenced attributes
3. Question Diversity (1-5): Variety of topics and question types
4. Conversational Relevance (1-5): Contextual appropriateness and flow
5. Naturalness (1-5): Organic embedding of attributes
"""

LINE_BREAK = "*" * 50 + "\n"

# ============================================================================
# Individual Criterion Prompts (for detailed evaluation)
# ============================================================================

JUDGE_ATTRIBUTE_COVERAGE = [
    {
        "role": "system",
        "content": """You are evaluating the attribute coverage of a generated dialogue.

Your task is to count how many demographic attributes are explicitly or implicitly
mentioned in the dialogue. The target attributes are:
1. Age (or age-related hints like "recent graduate", "mid-career", "near retirement")
2. Education level (degree, qualifications, academic background)
3. Occupation/Job title (current role, profession, industry)
4. Socioeconomic status (income hints, lifestyle indicators, class markers)
5. Geographic location (country, city, region)

Score 0-5 based on how many distinct attributes are inferable from the dialogue.

Always respond using JSON format: {"rating": int, "reason": str, "detected_attributes": [list of detected attributes]}
""",
    },
    {
        "role": "user",
        "content": "Ground truth user profile:\n" + LINE_BREAK + "{user_profile}\n" + LINE_BREAK,
    },
    {
        "role": "user",
        "content": "Dialogue to evaluate:\n" + LINE_BREAK + "{dialogue}\n" + LINE_BREAK,
    },
]

JUDGE_ATTRIBUTE_CORRECTNESS = [
    {
        "role": "system",
        "content": """You are evaluating the attribute correctness of a generated dialogue.

Your task is to verify whether the demographic attributes mentioned in the dialogue
CORRECTLY match the ground truth user profile. This score is bounded by coverage:
if only 3 attributes are mentioned, the maximum correctness score is 3.

For each mentioned attribute, check if its value matches the ground truth profile.
- Correct match: count as correct
- Incorrect value: do not count
- Attribute not mentioned: do not count (neither correct nor incorrect)

Score 0-5 based on how many mentioned attributes have CORRECT values.

Always respond using JSON format: {"rating": int, "reason": str, "correct_attributes": [...], "incorrect_attributes": [...]}
""",
    },
    {
        "role": "user",
        "content": "Ground truth user profile:\n" + LINE_BREAK + "{user_profile}\n" + LINE_BREAK,
    },
    {
        "role": "user",
        "content": "Dialogue to evaluate:\n" + LINE_BREAK + "{dialogue}\n" + LINE_BREAK,
    },
]

JUDGE_QUESTION_DIVERSITY = [
    {
        "role": "system",
        "content": """You are evaluating the question diversity of a generated dialogue.

Your task is to assess the variety of topics and question types across the dialogue turns.
Consider:
- Number of distinct subtopics covered
- Variety in question framing (advice-seeking, information-gathering, opinion-based)
- Progressive depth vs. repetitive surface-level questions

Scoring criteria:
* 5: Questions span ≥4 distinct subtopics with varied question types
* 4: 3 distinct subtopics with some variety in framing
* 3: 2 subtopics; questions show moderate variation
* 2: Single subtopic but different phrasings
* 1: Repetitive questions on a single narrow topic

Always respond using JSON format: {"rating": int, "reason": str, "subtopics_identified": [...]}
""",
    },
    {
        "role": "user",
        "content": "Dialogue to evaluate:\n" + LINE_BREAK + "{dialogue}\n" + LINE_BREAK,
    },
]

JUDGE_CONVERSATIONAL_RELEVANCE = [
    {
        "role": "system",
        "content": """You are evaluating the conversational relevance of a generated dialogue.

Your task is to assess whether the user's questions remain contextually appropriate
for the dialogue topic (e.g., career advice, investment advice) and maintain coherent
flow with prior assistant responses.

Consider:
- Are questions on-topic for the stated scenario?
- Do follow-up questions logically build on previous responses?
- Is there natural conversational progression?

Scoring criteria:
* 5: All questions are topically appropriate and logically follow from prior context
* 4: Minor tangents but overall coherent progression
* 3: Some questions feel disconnected but remain on-topic
* 2: Frequent topic drift; weak connection to prior turns
* 1: Questions are off-topic or contextually incoherent

Always respond using JSON format: {"rating": int, "reason": str}
""",
    },
    {
        "role": "user",
        "content": "Dialogue topic: {dialogue_topic}\n",
    },
    {
        "role": "user",
        "content": "Dialogue to evaluate:\n" + LINE_BREAK + "{dialogue}\n" + LINE_BREAK,
    },
]

JUDGE_NATURALNESS = [
    {
        "role": "system",
        "content": """You are evaluating the naturalness of attribute embedding in a generated dialogue.

Your task is to assess how organically demographic attributes are woven into the
conversation, rather than being artificially inserted.

Consider:
- Are attributes revealed through context (e.g., "as a recent graduate" vs. "I am 22 years old")
- Does the information sharing feel natural for the conversation flow?
- Would a real user share information this way?

Scoring criteria:
* 5: Attributes emerge naturally through context; feels like authentic conversation
* 4: Mostly natural with minor explicit statements
* 3: Mix of natural embedding and direct declarations
* 2: Attributes mostly stated explicitly and awkwardly
* 1: Attributes feel forced or artificially injected (e.g., listing profile at start)

Always respond using JSON format: {"rating": int, "reason": str, "natural_examples": [...], "awkward_examples": [...]}
""",
    },
    {
        "role": "user",
        "content": "Dialogue to evaluate:\n" + LINE_BREAK + "{dialogue}\n" + LINE_BREAK,
    },
]

# ============================================================================
# Combined Evaluation Prompt (for efficiency - single LLM call)
# ============================================================================

JUDGE_COMBINED_EVALUATION = [
    {
        "role": "system",
        "content": """You are evaluating the quality of a generated dialogue for persona-based conversation.

Evaluate the dialogue on these five criteria and provide integer scores:

1. **Attribute Coverage (0-5)**: Count of demographic attributes (age, education, occupation,
   SES, location) that are mentioned or inferable from the dialogue.

2. **Attribute Correctness (0-5)**: Of the attributes mentioned, how many correctly match
   the ground truth profile? (Bounded by coverage score)

3. **Question Diversity (1-5)**:
   - 5: ≥4 distinct subtopics, varied question types
   - 4: 3 subtopics with variety
   - 3: 2 subtopics, moderate variation
   - 2: Single subtopic, different phrasings
   - 1: Repetitive single-topic questions

4. **Conversational Relevance (1-5)**:
   - 5: All questions appropriate and logically follow context
   - 4: Minor tangents but coherent
   - 3: Some disconnection but on-topic
   - 2: Frequent drift, weak connections
   - 1: Off-topic or incoherent

5. **Naturalness (1-5)**:
   - 5: Attributes emerge organically through context
   - 4: Mostly natural with minor explicit statements
   - 3: Mix of natural and explicit
   - 2: Mostly awkward/explicit declarations
   - 1: Forced/artificial attribute insertion

Respond in JSON format:
{
    "coverage": <0-5>,
    "correctness": <0-5>,
    "diversity": <1-5>,
    "relevance": <1-5>,
    "naturalness": <1-5>,
    "reasoning": {
        "coverage": "<explanation>",
        "correctness": "<explanation>",
        "diversity": "<explanation>",
        "relevance": "<explanation>",
        "naturalness": "<explanation>"
    }
}
""",
    },
    {
        "role": "user",
        "content": "Ground truth user profile:\n" + LINE_BREAK + "{user_profile}\n" + LINE_BREAK,
    },
    {
        "role": "user",
        "content": "Dialogue topic: {dialogue_topic}\n",
    },
    {
        "role": "user",
        "content": "Dialogue to evaluate:\n" + LINE_BREAK + "{dialogue}\n" + LINE_BREAK,
    },
]

# ============================================================================
# Criterion Mapping
# ============================================================================

VALIDATION_CRITERIA = {
    "coverage": JUDGE_ATTRIBUTE_COVERAGE,
    "correctness": JUDGE_ATTRIBUTE_CORRECTNESS,
    "diversity": JUDGE_QUESTION_DIVERSITY,
    "relevance": JUDGE_CONVERSATIONAL_RELEVANCE,
    "naturalness": JUDGE_NATURALNESS,
    "combined": JUDGE_COMBINED_EVALUATION,
}

CRITERION_DESCRIPTIONS = {
    "coverage": "Attribute Coverage (0-5): Number of demographic attributes mentioned",
    "correctness": "Attribute Correctness (0-5): Number of correctly referenced attributes",
    "diversity": "Question Diversity (1-5): Variety of topics and question types",
    "relevance": "Conversational Relevance (1-5): Contextual appropriateness and flow",
    "naturalness": "Naturalness (1-5): Organic embedding of attributes",
}

# Score ranges for validation
SCORE_RANGES = {
    "coverage": (0, 5),
    "correctness": (0, 5),
    "diversity": (1, 5),
    "relevance": (1, 5),
    "naturalness": (1, 5),
}
