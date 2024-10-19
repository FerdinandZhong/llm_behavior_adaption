LLM_CHAT_MESSAGES = [
    {
        "role": "system",
        "content": "You're a reliable AI assistant.",
    }
]

SAMPLE_USER_CONTENT_TEMPLATE = (
    "Given the following dialogue history between two users:\n{dialogue_history}\n\n"
    "Select five persona attributes of {user} from the following list:\n{attributes_candidates}\n\n"
    "Only respond with attributes ids."
)
