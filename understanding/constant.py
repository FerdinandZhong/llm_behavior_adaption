import os

LLM_CHAT_MESSAGES = [
    {
        "role": "system",
        "content": "You're a reliable AI assistant.",
    }
]

SAMPLE_USER_CONTENT_TEMPLATE_FIVE = (
    "Given the following dialogue history between two users:\n{dialogue_history}\n\n"
    "Select five persona attributes of {user} from the following list:\n{attributes_candidates}\n\n"
    "Only respond with attributes ids."
)

SAMPLE_USER_CONTENT_TEMPLATE_SINGLE = (
    "Given the following dialogue history between two users:\n{dialogue_history}\n\n"
    "Select one most suitable persona attribute of {user} from the following list:\n{attributes_candidates}\n\n"
    "Only respond with attributes id."
)


DATASETS_FOLDER = os.path.join(os.getcwd(), "datasets")
OUTPUTS_FOLDER = os.path.join(os.getcwd(), os.getenv("output_folder", "outputs"))
DATASET_NAME = "personas_candidates.csv"
