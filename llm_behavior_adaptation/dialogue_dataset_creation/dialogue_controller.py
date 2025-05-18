"""
Dialogue controller
"""

import json
import logging
import os
from copy import deepcopy

from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from llm_behavior_adaptation.dialogue_dataset_creation.constant import (
    CHATBOT_SYSTEM_PROMPT,
    CONVERSATION_TEMPLATE_STRING,
    DIALOGUE_RUNS_THRESHOLD,
    LLM_BASED_OOC_DETECTION_PROMPT,
    PROFILE_TEMPLATE,
    USER_SIMULATOR_INITIAL_PROMPT_MESSAGES,
    USER_SIMULATOR_SUBSEQUENT_PROMPT_MESSAGES,
)
from llm_behavior_adaptation.dialogue_dataset_creation.generation_utils import (
    render_template,
    retrieve_user_profile,
)

logger = logging.getLogger(__name__)


class DialogueRun(BaseModel):
    user_content: str
    chatbot_content: str

    def convert_to_user_simulator_format(self):
        return (self.user_content, self.chatbot_content)

    def convert_to_openai_history(self):
        return [
            {"role": "user", "content": self.user_content},
            {"role": "assistant", "content": self.chatbot_content},
        ]


class DialogueGenerator:
    def __init__(
        self,
        user_simulator="gpt-4o",
        chatbot="gpt-4o",
        ooc_detector=None,
        ooc_detector_type="llm",
        openai_client=None,
        user_simulator_generation_parameters=None,
        chatbot_generation_parameters=None,
        dialogue_runs_threshold: int = DIALOGUE_RUNS_THRESHOLD,
        verbose: int = 0,
    ) -> None:
        """
        Initialize the conversation management system with specified components.

        Args:
            user_simulator (str, optional): The model used as a user simulator. Defaults to "gpt-4o".
            chatbot (str, optional): The model used for chatbot interactions. Defaults to "gpt-4o".
            ooc_detector (optional): The out-of-character (OOC) detection mechanism. Defaults to None.
            ooc_detector_type (optional, str): The OOC detector type. Defaults to llm.
            openai_client (optional): An OpenAI client instance for API interactions. If None, a default
                `AsyncOpenAI` client is instantiated using the API key from the environment. Defaults to None.
            user_simulator_generation_parameters (dict, optional): Parameters for generating responses
                from the user simulator model. Defaults to an empty dictionary.
            chatbot_generation_parameters (dict, optional): Parameters for generating chatbot responses.
                Defaults to an empty dictionary.
            dialogue_runs_threshold (int, optional): The maximum number of dialogue runs allowed
                before termination or reinitialization. Defaults to `DIALOGUE_RUNS_THRESHOLD`.
            verbose (int, optional): Logging verbosity level.
                0 = Only log errors.
                1 = Log all outputs (truncated to 20 tokens). Defaults to 0.

        Attributes:
            _dialogue_history (list): A record of the dialogue exchanges between the simulator and the chatbot.
            _openai_client (AsyncOpenAI): The OpenAI client instance for handling API requests.
            _user_simulator (str): The user simulator model identifier.
            _chatbot (str): The chatbot model identifier.
            _ooc_detector: The mechanism used for detecting out-of-character responses.
            _response_format (dict): The format expected for responses, defaulting to JSON objects.
            _user_profile (str): The rendered user profile based on the seed row data.
            _user_simulator_generation_parameters (dict): Parameters for customizing user simulator outputs.
            _chatbot_generation_parameters (dict): Parameters for customizing chatbot outputs.
            _conversation_history (list): A history of the full conversation sequence.
            _dialogue_runs_threshold (int): The threshold for dialogue runs before a specified action.

        Returns:
            None
        """
        self._dialogue_history = []
        if openai_client is None:
            self._openai_client = AsyncOpenAI(api_key=os.environ["api_key"])
        else:
            self._openai_client = openai_client
        self._user_simulator = user_simulator
        self._chatbot = chatbot
        self._ooc_detector = ooc_detector
        self._response_format = {"type": "json_object"}

        self._user_simulator_generation_parameters = (
            {}
            if user_simulator_generation_parameters is None
            else user_simulator_generation_parameters
        )
        self._chatbot_generation_parameters = (
            {}
            if chatbot_generation_parameters is None
            else chatbot_generation_parameters
        )
        self._conversation_history = []
        self._dialogue_runs_threshold = dialogue_runs_threshold
        self._verbose = verbose
        if ooc_detector_type == "llm":
            self.ooc_detection = self._llm_ooc_detection
        else:
            self.ooc_detection = None  # TODO

    # Getters
    @property
    def dialogue_history(self):
        return self._dialogue_history

    @property
    def openai_client(self):
        return self._openai_client

    @property
    def user_simulator(self):
        return self._user_simulator

    @property
    def chatbot(self):
        return self._chatbot

    @property
    def ooc_detector(self):
        return self._ooc_detector

    @property
    def response_format(self):
        return self._response_format

    @property
    def user_simulator_generation_parameters(self):
        return self._user_simulator_generation_parameters

    @property
    def chatbot_generation_parameters(self):
        return self._chatbot_generation_parameters

    @property
    def conversation_history(self):
        return self._conversation_history

    def add_to_conversation_history(self, dialogue_run):
        self._conversation_history.append(dialogue_run)

    @conversation_history.setter
    def conversation_history(self, new_list):
        self._conversation_history = new_list

    @property
    def dialogue_runs_threshold(self):
        return self._dialogue_runs_threshold

    async def _init_dialogue(self, user_profile):
        prompt_for_simulator = deepcopy(USER_SIMULATOR_INITIAL_PROMPT_MESSAGES)
        prompt_for_simulator[1]["content"] = prompt_for_simulator[1]["content"].format(
            user_details=user_profile
        )
        chat_completion_sample = await self.openai_client.chat.completions.create(
            model=self.user_simulator,
            messages=prompt_for_simulator,
            response_format=self.response_format,
            **self.user_simulator_generation_parameters
        )

        return json.loads(chat_completion_sample.choices[0].message.content)[
            "proposed_question"
        ]

    async def _followup_question(self, user_profile):
        formatted_conversation_history = []
        for run in self.conversation_history:
            formatted_conversation_history.append(
                run.convert_to_user_simulator_format()
            )

        prompt_for_simulator = deepcopy(USER_SIMULATOR_SUBSEQUENT_PROMPT_MESSAGES)
        prompt_for_simulator[1]["content"] = prompt_for_simulator[1]["content"].format(
            conversation_history=render_template(
                CONVERSATION_TEMPLATE_STRING,
                conversation_history=formatted_conversation_history,
            )
        )
        prompt_for_simulator[2]["content"] = prompt_for_simulator[2]["content"].format(
            user_details=user_profile
        )

        chat_completion_sample = await self.openai_client.chat.completions.create(
            model=self.user_simulator,
            messages=prompt_for_simulator,
            response_format=self.response_format,
            **self.user_simulator_generation_parameters
        )

        return json.loads(chat_completion_sample.choices[0].message.content)

    async def _query_chatbot(self, proposed_question):
        conversation_history_for_chatbot = [CHATBOT_SYSTEM_PROMPT]
        for run in self.conversation_history:
            conversation_history_for_chatbot += run.convert_to_openai_history()

        conversation_history_for_chatbot.append(
            {"role": "user", "content": proposed_question}
        )

        chatbot_answer = await self.openai_client.chat.completions.create(
            model=self.chatbot,
            messages=conversation_history_for_chatbot,
            **self.chatbot_generation_parameters
        )

        return chatbot_answer.choices[0].message.content

    async def _llm_ooc_detection(self, user_profile, proposed_question):
        ooc_detector_prompt = deepcopy(LLM_BASED_OOC_DETECTION_PROMPT)

        ooc_detector_prompt[1]["content"] = ooc_detector_prompt[1]["content"].format(
            user_details=user_profile, question=proposed_question
        )

        detection_result = await self.openai_client.chat.completions.create(
            model=self.ooc_detector,
            messages=ooc_detector_prompt,
            response_format={"type": "json_object"},
        )

        return json.loads(detection_result.choices[0].message.content)

    async def dialogue_generation(self, seed_row):
        """
        Asynchronously manages the dialogue generation process between the user simulator
        and the chatbot, maintaining a history of dialogue exchanges.

        This method initiates a conversation, processes follow-up interactions, and
        applies optional out-of-character (OOC) detection. The process continues until
        the dialogue history exceeds the predefined threshold or the conversation is terminated.

        Args:
            seed_row (Dict): The seed data used to retrieve and render the user's profile.

        Returns:
            list: The complete conversation history, represented as a list of `DialogueRun` objects.

        Workflow:
            1. Starts the dialogue with an initial question and retrieves the chatbot's response.
            2. Appends the interaction to the conversation history.
            3. Continues generating follow-up questions using the user simulator.
            4. Checks for conversation termination conditions:
            - If the user simulator signals to end the conversation.
            - If OOC detection is enabled and the proposed question is flagged as OOC.
            5. Fetches chatbot responses for valid follow-up questions and updates the conversation history.
            6. Stops when the conversation history length exceeds the threshold or the conversation ends.

        Raises:
            Exception: Propagates exceptions that occur during asynchronous operations like
                    `_init_dialogue` or `_query_chatbot`.

        Example:
            conversation = await self.dialogue_generation()
            for dialogue in conversation:
                print(f"User: {dialogue.user_content}, Chatbot: {dialogue.chatbot_content}")
        """
        if self._verbose == 1:
            logger.info("Starting dialogue generation process.")

        user_profile = render_template(
            PROFILE_TEMPLATE, profile_data=retrieve_user_profile(seed_row)
        )
        self.conversation_history = []
        try:
            # Initialize tqdm progress bar
            with tqdm(
                total=self.dialogue_runs_threshold,
                desc="Dialogue Generation",
                unit="turn",
            ) as pbar:
                # Start initial dialogue
                first_question = await self._init_dialogue(user_profile=user_profile)
                first_answer = await self._query_chatbot(first_question)

                if self._verbose == 1:
                    logger.info("User: %s", " ".join(first_question.split()[:20]))
                    logger.info("Chatbot: %s", " ".join(first_answer.split()[:20]))

                # Record initial dialogue exchange
                self.add_to_conversation_history(
                    DialogueRun(
                        user_content=first_question, chatbot_content=first_answer
                    )
                )
                pbar.update(1)

                # Continue generating follow-up dialogues
                while len(self.conversation_history) < self.dialogue_runs_threshold:
                    user_simulation = await self._followup_question(
                        user_profile=user_profile
                    )

                    if user_simulation["end_conversation"]:
                        if self._verbose == 1:
                            logger.info("Conversation ended by user simulation.")
                        return self.conversation_history

                    proposed_question = user_simulation["proposed_question"]

                    if self.ooc_detector is not None:
                        detection_result = await self.ooc_detection(
                            user_profile, proposed_question
                        )
                        if detection_result["has_out_of_context"]:
                            logger.warning(
                                "OOC detected: %s",
                                " ".join(proposed_question.split()),
                            )
                            logger.warning(
                                "New proposed question: %s",
                                " ".join(detection_result["updated_question"].split()),
                            )
                            proposed_question = detection_result["updated_question"]
                        # return self.conversation_history

                    chatbot_answer = await self._query_chatbot(proposed_question)

                    if self._verbose == 1:
                        logger.info(
                            "User: %s", " ".join(proposed_question.split()[:20])
                        )
                        logger.info(
                            "Chatbot: %s", " ".join(chatbot_answer.split()[:20])
                        )

                    # Update conversation history
                    self.add_to_conversation_history(
                        DialogueRun(
                            user_content=proposed_question,
                            chatbot_content=chatbot_answer,
                        )
                    )
                    pbar.update(1)

            if self._verbose == 1:
                logger.info("Dialogue generation process completed.")
            return self.conversation_history
        except Exception as e:
            logger.error("An error occurred: %s", str(e))
            raise
