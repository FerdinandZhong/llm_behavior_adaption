"""
Dialogue controller
"""

import json
import logging
import os
from copy import deepcopy
from typing import Dict, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from ..utils import register_logger
from .generation_utils import load_json_folder, render_json, retrieve_user_profile_wvs

logger = logging.getLogger(__name__)
register_logger(logger)


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


class ChatMessage(BaseModel):
    speaker: Literal["user", "chatbot"] = Field(..., description="Message author")
    text: str = Field(..., min_length=1, description="Message content")

    @staticmethod
    def convert_to_openai_history(run) -> list[dict[str, str]]:
        """Return this message as a single OpenAI chat message."""
        role = "user" if run["speaker"] == "user" else "assistant"
        return [{"role": role, "content": run["text"]}]


class DialogueGenerator:
    def __init__(
        self,
        prompts_folder: str,
        user_simulator="gpt-5",
        chatbot="gpt-4.1-mini",
        ooc_detector="o4-mini",
        dialogue_reviewer="gpt-5-mini",
        openai_client=None,
        user_simulator_generation_paramters: Dict = {
            "reasoning_effort": "low",
            "verbosity": "low",
        },
        chatbot_generation_parameters=None,
        ooc_detector_parameters: Dict = {
            "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
        },
        dialogue_reviewer_parameters: Dict = {
            "reasoning_effort": "low",
            "verbosity": "low",
            "response_format": {"type": "json_object"},
        },
        dialogue_runs_threshold: int = 10,
        verbose: int = 0,
    ) -> None:
        # --- init properties / fields ---
        self._dialogue_history: list = []

        # OpenAI client
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("api_key")
        if openai_client is None:
            if not api_key:
                raise RuntimeError("Missing OPENAI_API_KEY (or 'api_key') in environment.")
            self._openai_client = AsyncOpenAI(api_key=api_key)
        else:
            self._openai_client = openai_client

        # Model ids
        self._user_simulator: str = user_simulator
        self._chatbot: str = chatbot
        self._ooc_detector: str = ooc_detector
        self._dialogue_reviewer: str = dialogue_reviewer

        # Response format (exposed via .response_format)
        self._response_format: dict = {"type": "json_object"}

        # Safe copy helpers for dict defaults
        def _safe_params(passed: dict = None, fallback: dict = None) -> dict:
            if passed is not None:
                return dict(passed)
            return dict(fallback) if fallback is not None else {}

        self._user_simulator_generation_parameters: dict = _safe_params(
            user_simulator_generation_paramters,
            {"reasoning_effort": "low", "verbosity": "low"},
        )
        self._chatbot_generation_parameters: dict = _safe_params(chatbot_generation_parameters, {})
        self._ooc_detector_parameters: dict = _safe_params(
            ooc_detector_parameters,
            {
                "reasoning_effort": "low",
                "verbosity": "low",
                "response_format": {"type": "json_object"},
            },
        )
        self._dialogue_reviewer_parameters: dict = _safe_params(
            dialogue_reviewer_parameters,
            {
                "reasoning_effort": "low",
                "verbosity": "low",
                "response_format": {"type": "json_object"},
            },
        )

        self._dialogue_runs_threshold: int = int(dialogue_runs_threshold)
        self._verbose: int = int(verbose)

        # Prompts
        self._prompts_folder: str = str(prompts_folder)
        try:
            self._prompts: dict = load_json_folder(folder=self._prompts_folder)
        except Exception as e:
            raise RuntimeError(f"Failed to load prompts from '{self._prompts_folder}': {e}") from e
        if not isinstance(self._prompts, dict):
            raise TypeError(
                f"load_json_folder('{self._prompts_folder}') must return a dict; got {type(self._prompts)!r}"
            )

    # --- properties / helpers ---
    @property
    def dialogue_history(self) -> list:
        return self._dialogue_history

    def reset_dialogue_history(self) -> None:
        self._dialogue_history = []

    @property
    def openai_client(self):
        return self._openai_client

    @property
    def user_simulator(self) -> str:
        return self._user_simulator

    @property
    def chatbot(self) -> str:
        return self._chatbot

    @property
    def ooc_detector(self) -> str:
        return self._ooc_detector

    @property
    def dialogue_reviewer(self) -> str:
        return self._dialogue_reviewer

    @property
    def response_format(self) -> dict:
        return self._response_format

    # Back-compat alias (original typo)
    @property
    def user_simulator_generation_paramters(self) -> dict:
        return self._user_simulator_generation_parameters

    # Correct spelling alias
    @property
    def user_simulator_generation_parameters(self) -> dict:
        return self._user_simulator_generation_parameters

    @property
    def chatbot_generation_parameters(self) -> dict:
        return self._chatbot_generation_parameters

    @property
    def ooc_detector_parameters(self) -> dict:
        return self._ooc_detector_parameters

    @property
    def dialogue_reviewer_parameters(self) -> dict:
        return self._dialogue_reviewer_parameters

    @property
    def dialogue_runs_threshold(self) -> int:
        return self._dialogue_runs_threshold

    @property
    def verbose(self) -> int:
        return self._verbose

    @property
    def prompts_folder(self) -> str:
        return self._prompts_folder

    @property
    def prompts(self) -> dict:
        return self._prompts

    def _prompt(self, key: str) -> list[dict]:
        try:
            return deepcopy(self.prompts[key])
        except KeyError as e:
            raise KeyError(f"Missing prompt '{key}' in {self._prompts_folder}") from e

    def _history_as_openai(self) -> list[dict[str, str]]:
        msgs = []
        for turn in self.dialogue_history:
            role = "user" if turn["speaker"] == "user" else "assistant"
            msgs.append({"role": role, "content": turn["text"]})
        return msgs

    # -------- LLM calls --------
    async def _init_dialogue(self, user_profile: str) -> str:
        init_prompt = self._prompt("user_simulator_initial_prompt")
        init_prompt[1]["content"] = init_prompt[1]["content"].format(user_details=user_profile)
        resp = await self.openai_client.chat.completions.create(
            model=self.user_simulator,
            messages=init_prompt,
            **self.user_simulator_generation_paramters,
        )
        return resp.choices[0].message.content

    async def _llm_ooc_detection(self, user_profile: str, proposed_question: str) -> dict:
        """
        Returns JSON like: {"has_out_of_context": bool, "reason": "<string or empty>"}
        (No rewriting here.)
        """
        prompt = self._prompt("ooc_detector_prompt")
        prompt[1]["content"] = prompt[1]["content"].format(
            user_details=user_profile, generated_question=proposed_question
        )
        det = await self.openai_client.chat.completions.create(
            model=self.ooc_detector,
            messages=prompt,
            **self.ooc_detector_parameters,  # includes response_format={"type":"json_object"}
        )
        # JSON mode constrains output to valid JSON parseable string. :contentReference[oaicite:1]{index=1}
        return json.loads(det.choices[0].message.content)

    async def _rewrite_question(self, user_profile: str, last_user_question: str, ooc_reason: str) -> str:
        """
        Rewrites the last user question using the OOC reason + full dialogue + user profile.
        """
        history_str = render_json(self.dialogue_history)
        prompt = self._prompt("user_simulator_rewriter_prompt")
        # Typical layout: [system,...] indices may vary based on your prompt file
        prompt[1]["content"] = prompt[1]["content"].format(conversation_history=history_str)
        prompt[2]["content"] = prompt[2]["content"].format(user_details=user_profile)
        prompt[3]["content"] = prompt[3]["content"].format(user_last_message=last_user_question)
        prompt[4]["content"] = prompt[4]["content"].format(expert_review=ooc_reason)

        resp = await self.openai_client.chat.completions.create(
            model=self.user_simulator,  # reuse the simulator to rewrite
            messages=prompt,
            **self.user_simulator_generation_paramters,
        )
        return resp.choices[0].message.content

    async def _review_dialogue(self) -> dict:
        history_str = render_json(self.dialogue_history)
        prompt = self._prompt("dialogue_reviewer_prompt")
        prompt[1]["content"] = prompt[1]["content"].format(conversation_history=history_str)
        resp = await self.openai_client.chat.completions.create(
            model=self.dialogue_reviewer,
            messages=prompt,
            **self.dialogue_reviewer_parameters,  # includes JSON mode
        )
        return json.loads(resp.choices[0].message.content)

    async def _followup_question(self, user_profile: str) -> str:
        history_str = render_json(self.dialogue_history)
        prompt = self._prompt("user_simulator_subsequent_prompt")
        prompt[1]["content"] = prompt[1]["content"].format(conversation_history=history_str)
        prompt[2]["content"] = prompt[2]["content"].format(user_details=user_profile)
        resp = await self.openai_client.chat.completions.create(
            model=self.user_simulator,
            messages=prompt,
            **self.user_simulator_generation_paramters,
        )
        return resp.choices[0].message.content

    async def _query_chatbot(self, proposed_question: str) -> str:
        msgs = self._history_as_openai()
        msgs.append({"role": "user", "content": proposed_question})
        resp = await self.openai_client.chat.completions.create(
            model=self.chatbot,
            messages=msgs,
            **(self.chatbot_generation_parameters or {}),
        )
        return resp.choices[0].message.content

    def append_to_dialogue(self, proposed_question: str, chatbot_answer: str) -> None:
        user_turn = ChatMessage(speaker="user", text=proposed_question)
        bot_turn = ChatMessage(speaker="chatbot", text=chatbot_answer)
        self.dialogue_history.append(user_turn.model_dump())
        self.dialogue_history.append(bot_turn.model_dump())

    # -------- Orchestration --------
    async def dialogue_generation(self, seed_row: Dict) -> list:
        """
        Adds a separate OOC detector (reason only) + user_simulator_rewriter step.
        """
        if self._verbose == 1:
            logger.info("Starting dialogue generation process.")

        user_profile = render_json(json_input=retrieve_user_profile_wvs(seed_row))
        self.reset_dialogue_history()

        try:
            with tqdm(
                total=self.dialogue_runs_threshold,
                desc="Dialogue Generation",
                unit="turn",
            ) as pbar:
                # ---- Initial turn ----
                first_question = await self._init_dialogue(user_profile=user_profile)

                det = await self._llm_ooc_detection(user_profile=user_profile, proposed_question=first_question)
                if det.get("has_out_of_context"):
                    reason = det.get("reason", "")
                    if self._verbose == 1:
                        logger.warning("OOC detected on initial question. Reason: %s", reason[:200])
                    # rewrite (instead of detector rewriting directly)
                    first_question = await self._rewrite_question(
                        user_profile=user_profile,
                        last_user_question=first_question,
                        ooc_reason=reason,
                    )

                first_answer = await self._query_chatbot(first_question)

                if self._verbose == 1:
                    logger.info("User: %s", " ".join(first_question.split()[:20]))
                    logger.info("Chatbot: %s", " ".join(first_answer.split()[:20]))

                self.append_to_dialogue(first_question, first_answer)
                pbar.update(1)

                # ---- Loop ----
                while len(self.dialogue_history) < self.dialogue_runs_threshold:
                    review = await self._review_dialogue()
                    if review.get("end_conversation"):
                        if self._verbose == 1:
                            logger.info("Conversation ended by reviewer.")
                        return self.dialogue_history

                    proposed_question = await self._followup_question(user_profile=user_profile)

                    det = await self._llm_ooc_detection(user_profile=user_profile, proposed_question=proposed_question)
                    if det.get("has_out_of_context"):
                        reason = det.get("reason", "")
                        if self._verbose == 1:
                            logger.warning("OOC detected on follow-up. Reason: %s", reason[:200])
                        proposed_question = await self._rewrite_question(
                            user_profile=user_profile,
                            last_user_question=proposed_question,
                            ooc_reason=reason,
                        )

                    chatbot_answer = await self._query_chatbot(proposed_question)

                    if self._verbose == 1:
                        logger.info("User: %s", " ".join(proposed_question.split()[:20]))
                        logger.info("Chatbot: %s", " ".join(chatbot_answer.split()[:20]))

                    self.append_to_dialogue(proposed_question, chatbot_answer)
                    pbar.update(1)

            if self._verbose == 1:
                logger.info("Dialogue generation process completed.")
            return self.dialogue_history

        except Exception as e:
            logger.error("An error occurred: %s", str(e))
            raise
