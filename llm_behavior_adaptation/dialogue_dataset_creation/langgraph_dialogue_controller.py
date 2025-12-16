from __future__ import annotations

import json
import logging
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from openai import AsyncOpenAI
from pydantic import BaseModel

from llm_behavior_adaptation.dialogue_dataset_creation.generation_utils import (
    load_json_folder,
    render_json,
    retrieve_user_profile_wvs,
)

# ---------- Logger ----------
logger = logging.getLogger(__name__)


# ---------- Only keep the 20-word summary for response logs ----------
def _twenty_word_summary(text: str) -> str:
    if not text:
        return ""
    words = text.replace("\n", " ").split()
    return " ".join(words[:20]) + ("..." if len(words) > 20 else "")


# ---------- State models ----------
class Turn(BaseModel):
    speaker: Literal["user", "chatbot"]
    text: str


class DGState(TypedDict, total=False):
    history: List[Turn]
    turns: int
    end: bool

    user_profile: str
    proposed_question: str
    chatbot_answer: str
    ooc_reason: Optional[str]
    review: Dict[str, Any]


class DialogueAgent:
    """
    Reusable LangGraph agent:
      - Reuses a single AsyncOpenAI client
      - Compiles the graph once
      - Start new runs with different user_profile / seed_row
      - Logs ONLY a 20-word summary for each model response
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        prompts_folder: str,
        models: Dict[str, str] | None = None,
        params: Dict[str, Dict[str, Any]] | None = None,
        threshold: int = 10,
        checkpointer=None,
        verbose: int = 0,
    ):
        self.client = client
        self.prompts_folder = prompts_folder
        self.prompts = load_json_folder(prompts_folder)
        self.threshold = threshold
        self.checkpointer = checkpointer or MemorySaver()
        self.verbose = verbose

        self.models = models or {
            "user_sim": "gpt-5",
            "chatbot": "gpt-4.1-mini",
            "ooc": "o4-mini",
            "reviewer": "gpt-5-mini",
        }
        self.params = params or {
            "user_sim": {"reasoning_effort": "low", "verbosity": "low"},
            "chatbot": {},
            "ooc": {
                "reasoning_effort": "low",
                "response_format": {"type": "json_object"},
            },
            "reviewer": {
                "reasoning_effort": "low",
                "verbosity": "low",
                "response_format": {"type": "json_object"},
            },
        }

        self.graph = self._build_graph()

    # ---------- Prompt & message utils ----------
    def _prompt(self, key: str) -> list[dict]:
        if key not in self.prompts:
            raise KeyError(f"Missing prompt '{key}'")
        return deepcopy(self.prompts[key])

    @staticmethod
    def _jsonable_history(history: List[Turn] | List[Dict[str, Any]]):
        return [h.model_dump() if hasattr(h, "model_dump") else h for h in history]

    @staticmethod
    def _history_as_openai(history: List[Turn]) -> list[dict]:
        msgs = []
        for t in history:
            role = "user" if t.speaker == "user" else "assistant"
            msgs.append({"role": role, "content": t.text})
        return msgs

    # ---------- Nodes ----------
    async def init_node(self, state: DGState) -> DGState:
        p = self._prompt("user_simulator_initial_prompt")
        p[1]["content"] = p[1]["content"].format(user_details=state["user_profile"])

        r = await self.client.chat.completions.create(
            model=self.models["user_sim"],
            messages=p,
            **self.params["user_sim"],
        )
        first_q = r.choices[0].message.content

        if self.verbose >= 1:
            # Response: ONLY 20-word summary
            logger.info(
                "[OPENAI][RESP] %s | summary=%s",
                "user_sim:init",
                _twenty_word_summary(first_q),
            )

        history = state.get("history", [])
        history.append(Turn(speaker="user", text=first_q))
        state["history"] = history
        state["turns"] = state.get("turns", 0) + 1
        state["proposed_question"] = first_q
        return state

    async def followup_node(self, state: DGState) -> DGState:
        history_str = render_json(self._jsonable_history(state["history"]))
        p = self._prompt("user_simulator_subsequent_prompt")
        p[1]["content"] = p[1]["content"].format(conversation_history=history_str)
        p[2]["content"] = p[2]["content"].format(user_details=state["user_profile"])

        r = await self.client.chat.completions.create(
            model=self.models["user_sim"],
            messages=p,
            **self.params["user_sim"],
        )
        q = r.choices[0].message.content

        if self.verbose >= 1:
            logger.info(
                "[OPENAI][RESP] %s | summary=%s",
                "user_sim:followup",
                _twenty_word_summary(q),
            )

        state["history"].append(Turn(speaker="user", text=q))
        state["turns"] = state.get("turns", 0) + 1
        state["proposed_question"] = q
        return state

    async def ooc_detect_node(self, state: DGState) -> DGState:
        p = self._prompt("ooc_detector_prompt")
        p[1]["content"] = p[1]["content"].format(
            user_details=state["user_profile"],
            generated_question=state["proposed_question"],
        )

        r = await self.client.chat.completions.create(
            model=self.models["ooc"],
            messages=p,
            **self.params["ooc"],
        )
        raw = r.choices[0].message.content

        if self.verbose >= 1:
            logger.info(
                "[OPENAI][RESP] %s | summary=%s",
                "ooc_detector",
                _twenty_word_summary(raw),
            )

        data = json.loads(raw)
        state["ooc_reason"] = data.get("reason") if data.get("has_out_of_context") else None
        return state

    async def rewriter_node(self, state: DGState) -> DGState:
        if not state.get("ooc_reason"):
            return state

        hist_str = render_json(self._jsonable_history(state["history"]))
        p = self._prompt("user_simulator_rewriter_prompt")
        p[1]["content"] = p[1]["content"].format(conversation_history=hist_str)
        p[2]["content"] = p[2]["content"].format(user_details=state["user_profile"])
        p[3]["content"] = p[3]["content"].format(user_last_message=state["proposed_question"])
        p[4]["content"] = p[4]["content"].format(expert_review=state["ooc_reason"])

        r = await self.client.chat.completions.create(
            model=self.models["user_sim"],
            messages=p,
            **self.params["user_sim"],
        )
        new_q = r.choices[0].message.content

        if self.verbose >= 1:
            logger.info(
                "[OPENAI][RESP] %s | summary=%s",
                "user_sim:rewriter",
                _twenty_word_summary(new_q),
            )

        state["history"][-1] = Turn(speaker="user", text=new_q)
        state["proposed_question"] = new_q
        state["ooc_reason"] = None
        return state

    async def chatbot_node(self, state: DGState) -> DGState:
        msgs = self._history_as_openai(state["history"])
        if msgs[-1]["role"] != "user" or msgs[-1]["content"] != state["proposed_question"]:
            msgs.append({"role": "system", "content": "Answer the question concisely"})
            msgs.append({"role": "user", "content": state["proposed_question"]})

        r = await self.client.chat.completions.create(
            model=self.models["chatbot"],
            messages=msgs,
            **(self.params.get("chatbot") or {}),
        )
        ans = r.choices[0].message.content

        if self.verbose >= 1:
            logger.info("[OPENAI][RESP] %s | summary=%s", "chatbot", _twenty_word_summary(ans))

        state["chatbot_answer"] = ans
        state["history"].append(Turn(speaker="chatbot", text=ans))
        return state

    async def reviewer_node(self, state: DGState) -> DGState:
        hist_str = render_json(self._jsonable_history(state["history"]))
        p = self._prompt("dialogue_reviewer_prompt")
        p[1]["content"] = p[1]["content"].format(conversation_history=hist_str)

        r = await self.client.chat.completions.create(
            model=self.models["reviewer"],
            messages=p,
            **self.params["reviewer"],
        )
        raw = r.choices[0].message.content

        if self.verbose >= 1:
            logger.info("[OPENAI][RESP] %s | summary=%s", "reviewer", _twenty_word_summary(raw))

        data = json.loads(raw)
        state["review"] = data
        state["end"] = bool(data.get("end_conversation")) or (state.get("turns", 0) >= self.threshold)
        return state

    # ---------- Routers ----------
    @staticmethod
    def need_rewrite_router(state: DGState) -> Literal["maybe_rewrite", "skip_rewrite"]:
        return "maybe_rewrite" if state.get("ooc_reason") else "skip_rewrite"

    @staticmethod
    def continue_or_end_router(state: DGState) -> Literal["continue", "__end__"]:
        return "continue" if not state.get("end") else END

    # ---------- Graph build ----------
    def _build_graph(self) -> CompiledStateGraph:
        g = StateGraph(DGState)
        g.add_node("init", self.init_node)
        g.add_node("followup", self.followup_node)
        g.add_node("ooc_detect", self.ooc_detect_node)
        g.add_node("rewriter", self.rewriter_node)
        g.add_node("chatbot", self.chatbot_node)
        g.add_node("reviewer", self.reviewer_node)

        g.add_edge("init", "ooc_detect")
        g.add_conditional_edges(
            "ooc_detect",
            self.need_rewrite_router,
            {"maybe_rewrite": "rewriter", "skip_rewrite": "chatbot"},
        )
        g.add_edge("rewriter", "chatbot")
        g.add_edge("chatbot", "reviewer")
        g.add_conditional_edges(
            "reviewer",
            self.continue_or_end_router,
            {"continue": "followup", END: END},
        )
        g.add_edge("followup", "ooc_detect")
        g.set_entry_point("init")

        return g.compile(checkpointer=self.checkpointer)

    # ---------- State factory ----------
    def make_start_state_from_profile(self, user_profile_text: str) -> DGState:
        return {
            "history": [],
            "turns": 0,
            "end": False,
            "user_profile": user_profile_text,
        }

    def make_start_state_from_seed(self, seed_row: Dict[str, Any]) -> DGState:
        user_profile = render_json(retrieve_user_profile_wvs(seed_row))
        return self.make_start_state_from_profile(user_profile)

    # ---------- Run ----------
    async def run(self, start_state: DGState) -> DGState:
        """Run a full dialogue and return the final state."""
        thread = {
            "recursion_limit": 5 * self.threshold,
            "configurable": {"thread_id": str(uuid.uuid4())},
        }
        final_state = await self.graph.ainvoke(start_state, thread)
        return final_state
