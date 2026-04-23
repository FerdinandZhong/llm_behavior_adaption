r"""
Retranslate Missed Dialogues
============================
Fixes the 173 dialogues that were incorrectly kept in English due to
missing/mismatched country names in COUNTRY_LANGUAGE_MAP.

For each entry in the existing translated JSONL whose
``language_selection_reason`` contains "no mapped language" and whose
``place_of_residence`` is now covered by the fixed map, this script:

  1. Re-determines the correct target language using the updated map.
  2. Re-translates every turn using the ``original_*_content`` fields
     already stored inside the translated JSONL (no need to read the
     source dialogue files again).
  3. Writes a fully patched translated JSONL (same path by default).

After this script finishes, re-run ``prepare_translated_dialogues.py``
on each patched file to refresh the ``*_translated_formatted.jsonl``
files that feed the model.

Usage
-----
# Career
python -m llm_behavior_adaptation.dialogue_dataset_creation.retranslate_missed_dialogues \\
    --input  wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl \\
    --output wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl

# Investment
python -m llm_behavior_adaptation.dialogue_dataset_creation.retranslate_missed_dialogues \\
    --input  wvs_generated_dialogues/translated_dialogues/investment/investment_translated.jsonl \\
    --output wvs_generated_dialogues/translated_dialogues/investment/investment_translated.jsonl

Set OPENAI_API_KEY (or pass --api-key) before running.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

from llm_behavior_adaptation.dialogue_dataset_creation.dialogue_translator import (
    COMMON_LANGUAGES,
    COUNTRY_LANGUAGE_MAP,
    DialogueTranslator,
    TranslatedTurn,
)
from llm_behavior_adaptation.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _needs_retranslation(entry: Dict[str, Any]) -> bool:
    """Return True if this entry defaulted to English due to missing map entry."""
    return entry.get("target_language") == "English" and "no mapped language" in entry.get(
        "language_selection_reason", ""
    )


def _correct_language(entry: Dict[str, Any]) -> Optional[str]:
    """
    Given a translated entry that defaulted to English, determine the correct
    target language using the *fixed* COUNTRY_LANGUAGE_MAP.

    Returns the new target language, or None if the place is still unmapped
    (meaning the English default was intentional).
    """
    place = (entry.get("user_profile") or {}).get("place_of_residence", "")
    if not place:
        return None
    lang = COUNTRY_LANGUAGE_MAP.get(place)
    if lang and lang != "English" and lang in COMMON_LANGUAGES:
        return lang
    return None


async def _retranslate_entry(
    entry: Dict[str, Any],
    target_language: str,
    translator: DialogueTranslator,
) -> Dict[str, Any]:
    """
    Retranslate all turns in *entry* to *target_language* using the original
    English content stored in the existing translated JSONL.

    Returns a new entry dict (same structure as the input).
    """
    new_turns: List[Dict[str, Any]] = []
    for turn in entry.get("translated_dialogue", []):
        orig_user = turn.get("original_user_content", turn.get("user_content", ""))
        orig_chat = turn.get("original_chatbot_content", turn.get("chatbot_content", ""))

        translated_turn: TranslatedTurn = await translator.translate_turn(
            user_content=orig_user,
            chatbot_content=orig_chat,
            target_language=target_language,
        )
        new_turns.append(translated_turn.model_dump())

    new_entry = dict(entry)
    new_entry["target_language"] = target_language
    new_entry["language_selection_reason"] = (
        f"Selected {target_language} based on current residence: "
        f"{(entry.get('user_profile') or {}).get('place_of_residence', '')} "
        f"[retranslated — map was previously missing this place name]"
    )
    new_entry["translated_dialogue"] = new_turns
    return new_entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def retranslate(
    input_path: str,
    output_path: str,
    api_key: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
    dry_run: bool = False,
    concurrency: int = 5,
) -> None:
    # Load all entries
    with open(input_path, "r", encoding="utf-8") as f:
        entries: List[Dict[str, Any]] = [json.loads(line) for line in f if line.strip()]

    # Identify which need retranslation
    to_fix: List[int] = []
    lang_map: Dict[int, str] = {}
    for i, entry in enumerate(entries):
        if _needs_retranslation(entry):
            lang = _correct_language(entry)
            if lang:
                to_fix.append(i)
                lang_map[i] = lang

    logger.info(
        "Found %d entries needing retranslation out of %d total.",
        len(to_fix),
        len(entries),
    )

    if dry_run:
        from collections import Counter

        lang_counter = Counter(lang_map.values())
        print(f"\n[DRY RUN] Would retranslate {len(to_fix)} entries:")
        for lang, count in lang_counter.most_common():
            print(f"  {lang}: {count}")
        return

    if not to_fix:
        logger.info("Nothing to retranslate. Exiting.")
        return

    client = AsyncOpenAI(api_key=api_key)
    translator = DialogueTranslator(client=client, model=model, temperature=temperature, verbose=0)

    # Process in batches to respect concurrency limit
    sem = asyncio.Semaphore(concurrency)

    async def process_one(idx: int) -> None:
        async with sem:
            new_entry = await _retranslate_entry(entries[idx], lang_map[idx], translator)
            entries[idx] = new_entry
            logger.info(
                "Retranslated index %d → %s",
                idx,
                lang_map[idx],
            )

    tasks = [process_one(i) for i in to_fix]
    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Retranslating"):
        await coro

    # Back up original before overwriting
    in_path = Path(input_path)
    out_path = Path(output_path)
    if in_path.resolve() == out_path.resolve():
        backup = in_path.with_suffix(".jsonl.bak")
        shutil.copy2(in_path, backup)
        logger.info("Backed up original to %s", backup)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Wrote patched file to %s  (%d total entries)", out_path, len(entries))
    print(f"\nDone. Retranslated {len(to_fix)} entries → {out_path}")
    print(
        "Next step: re-run prepare_translated_dialogues.py on the patched file "
        "to refresh the *_formatted.jsonl used by the model pipeline."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retranslate dialogues that were incorrectly kept in English.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Existing translated JSONL (e.g. career_translated.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for patched JSONL (can be same as --input; original is backed up)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI / OpenRouter API key (defaults to OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Translation model (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max parallel translation requests (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be retranslated without calling the API",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("api_key")
    if not api_key and not args.dry_run:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY env var or pass --api-key.")

    asyncio.run(
        retranslate(
            input_path=args.input,
            output_path=args.output,
            api_key=api_key or "",
            model=args.model,
            temperature=args.temperature,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
        )
    )


if __name__ == "__main__":
    main()
