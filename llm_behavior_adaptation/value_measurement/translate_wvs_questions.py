r"""
Translate WVS Questions
=======================
Pre-translates all WVS survey questions and the dialogue-followup instruction
into every target language found in the translated-dialogue files.

Output
------
A single JSON file (default: datasets/wvs_benchmarks/translated_questions.json)
with structure:

    {
      "Spanish": {
        "_instruction": "Asume que eres yo, ayúdame a encontrar...",
        "Q2":  "En una escala de 1 a 4, 1 significando ...",
        "Q3":  "...",
        ...
      },
      "Chinese": { ... },
      ...
    }

The special key ``_instruction`` holds the translated version of:
    "Assume you are me, help me find the most suitable answer to the
     following question:"

Usage
-----
python -m llm_behavior_adaptation.value_measurement.translate_wvs_questions \\
    --output datasets/wvs_benchmarks/translated_questions.json

Requires OPENAI_API_KEY (or pass --api-key).
Runs all languages/questions concurrently; use --concurrency to throttle.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

from llm_behavior_adaptation.utils import register_logger
from llm_behavior_adaptation.value_measurement.wvs_values_comparison import DATASET_DIR

logger = logging.getLogger(__name__)
register_logger(logger)

# The English instruction text that appears in dialogue_followup.json
ENGLISH_INSTRUCTION = "Assume you are me, help me find the most suitable answer to the following question:"

# All non-English target languages in our translated dialogue dataset
DEFAULT_LANGUAGES = [
    "Arabic",
    "Bengali",
    "Chinese",
    "Dutch",
    "German",
    "Greek",
    "Indonesian",
    "Japanese",
    "Korean",
    "Malay",
    "Persian",
    "Portuguese",
    "Romanian",
    "Russian",
    "Serbian",
    "Spanish",
    "Swahili",
    "Thai",
    "Turkish",
    "Ukrainian",
    "Urdu",
    "Vietnamese",
]


async def _translate_text(
    client: AsyncOpenAI,
    text: str,
    target_language: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Translate a single piece of text to target_language."""
    async with semaphore:
        prompt = (
            f"Translate the following text to {target_language}. "
            "Return only the translated text, nothing else.\n\n"
            f"{text}"
        )
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()


async def translate_all(
    questions: Dict[str, str],
    languages: List[str],
    api_key: str,
    model: str = "gpt-4.1-mini",
    concurrency: int = 20,
    existing: Dict = None,
) -> Dict[str, Dict[str, str]]:
    """
    Translate all questions + instruction into all languages.

    Parameters
    ----------
    questions : {qid: english_text}
    languages : list of target languages
    existing  : previously-saved dict to resume from (skips already-done entries)

    Returns
    -------
    {language: {qid: translated_text, "_instruction": translated_instruction}}
    """
    client = AsyncOpenAI(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    result: Dict[str, Dict[str, str]] = {lang: {} for lang in languages}

    # Merge existing to avoid re-translating
    if existing:
        for lang, translations in existing.items():
            if lang in result:
                result[lang].update(translations)

    # Build work list: (lang, key, text)
    work = []
    for lang in languages:
        done = result.get(lang, {})
        if "_instruction" not in done:
            work.append((lang, "_instruction", ENGLISH_INSTRUCTION))
        for qid, text in questions.items():
            if qid not in done:
                work.append((lang, qid, text))

    logger.info("%d translation calls needed (%d already cached)", len(work), sum(len(v) for v in result.values()))

    if not work:
        logger.info("All translations already cached.")
        return result

    async def do_one(lang: str, key: str, text: str) -> None:
        translated = await _translate_text(client, text, lang, model, sem)
        result[lang][key] = translated

    tasks = [do_one(lang, key, text) for lang, key, text in work]
    for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Translating"):
        await coro

    return result


def _load_all_questions(picked_questions_path: str) -> Dict[str, str]:
    """Flatten picked_questions.json into {qid: question_text}."""
    with open(picked_questions_path, "r", encoding="utf-8") as f:
        picked = json.load(f)
    questions: Dict[str, str] = {}
    for cat_dict in picked.values():
        for qid, qdata in cat_dict.items():
            questions[qid] = qdata["question"]
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-translate WVS questions into all target languages.")
    parser.add_argument(
        "--output",
        type=str,
        default=f"{DATASET_DIR}/translated_questions.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--picked-questions",
        type=str,
        default=f"{DATASET_DIR}/picked_questions.json",
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help="Languages to translate into (default: all 22 in dataset)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Max parallel API requests (default: 20)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("api_key")
    if not api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    questions = _load_all_questions(args.picked_questions)
    logger.info("Loaded %d questions", len(questions))

    # Load existing output to support resuming
    out_path = Path(args.output)
    existing = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        logger.info("Loaded existing translations for %d languages", len(existing))

    result = asyncio.run(
        translate_all(
            questions=questions,
            languages=args.languages,
            api_key=api_key,
            model=args.model,
            concurrency=args.concurrency,
            existing=existing,
        )
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in result.values())
    logger.info("Saved %d translations to %s", total, out_path)
    print(f"\nDone. {total} translations written to {out_path}")
    print(f"Languages covered: {sorted(result.keys())}")


if __name__ == "__main__":
    main()
