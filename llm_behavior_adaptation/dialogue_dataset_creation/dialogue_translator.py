"""
Translation script for generated dialogues.

This script translates generated dialogues to the appropriate language based on user profiles,
prioritizing languages from current living country, then born country, and falling back to English
for uncommon languages.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

# ---------- Logger ----------
logger = logging.getLogger(__name__)


# ---------- Language Standards ----------
# Languages considered "common" for LLM translation - those with good LLM support
# Based on: widespread use, digital presence, and LLM training data availability
COMMON_LANGUAGES = {
    # Major international languages
    "English",
    "Spanish",
    "French",
    "German",
    "Italian",
    "Portuguese",
    "Russian",
    "Arabic",
    "Japanese",
    "Korean",
    "Chinese",
    # Regional languages with strong LLM support
    "Hindi",
    "Bengali",
    "Urdu",
    "Turkish",
    "Vietnamese",
    "Thai",
    "Indonesian",
    "Malay",
    "Dutch",
    "Polish",
    "Ukrainian",
    "Romanian",
    "Czech",
    "Greek",
    "Hebrew",
    "Persian",
    "Swedish",
    "Norwegian",
    "Danish",
    "Finnish",
    "Hungarian",
    "Serbian",
    "Croatian",
    "Bosnian",
    "Swahili",
}

# Country to primary language mapping
COUNTRY_LANGUAGE_MAP = {
    # Major countries
    "United States": "English",
    "United Kingdom": "English",
    "Canada": "English",
    "Australia": "English",
    "New Zealand": "English",
    "Ireland": "English",
    "South Africa": "English",
    "Spain": "Spanish",
    "Mexico": "Spanish",
    "Argentina": "Spanish",
    "Colombia": "Spanish",
    "Chile": "Spanish",
    "Peru": "Spanish",
    "Venezuela": "Spanish",
    "France": "French",
    "Belgium": "French",
    "Switzerland": "French",
    "Luxembourg": "French",
    "Germany": "German",
    "Austria": "German",
    "Italy": "Italian",
    "Brazil": "Portuguese",
    "Portugal": "Portuguese",
    "Russia": "Russian",
    "Belarus": "Russian",
    "Kazakhstan": "Russian",
    "China": "Chinese",
    "Taiwan": "Chinese",
    "Singapore": "Chinese",
    "Hong Kong": "Chinese",
    "Japan": "Japanese",
    "South Korea": "Korean",
    "North Korea": "Korean",
    "India": "Hindi",
    "Pakistan": "Urdu",
    "Bangladesh": "Bengali",
    "Turkey": "Turkish",
    "Vietnam": "Vietnamese",
    "Thailand": "Thai",
    "Indonesia": "Indonesian",
    "Malaysia": "Malay",
    "Netherlands": "Dutch",
    "Poland": "Polish",
    "Ukraine": "Ukrainian",
    "Romania": "Romanian",
    "Czech Republic": "Czech",
    "Greece": "Greek",
    "Israel": "Hebrew",
    "Iran": "Persian",
    "Sweden": "Swedish",
    "Norway": "Norwegian",
    "Denmark": "Danish",
    "Finland": "Finnish",
    "Hungary": "Hungarian",
    # Middle East
    "Saudi Arabia": "Arabic",
    "Egypt": "Arabic",
    "United Arab Emirates": "Arabic",
    "Jordan": "Arabic",
    "Lebanon": "Arabic",
    "Iraq": "Arabic",
    "Syria": "Arabic",
    "Morocco": "Arabic",
    "Algeria": "Arabic",
    "Tunisia": "Arabic",
    # Additional countries
    "Serbia": "Serbian",
    "Croatia": "Croatian",
    "Bosnia and Herzegovina": "Bosnian",
    "Kenya": "Swahili",
    "Tanzania": "Swahili",
    "Uganda": "English",
    "Philippines": "English",
    # WVS name variants — these were previously missing due to name mismatches
    "Hong Kong SAR": "Chinese",
    "Macao SAR": "Chinese",
    "Taiwan ROC": "Chinese",
    "Czechia": "Czech",
    "Bolivia": "Spanish",
    "Nicaragua": "Spanish",
    "Ecuador": "Spanish",
    "Guatemala": "Spanish",
    "Andorra": "Spanish",
    "Puerto Rico": "Spanish",
    "Libya": "Arabic",
    "Cyprus": "Greek",
}


# ---------- Models ----------
class TranslatedTurn(BaseModel):
    """A single translated turn in the dialogue."""

    user_content: str
    chatbot_content: str
    original_user_content: str
    original_chatbot_content: str


class TranslatedDialogue(BaseModel):
    """Complete translated dialogue with metadata."""

    index: int
    target_language: str
    language_selection_reason: str
    translated_dialogue: List[TranslatedTurn]
    user_profile: Optional[Dict[str, Any]] = None


# ---------- Language Selection Logic ----------
def get_country_language(country: str) -> Optional[str]:
    """Get the primary language for a given country."""
    return COUNTRY_LANGUAGE_MAP.get(country)


def is_common_language(language: str) -> bool:
    """Check if a language is considered common for LLM translation."""
    if not language:
        return False
    return language in COMMON_LANGUAGES


def select_target_language(user_profile: Dict[str, Any]) -> tuple[str, str]:
    """
    Select the target language for translation based on user profile.

    Priority:
    1. Language of current living country (if common)
    2. Language of born country (if current country's language is uncommon)
    3. English (if both are uncommon)

    Returns:
        tuple: (target_language, reason_for_selection)
    """
    current_country = user_profile.get("place_of_residence", "")
    born_country = user_profile.get("place_of_birth", user_profile.get("country_of_birth", ""))

    # Try current living country
    uncommon_lang = None
    if current_country:
        current_language = get_country_language(current_country)
        if current_language and is_common_language(current_language):
            reason = f"Selected {current_language} based on " f"current residence: {current_country}"
            return current_language, reason
        if current_language:
            # Language exists but uncommon, note it
            uncommon_lang = current_language

    # Try born country if current country's language is uncommon or not found
    if born_country:
        born_language = get_country_language(born_country)
        if born_language and is_common_language(born_language):
            if uncommon_lang:
                reason = (
                    f"Selected {born_language} from birth country "
                    f"({born_country}) because current residence language "
                    f"({uncommon_lang}) is uncommon"
                )
            else:
                reason = (
                    f"Selected {born_language} from birth country "
                    f"({born_country}) because current residence "
                    f"({current_country}) has no mapped language"
                )
            return born_language, reason

    # Default to English
    reason_parts = []
    if current_country:
        current_lang = get_country_language(current_country)
        if current_lang:
            reason_parts.append(f"current residence language ({current_lang} from " f"{current_country}) is uncommon")
        else:
            reason_parts.append(f"current residence ({current_country}) has no mapped language")

    if born_country:
        born_lang = get_country_language(born_country)
        if born_lang:
            reason_parts.append(f"birth country language ({born_lang} from " f"{born_country}) is uncommon")
        else:
            reason_parts.append(f"birth country ({born_country}) has no mapped language")

    if not reason_parts:
        reason = "Defaulting to English (no country information available)"
    else:
        reason = f"Defaulting to English because {' and '.join(reason_parts)}"

    return "English", reason


# ---------- Translation Agent ----------
class DialogueTranslator:
    """
    Agent for translating dialogues to appropriate languages based on user profiles.
    """

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.3,
        verbose: int = 0,
    ):
        """
        Initialize the DialogueTranslator.

        Args:
            client: AsyncOpenAI client for translation
            model: Model to use for translation (default: gpt-4.1-mini)
            temperature: Temperature for translation (lower = more consistent)
            verbose: Verbosity level for logging
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.verbose = verbose

    async def translate_turn(
        self,
        user_content: str,
        chatbot_content: str,
        target_language: str,
    ) -> TranslatedTurn:
        """
        Translate a single dialogue turn (user input + chatbot response).

        Args:
            user_content: Original user message
            chatbot_content: Original chatbot response
            target_language: Target language for translation

        Returns:
            TranslatedTurn with both original and translated content
        """
        # If target is English, no translation needed
        if target_language == "English":
            return TranslatedTurn(
                user_content=user_content,
                chatbot_content=chatbot_content,
                original_user_content=user_content,
                original_chatbot_content=chatbot_content,
            )

        # Create translation prompt
        system_prompt = f"""You are a professional translator. Translate the following dialogue turn to {target_language}.
Maintain the tone, formality, and intent of the original messages.
Preserve any technical terms, proper nouns, or culturally specific references appropriately.

Return your translation in JSON format with two fields:
- "user_message": the translated user message
- "chatbot_message": the translated chatbot message"""

        user_prompt = f"""Translate this dialogue turn to {target_language}:

USER MESSAGE:
{user_content}

CHATBOT MESSAGE:
{chatbot_content}

Remember to return your response as JSON with "user_message" and "chatbot_message" fields."""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            if self.verbose >= 1:
                logger.info(
                    "[TRANSLATION] %s | user: %s... | chatbot: %s...",
                    target_language,
                    result.get("user_message", "")[:50],
                    result.get("chatbot_message", "")[:50],
                )

            return TranslatedTurn(
                user_content=result.get("user_message", user_content),
                chatbot_content=result.get("chatbot_message", chatbot_content),
                original_user_content=user_content,
                original_chatbot_content=chatbot_content,
            )

        except Exception as e:
            logger.error("Translation failed: %s", e)
            # Return original content if translation fails
            return TranslatedTurn(
                user_content=user_content,
                chatbot_content=chatbot_content,
                original_user_content=user_content,
                original_chatbot_content=chatbot_content,
            )

    async def translate_dialogue(
        self,
        dialogue_data: Dict[str, Any],
        user_profile: Optional[Dict[str, Any]] = None,
        force_language: Optional[str] = None,
    ) -> TranslatedDialogue:
        """
        Translate an entire dialogue based on user profile.

        Args:
            dialogue_data: Original dialogue data with 'index' and 'generated_dialogue'
            user_profile: User profile for language selection
            force_language: Optional language override

        Returns:
            TranslatedDialogue with all turns translated
        """
        # Determine target language
        if force_language:
            target_language = force_language
            reason = f"Language forced to {force_language}"
        elif user_profile:
            target_language, reason = select_target_language(user_profile)
        else:
            target_language = "English"
            reason = "No user profile provided, defaulting to English"

        if self.verbose >= 1:
            logger.info(
                "[DIALOGUE %d] %s",
                dialogue_data.get("index", -1),
                reason,
            )

        # Translate each turn
        translated_turns = []
        for turn in dialogue_data.get("generated_dialogue", []):
            translated_turn = await self.translate_turn(
                user_content=turn.get("user_content", ""),
                chatbot_content=turn.get("chatbot_content", ""),
                target_language=target_language,
            )
            translated_turns.append(translated_turn)

        return TranslatedDialogue(
            index=dialogue_data.get("index", -1),
            target_language=target_language,
            language_selection_reason=reason,
            translated_dialogue=translated_turns,
            user_profile=user_profile,
        )

    async def translate_dialogue_file(
        self,
        input_file: Path | str,
        output_file: Path | str,
        user_profiles: Optional[Dict[int, Dict[str, Any]]] = None,
        max_dialogues: Optional[int] = None,
    ) -> List[TranslatedDialogue]:
        """
        Translate all dialogues from a JSONL file.

        Args:
            input_file: Path to input JSONL file with dialogues
            output_file: Path to output JSONL file for translated dialogues
            user_profiles: Optional mapping of dialogue index to user profile
            max_dialogues: Optional limit on number of dialogues to translate

        Returns:
            List of translated dialogues
        """
        input_file = Path(input_file)
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        translated_dialogues = []

        with open(input_file, "r", encoding="utf-8") as f_in:
            with open(output_file, "w", encoding="utf-8") as f_out:
                for i, line in enumerate(f_in):
                    if max_dialogues and i >= max_dialogues:
                        break

                    dialogue_data = json.loads(line)
                    idx = dialogue_data.get("index", i)

                    # Get user profile if available
                    user_profile = user_profiles.get(idx) if user_profiles else None

                    # Translate dialogue
                    translated = await self.translate_dialogue(
                        dialogue_data=dialogue_data,
                        user_profile=user_profile,
                    )

                    # Write to output file
                    f_out.write(translated.model_dump_json() + "\n")
                    translated_dialogues.append(translated)

                    if self.verbose >= 1 and (i + 1) % 10 == 0:
                        logger.info("Translated %d dialogues...", i + 1)

        logger.info("Translation complete. Output saved to %s", output_file)
        return translated_dialogues


# ---------- Helper function to load user profiles ----------
def load_user_profiles_from_csv(csv_file: Path | str, id_column: str = "D_INTERVIEW") -> Dict[Any, Dict[str, Any]]:
    """
    Load user profiles from a CSV file.

    Args:
        csv_file: Path to CSV file with user profiles
        id_column: Column name to use as dictionary key (default: D_INTERVIEW)

    Returns:
        Dictionary mapping ID (from id_column) to user profile
        If id_column not found, falls back to integer index
    """
    import csv

    csv_file = Path(csv_file)
    profiles = {}

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Try to use specified ID column as key
            if id_column in row and row[id_column]:
                key = row[id_column]
                # Try to convert to int if it's numeric
                try:
                    key = int(key)
                except (ValueError, TypeError):
                    pass  # Keep as string
            else:
                # Fallback to integer index
                key = i

            profiles[key] = dict(row)

    return profiles


# ---------- Main execution example ----------
async def main():
    """Example usage of the DialogueTranslator."""
    import argparse

    parser = argparse.ArgumentParser(description="Translate generated dialogues")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with dialogues",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file for translated dialogues",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        help="CSV file with user profiles",
    )
    parser.add_argument(
        "--max",
        type=int,
        help="Maximum number of dialogues to translate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="Model to use for translation",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level",
    )

    args = parser.parse_args()

    # Load user profiles if provided
    user_profiles = None
    if args.profiles:
        user_profiles = load_user_profiles_from_csv(args.profiles)
        logger.info("Loaded %d user profiles", len(user_profiles))

    # Initialize client and translator
    client = AsyncOpenAI()
    translator = DialogueTranslator(
        client=client,
        model=args.model,
        verbose=args.verbose,
    )

    # Translate dialogues
    await translator.translate_dialogue_file(
        input_file=args.input,
        output_file=args.output,
        user_profiles=user_profiles,
        max_dialogues=args.max,
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
