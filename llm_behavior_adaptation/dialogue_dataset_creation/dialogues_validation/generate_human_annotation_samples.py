"""
Generate Human Annotation Samples for Dialogue Quality Evaluation.

This script randomly selects dialogues from a dataset and generates a scoring file
for human annotators to evaluate dialogue quality using a five-criterion rubric.

Output formats:
- Excel (.xlsx): Human-friendly format with instructions, rubric, and scoring sheets
- CSV (.csv): Simple format for programmatic processing

Usage:
    python generate_human_annotation_samples.py --config config.yaml
    python generate_human_annotation_samples.py --topic "career advice" --num-samples 50
"""

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

# Optional dependency for Excel output
try:
    pass

    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# Scoring Rubric for Human Annotators
# =============================================================================

SCORING_RUBRIC = {
    "coverage": {
        "name": "Attribute Coverage",
        "range": "0-5",
        "description": """Count how many demographic attributes from the user profile are
mentioned or clearly inferable from the dialogue.

Target attributes to look for:
1. Age (or age-related hints: "recent graduate", "mid-career", "near retirement")
2. Education level (degree, qualifications, academic background)
3. Occupation/Job title (current role, profession, industry)
4. Socioeconomic status (income hints, lifestyle indicators, class markers)
5. Geographic location (country, city, region)

Scoring:
- 0: No attributes mentioned
- 1: 1 attribute mentioned/inferable
- 2: 2 attributes mentioned/inferable
- 3: 3 attributes mentioned/inferable
- 4: 4 attributes mentioned/inferable
- 5: All 5 attributes mentioned/inferable""",
    },
    "correctness": {
        "name": "Attribute Correctness",
        "range": "0-5",
        "description": """Of the attributes mentioned in the dialogue, how many CORRECTLY
match the ground truth user profile?

Important: This score is bounded by the coverage score. If only 3 attributes are
mentioned, the maximum correctness score is 3.

Scoring:
- Count only attributes that are BOTH mentioned AND correct
- An attribute mentioned with wrong value = incorrect (don't count)
- An attribute not mentioned = N/A (don't count either way)

Example: If profile says "Bachelor's Degree" but dialogue mentions "PhD" = incorrect""",
    },
    "diversity": {
        "name": "Question Diversity",
        "range": "1-5",
        "description": """Assess the variety of topics and question types across dialogue turns.

Consider:
- Number of distinct subtopics covered within the main topic
- Variety in question framing (advice-seeking, information-gathering, opinion-based)
- Progressive depth vs. repetitive surface-level questions

Scoring:
- 5: Questions span ≥4 distinct subtopics with varied question types
- 4: 3 distinct subtopics with some variety in framing
- 3: 2 subtopics; questions show moderate variation
- 2: Single subtopic but different phrasings
- 1: Repetitive questions on a single narrow topic""",
    },
    "relevance": {
        "name": "Conversational Relevance",
        "range": "1-5",
        "description": """Assess whether the user's questions remain contextually appropriate
for the dialogue topic and maintain coherent flow with prior assistant responses.

Consider:
- Are questions on-topic for the stated scenario (e.g., career advice)?
- Do follow-up questions logically build on previous responses?
- Is there natural conversational progression?

Scoring:
- 5: All questions are topically appropriate and logically follow from prior context
- 4: Minor tangents but overall coherent progression
- 3: Some questions feel disconnected but remain on-topic
- 2: Frequent topic drift; weak connection to prior turns
- 1: Questions are off-topic or contextually incoherent""",
    },
    "naturalness": {
        "name": "Naturalness",
        "range": "1-5",
        "description": """Assess how organically demographic attributes are woven into
the conversation, rather than being artificially inserted.

Consider:
- Are attributes revealed through context (e.g., "as a recent graduate" vs. "I am 22")
- Does the information sharing feel natural for the conversation flow?
- Would a real user share information this way?

Scoring:
- 5: Attributes emerge naturally through context; feels like authentic conversation
- 4: Mostly natural with minor explicit statements
- 3: Mix of natural embedding and direct declarations
- 2: Attributes mostly stated explicitly and awkwardly
- 1: Attributes feel forced or artificially injected (e.g., listing profile at start)""",
    },
}


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML or JSON file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()

    with open(config_path, "r", encoding="utf-8") as f:
        if suffix in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        elif suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

    return config or {}


def format_dialogue_for_display(dialogue_data: List[Dict]) -> str:
    """Format dialogue turns into a readable string.

    Supports two formats:
    1. Legacy format: [{"user_content": ..., "chatbot_content": ...}, ...]
    2. WVS format: [{"role": "user", "content": ...}, {"role": "chatbot", "content": ...}, ...]
    """
    lines = []

    # Check if this is the WVS format (role/content pairs)
    if dialogue_data and "role" in dialogue_data[0]:
        turn_num = 0
        for turn in dialogue_data:
            role = turn.get("role", "").lower()
            content = turn.get("content", "")

            if role == "user":
                turn_num += 1
                lines.append(f"[Turn {turn_num}] USER: {content}")
            elif role in ["chatbot", "assistant"]:
                # Truncate very long responses for readability
                if len(content) > 1000:
                    content = content[:1000] + "... [truncated]"
                lines.append(f"[Turn {turn_num}] ASSISTANT: {content}")
                lines.append("")  # Empty line between turns
    else:
        # Legacy format
        for i, turn in enumerate(dialogue_data, 1):
            user_key = "user_content" if "user_content" in turn else "user"
            assistant_key = (
                "chatbot_content"
                if "chatbot_content" in turn
                else "assistant_content" if "assistant_content" in turn else "assistant"
            )

            if user_key in turn:
                lines.append(f"[Turn {i}] USER: {turn[user_key]}")
            if assistant_key in turn:
                # Truncate very long responses for readability
                response = turn[assistant_key]
                if len(response) > 1000:
                    response = response[:1000] + "... [truncated]"
                lines.append(f"[Turn {i}] ASSISTANT: {response}")
            lines.append("")  # Empty line between turns

    return "\n".join(lines)


def format_user_profile(row_dict: Dict) -> str:
    """Format user profile dictionary as string.

    Supports both legacy format and WVS format profiles.
    """
    # Key attributes to highlight (supporting both formats)
    # Legacy format keys
    legacy_key_attrs = [
        "Gender",
        "Date of Birth",
        "Country",
        "Education Level",
        "Years of Experience",
        "Job Title",
        "Desired Salary",
    ]
    # WVS format keys
    wvs_key_attrs = [
        "gender",
        "age",
        "place_of_residence",
        "continent_of_residence",
        "highest_level_of_education",
        "socioeconomic_status",
        "occupation_group",
        "immigration_status",
    ]
    # Keys to exclude from output
    exclude_keys = [
        "Applicant ID",
        "Application Date",
        "Phone Number",
        "Email",
        "Address",
        "Zip Code",
        "Status",
        "D_INTERVIEW",
    ]

    # Determine format based on keys present
    is_wvs_format = "gender" in row_dict or "D_INTERVIEW" in row_dict
    key_attrs = wvs_key_attrs if is_wvs_format else legacy_key_attrs

    # Format key for display (capitalize WVS keys)
    def format_key(key: str) -> str:
        if is_wvs_format:
            return key.replace("_", " ").title()
        return key

    lines = []
    for key in key_attrs:
        if key in row_dict and pd.notna(row_dict[key]):
            lines.append(f"{format_key(key)}: {row_dict[key]}")

    # Add any other non-empty attributes
    for key, value in row_dict.items():
        if key not in key_attrs and pd.notna(value) and key not in exclude_keys:
            lines.append(f"{format_key(key)}: {value}")

    return "\n".join(lines)


def generate_rubric_text() -> str:
    """Generate formatted rubric text for instructions."""
    lines = [
        "=" * 80,
        "DIALOGUE QUALITY SCORING RUBRIC",
        "=" * 80,
        "",
        "Please evaluate each dialogue on the following five criteria.",
        "Read the dialogue and user profile carefully before scoring.",
        "",
    ]

    for _criterion, details in SCORING_RUBRIC.items():
        lines.append("-" * 80)
        lines.append(f"CRITERION: {details['name']} (Score: {details['range']})")
        lines.append("-" * 80)
        lines.append(details["description"])
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


def create_annotation_dataframe(
    samples: List[Dict],
    user_profiles: pd.DataFrame,
    dialogue_topic: str,
) -> pd.DataFrame:
    """Create a DataFrame for human annotation.

    Supports two sample formats:
    1. Legacy: {"index": int, "generated_dialogue": [...]}
    2. WVS: {"interview_id": str, "dialogue": [...]}
    """
    rows = []

    # Check if profiles use D_INTERVIEW column (WVS format)
    is_wvs_profiles = "D_INTERVIEW" in user_profiles.columns

    # Create lookup dict for WVS profiles
    if is_wvs_profiles:
        profile_lookup = user_profiles.set_index("D_INTERVIEW").to_dict("index")

    for sample in samples:
        # Handle both legacy and WVS formats
        if "interview_id" in sample:
            # WVS format
            sample_id = sample["interview_id"]
            dialogue_data = sample.get("dialogue", [])

            # Look up profile by interview ID
            if is_wvs_profiles and sample_id in profile_lookup:
                profile_row = profile_lookup[sample_id]
                profile_str = format_user_profile(profile_row)
            else:
                profile_str = "[Profile not found]"
        else:
            # Legacy format
            sample_id = sample.get("index", 0)
            dialogue_data = sample.get("generated_dialogue", [])

            # Get corresponding user profile by index
            if sample_id < len(user_profiles):
                profile_row = user_profiles.iloc[sample_id].to_dict()
                profile_str = format_user_profile(profile_row)
            else:
                profile_str = "[Profile not found]"

        # Format dialogue
        dialogue_str = format_dialogue_for_display(dialogue_data)

        rows.append(
            {
                "Sample_ID": sample_id,
                "Dialogue_Topic": dialogue_topic,
                "User_Profile": profile_str,
                "Dialogue": dialogue_str,
                "Coverage_0to5": "",
                "Correctness_0to5": "",
                "Diversity_1to5": "",
                "Relevance_1to5": "",
                "Naturalness_1to5": "",
                "Annotator_Notes": "",
            }
        )

    return pd.DataFrame(rows)


def save_as_excel(
    df: pd.DataFrame,
    output_path: str,
    dialogue_topic: str,
) -> bool:
    """Save annotation data as Excel file with instructions sheet.

    Returns:
        True if successful, False if openpyxl is not available.
    """
    if not OPENPYXL_AVAILABLE:
        logger.warning(
            "openpyxl not installed. Install with 'pip install openpyxl' for Excel output. "
            "Falling back to CSV format."
        )
        return False

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: Instructions and Rubric
        instructions_df = pd.DataFrame(
            {
                "Instructions": [
                    "DIALOGUE QUALITY ANNOTATION TASK",
                    "",
                    f"Topic: {dialogue_topic}",
                    f"Number of samples: {len(df)}",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    "",
                    "HOW TO ANNOTATE:",
                    "1. Go to the 'Samples' sheet",
                    "2. For each sample, read the User_Profile and Dialogue",
                    "3. Score each criterion in the corresponding column",
                    "4. Add any notes in the Annotator_Notes column",
                    "",
                    "IMPORTANT:",
                    "- Read the full dialogue before scoring",
                    "- Compare dialogue content against the User_Profile",
                    "- Use the Rubric sheet for detailed scoring guidelines",
                    "",
                    "SCORING SUMMARY:",
                    "- Coverage (0-5): # of profile attributes mentioned",
                    "- Correctness (0-5): # of correctly mentioned attributes",
                    "- Diversity (1-5): Variety of question topics/types",
                    "- Relevance (1-5): Questions stay on-topic and coherent",
                    "- Naturalness (1-5): Attributes embedded organically",
                ]
            }
        )
        instructions_df.to_excel(writer, sheet_name="Instructions", index=False)

        # Sheet 2: Detailed Rubric
        rubric_rows = []
        for _criterion, details in SCORING_RUBRIC.items():
            rubric_rows.append(
                {
                    "Criterion": details["name"],
                    "Score_Range": details["range"],
                    "Description": details["description"],
                }
            )
        rubric_df = pd.DataFrame(rubric_rows)
        rubric_df.to_excel(writer, sheet_name="Rubric", index=False)

        # Sheet 3: Samples to annotate
        df.to_excel(writer, sheet_name="Samples", index=False)

        # Adjust column widths for Samples sheet
        worksheet = writer.sheets["Samples"]
        worksheet.column_dimensions["A"].width = 10  # Sample_ID
        worksheet.column_dimensions["B"].width = 15  # Dialogue_Topic
        worksheet.column_dimensions["C"].width = 40  # User_Profile
        worksheet.column_dimensions["D"].width = 80  # Dialogue
        worksheet.column_dimensions["E"].width = 15  # Coverage
        worksheet.column_dimensions["F"].width = 15  # Correctness
        worksheet.column_dimensions["G"].width = 15  # Diversity
        worksheet.column_dimensions["H"].width = 15  # Relevance
        worksheet.column_dimensions["I"].width = 15  # Naturalness
        worksheet.column_dimensions["J"].width = 30  # Notes

    logger.info(f"Excel file saved: {output_path}")
    return True


def save_as_csv(
    df: pd.DataFrame,
    output_path: str,
    rubric_path: Optional[str] = None,
) -> None:
    """Save annotation data as CSV file."""
    df.to_csv(output_path, index=False)
    logger.info(f"CSV file saved: {output_path}")

    # Optionally save rubric as separate file
    if rubric_path:
        with open(rubric_path, "w", encoding="utf-8") as f:
            f.write(generate_rubric_text())
        logger.info(f"Rubric saved: {rubric_path}")


def main():
    """Main entry point for generating human annotation samples."""
    parser = argparse.ArgumentParser(
        description="Generate human annotation samples for dialogue quality evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Using config file
  python generate_human_annotation_samples.py --config config.yaml

  # Using CLI arguments
  python generate_human_annotation_samples.py \\
      --dialogue-file dialogues.jsonl \\
      --user-profile-dataset profiles.csv \\
      --output-path annotations.xlsx \\
      --topic "career advice" \\
      --num-samples 50
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--dialogue-file",
        type=str,
        default=None,
        help="Path to generated dialogues JSONL file",
    )
    parser.add_argument(
        "--user-profile-dataset",
        type=str,
        default=None,
        help="Path to user profile CSV file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for annotation file (.xlsx or .csv)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="career advice",
        choices=["career advice", "investment advice"],
        help="Dialogue topic",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to select (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="excel",
        choices=["excel", "csv", "both"],
        help="Output format (default: excel)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load config file if provided
    config = {}
    if args.config:
        config = load_config_file(args.config)

    # Merge config with CLI args (CLI takes precedence)
    dialogue_file = args.dialogue_file or config.get("dialogue_file")
    user_profile_dataset = args.user_profile_dataset or config.get("user_profile_dataset")
    output_path = args.output_path or config.get("output_path")
    topic = args.topic if args.topic != "career advice" else config.get("topic", "career advice")
    num_samples = args.num_samples or config.get("num_samples", 50)
    seed = args.seed if args.seed is not None else config.get("seed")
    output_format = args.output_format or config.get("output_format", "excel")

    # Validate required arguments
    if not dialogue_file:
        raise ValueError("--dialogue-file is required")
    if not user_profile_dataset:
        raise ValueError("--user-profile-dataset is required")
    if not output_path:
        # Generate default output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"human_annotation_samples_{timestamp}.xlsx"

    # Set random seed
    if seed is not None:
        random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

    # Load dialogues
    logger.info(f"Loading dialogues from: {dialogue_file}")
    dialogues = []
    with open(dialogue_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # Check if this is WVS format (dict with interview_id as key)
            # WVS format: {"688070395": [{"role": "user", "content": ...}, ...]}
            if isinstance(data, dict) and "index" not in data and "generated_dialogue" not in data:
                # WVS format - convert to standard format
                for interview_id, dialogue in data.items():
                    dialogues.append(
                        {
                            "interview_id": int(interview_id) if interview_id.isdigit() else interview_id,
                            "dialogue": dialogue,
                        }
                    )
            else:
                # Legacy format: {"index": 0, "generated_dialogue": [...]}
                dialogues.append(data)
    logger.info(f"Loaded {len(dialogues)} dialogues")

    # Load user profiles
    logger.info(f"Loading user profiles from: {user_profile_dataset}")
    profiles_df = pd.read_csv(user_profile_dataset)
    profiles_df = profiles_df.loc[:, ~profiles_df.columns.str.contains("^Unnamed")]
    logger.info(f"Loaded {len(profiles_df)} user profiles")

    # Sample dialogues
    num_samples = min(num_samples, len(dialogues))
    sampled_dialogues = random.sample(dialogues, num_samples)
    logger.info(f"Randomly selected {num_samples} samples")

    # Create annotation DataFrame
    annotation_df = create_annotation_dataframe(
        samples=sampled_dialogues,
        user_profiles=profiles_df,
        dialogue_topic=topic,
    )

    # Save output
    output_path = Path(output_path)
    excel_saved = False

    if output_format in ["excel", "both"]:
        excel_path = output_path.with_suffix(".xlsx")
        excel_saved = save_as_excel(annotation_df, str(excel_path), topic)
        if excel_saved:
            print(f"Excel annotation file saved: {excel_path}")

    # Fall back to CSV if Excel failed or CSV was requested
    if output_format in ["csv", "both"] or (output_format == "excel" and not excel_saved):
        csv_path = output_path.with_suffix(".csv")
        rubric_path = output_path.with_name(f"{output_path.stem}_rubric.txt")
        save_as_csv(annotation_df, str(csv_path), str(rubric_path))
        print(f"CSV annotation file saved: {csv_path}")
        print(f"Rubric file saved: {rubric_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANNOTATION SAMPLE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Topic: {topic}")
    print(f"Samples selected: {num_samples}")
    print(f"Random seed: {seed if seed is not None else 'None (random)'}")
    print("=" * 60)
    print("\nScoring criteria included:")
    for _criterion, details in SCORING_RUBRIC.items():
        print(f"  - {details['name']} ({details['range']})")
    print("=" * 60)


if __name__ == "__main__":
    main()
