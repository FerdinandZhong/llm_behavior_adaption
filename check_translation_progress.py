"""
Helper script to check translation progress and determine resume point.

Usage:
    python check_translation_progress.py <translated_file.jsonl>
    python check_translation_progress.py wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl
"""

import json
import sys
from pathlib import Path


def check_progress(translated_file: str, input_file: str = None):
    """
    Check translation progress and provide resume information.

    Args:
        translated_file: Path to the translated output file
        input_file: Optional path to input file to compare against
    """
    translated_path = Path(translated_file)

    print("=" * 70)
    print("TRANSLATION PROGRESS CHECK")
    print("=" * 70)

    # Check if file exists
    if not translated_path.exists():
        print(f"\n❌ Translated file not found: {translated_path}")
        print("\nThis means translation hasn't started yet.")
        print("Start from the beginning with: --starting-row 0")
        return

    # Count translated dialogues
    translated_count = 0
    last_index = -1
    interview_ids = []

    try:
        with open(translated_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    translated_count += 1
                    data = json.loads(line)
                    last_index = data.get("index", -1)
                    interview_ids.append(last_index)
    except Exception as e:
        print(f"\n❌ Error reading translated file: {e}")
        return

    print(f"\n📊 Translation Progress:")
    print(f"   Translated file: {translated_path}")
    print(f"   Dialogues completed: {translated_count}")
    print(f"   Last processed index: {last_index}")

    if translated_count > 0:
        print(f"   First 5 indices: {interview_ids[:5]}")
        print(f"   Last 5 indices: {interview_ids[-5:]}")

    # Check input file if provided
    if input_file:
        input_path = Path(input_file)
        if input_path.exists():
            total_input = sum(1 for _ in open(input_path, "r", encoding="utf-8"))
            remaining = total_input - translated_count

            print(f"\n📁 Input File:")
            print(f"   Input file: {input_path}")
            print(f"   Total dialogues: {total_input}")
            print(f"   Remaining: {remaining}")
            print(f"   Progress: {translated_count}/{total_input} ({100 * translated_count / total_input:.1f}%)")

            # Provide resume command
            if remaining > 0:
                print(f"\n✅ To resume translation:")
                print(f"\n   Using CLI:")
                print(f"   python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \\")
                print(f"       --config <your_config.yaml> \\")
                print(f"       --starting-row {last_index + 1}")

                print(f"\n   Or update config YAML:")
                print(f"   starting_row: {last_index + 1}")
                print(f"   ending_row: null  # or specify end index")
            else:
                print(f"\n✅ Translation complete!")
        else:
            print(f"\n⚠️  Input file not found: {input_path}")

    # Check for gaps in indices
    if len(interview_ids) > 1:
        sorted_ids = sorted(interview_ids)
        gaps = []
        for i in range(len(sorted_ids) - 1):
            if sorted_ids[i + 1] - sorted_ids[i] > 1:
                gaps.append((sorted_ids[i], sorted_ids[i + 1]))

        if gaps:
            print(f"\n⚠️  Warning: Found {len(gaps)} gaps in indices:")
            for start, end in gaps[:5]:  # Show first 5 gaps
                print(f"   Gap between {start} and {end}")
            if len(gaps) > 5:
                print(f"   ... and {len(gaps) - 5} more gaps")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python check_translation_progress.py <translated_file.jsonl> [input_file.jsonl]")
        print("\nExample:")
        print("  python check_translation_progress.py \\")
        print("      wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl \\")
        print("      datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl")
        sys.exit(1)

    translated_file = sys.argv[1]
    input_file = sys.argv[2] if len(sys.argv) > 2 else None

    check_progress(translated_file, input_file)


if __name__ == "__main__":
    main()
