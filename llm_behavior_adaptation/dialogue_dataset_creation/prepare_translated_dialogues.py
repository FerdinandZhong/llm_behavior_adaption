"""
Convert translated dialogue JSONL files into the format expected by wvs_values_prediction.py.

Input format (per line):
  {
    "index": 0,
    "target_language": "Serbian",
    "language_selection_reason": "...",
    "translated_dialogue": [
      {"user_content": "...", "chatbot_content": "...",
       "original_user_content": "...", "original_chatbot_content": "..."},
      ...
    ],
    "user_profile": {"D_INTERVIEW": "688070395", ...}
  }

Output format (per line, compatible with wvs_values_prediction.py):
  {"688070395": [{"role": "user", "content": "..."}, {"role": "chatbot", "content": "..."}, ...]}
"""

import argparse
import json
from pathlib import Path


def convert(input_path: str, output_path: str, use_original: bool = False) -> None:
    """
    Convert translated JSONL to pipeline-compatible format.

    Args:
        input_path: Path to translated dialogue JSONL.
        output_path: Path to write converted JSONL.
        use_original: If True, use original English content instead of translated.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: JSON parse error — {e}. Skipping.")
                skipped += 1
                continue

            user_profile = entry.get("user_profile", {})
            user_id = str(user_profile.get("D_INTERVIEW", ""))
            if not user_id:
                print(f"[WARN] Line {line_num}: missing D_INTERVIEW. Skipping.")
                skipped += 1
                continue

            raw_turns = entry.get("translated_dialogue", [])
            messages = []
            for turn in raw_turns:
                if use_original:
                    user_text = turn.get("original_user_content", "")
                    bot_text = turn.get("original_chatbot_content", "")
                else:
                    user_text = turn.get("user_content", "")
                    bot_text = turn.get("chatbot_content", "")

                if user_text:
                    messages.append({"role": "user", "content": user_text})
                if bot_text:
                    messages.append({"role": "chatbot", "content": bot_text})

            fout.write(json.dumps({user_id: messages}, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Done. Converted {converted} dialogues, skipped {skipped}.")
    print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert translated dialogue JSONL to wvs_values_prediction-compatible format."
    )
    parser.add_argument("input", help="Path to translated JSONL file")
    parser.add_argument("output", help="Path to write converted JSONL file")
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="Use original English content instead of translated text",
    )
    args = parser.parse_args()
    convert(args.input, args.output, use_original=args.use_original)


if __name__ == "__main__":
    main()
