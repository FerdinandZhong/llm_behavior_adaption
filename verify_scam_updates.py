#!/usr/bin/env python3
"""Verification script for scam adaptation updates."""

import sys


def verify_implementation():
    """Verify that all updates are present in the code."""

    with open("llm_behavior_adaptation/value_measurement/wvs_scam_adaption.py", "r") as f:
        content = f.read()

    checks = {
        "1. _pick_farthest_scam_option method exists": "_pick_farthest_scam_option" in content,
        "2. Uses _pick_farthest_scam_option in scam selection": "self._pick_farthest_scam_option(" in content,
        "3. summary_results field initialized": '"summary_results"' in content,
        "4. summary_results populated": 'one_user_results["summary_results"]' in content,
        "5. model_initial_choice in results": '"model_initial_choice"' in content,
        "6. human_choice in results": '"human_choice"' in content,
        "7. scam_option in results": '"scam_option"' in content,
        "8. model_scam_response in results": '"model_scam_response"' in content,
        "9. switched_to_scam flag": '"switched_to_scam"' in content,
        "10. switched_to_human flag": '"switched_to_human"' in content,
        "11. maintained_initial flag": '"maintained_initial"' in content,
    }

    print("\n" + "=" * 70)
    print("SCAM ADAPTATION IMPLEMENTATION VERIFICATION")
    print("=" * 70 + "\n")

    all_pass = True
    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check}")
        if not result:
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("✓ ALL CHECKS PASSED - Implementation is complete!")
    else:
        print("✗ Some checks failed - Review implementation")
        sys.exit(1)
    print("=" * 70 + "\n")

    # Show key code snippets
    print("\nKEY IMPLEMENTATION DETAILS:\n")

    # Find and show the _pick_farthest_scam_option method signature
    if "def _pick_farthest_scam_option" in content:
        start = content.find("def _pick_farthest_scam_option")
        end = content.find("\n    def ", start + 1)
        method_text = content[start:end]
        lines = method_text.split("\n")[:15]  # First 15 lines
        print("_pick_farthest_scam_option method:")
        for line in lines:
            print(f"  {line}")

    print("\n" + "=" * 70 + "\n")
    print("NEW OUTPUT STRUCTURE:")
    print(
        """
{
  "user_id": {
    ...existing fields...,
    "summary_results": {                    ← NEW FIELD
      "category_name": [
        {
          "question_id": "Q3",
          "model_initial_choice": 3,        ← Model's initial prediction
          "human_choice": 1,                ← Human's actual choice
          "scam_option": 7,                 ← Farthest from both 3 & 1
          "tested": true,
          "model_scam_response": 7,         ← Model's response to scam
          "switched_to_scam": true,         ← Boolean: Switched to scam?
          "switched_to_human": false,       ← Boolean: Switched to human?
          "maintained_initial": false       ← Boolean: Maintained initial?
        }
      ]
    }
  }
}
"""
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    verify_implementation()
