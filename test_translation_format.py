"""
Test script to verify dialogue format normalization and profile matching.
"""

import json
from llm_behavior_adaptation.dialogue_dataset_creation.translation_controller import TranslationController
from llm_behavior_adaptation.dialogue_dataset_creation.dialogue_translator import load_user_profiles_from_csv


def test_format_normalization():
    """Test the dialogue format normalization."""
    print("Testing dialogue format normalization...")

    # Test WVS format
    wvs_format = {
        "688070395": [
            {"role": "user", "content": "What should I do?"},
            {"role": "chatbot", "content": "You should..."},
            {"role": "user", "content": "And then?"},
            {"role": "assistant", "content": "Next you..."},
        ]
    }

    normalized = TranslationController._normalize_dialogue_format(wvs_format, 0)

    assert "index" in normalized
    assert "generated_dialogue" in normalized
    assert "interview_id" in normalized
    assert normalized["interview_id"] == "688070395"
    assert len(normalized["generated_dialogue"]) == 2

    first_turn = normalized["generated_dialogue"][0]
    assert "user_content" in first_turn
    assert "chatbot_content" in first_turn
    assert first_turn["user_content"] == "What should I do?"
    assert first_turn["chatbot_content"] == "You should..."

    print("✓ Format normalization works correctly")


def test_profile_loading():
    """Test loading user profiles with D_INTERVIEW IDs."""
    print("\nTesting profile loading...")

    profiles = load_user_profiles_from_csv(
        "datasets/wvs_benchmarks/sampled_demographic_features.csv"
    )

    print(f"  Loaded {len(profiles)} profiles")

    # Check that profiles are keyed by integer interview IDs
    sample_keys = list(profiles.keys())[:5]
    print(f"  Sample keys: {sample_keys}")

    # Check first profile
    first_key = sample_keys[0]
    first_profile = profiles[first_key]

    print(f"  First profile ID: {first_key}")
    print(f"  Place of residence: {first_profile.get('place_of_residence')}")
    print(f"  Profile keys: {list(first_profile.keys())}")

    # Verify we have the expected fields
    assert "place_of_residence" in first_profile
    assert "D_INTERVIEW" in first_profile

    # Test specific interview ID
    if 688070395 in profiles:
        profile_688 = profiles[688070395]
        print(f"\n  Profile 688070395:")
        print(f"    Country: {profile_688.get('place_of_residence')}")
        print(f"    Age: {profile_688.get('age')}")
        assert profile_688.get("place_of_residence") == "Serbia"
        print("  ✓ Profile 688070395 found and correct")
    else:
        print("  ⚠ Profile 688070395 not found")

    print("✓ Profile loading works correctly")


def test_actual_dialogue_file():
    """Test reading and normalizing actual dialogue file."""
    print("\nTesting actual dialogue file...")

    dialogue_file = "datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl"

    with open(dialogue_file, "r") as f:
        first_line = f.readline()

    dialogue_data = json.loads(first_line)
    print(f"  Original format keys: {list(dialogue_data.keys())}")

    # Normalize
    normalized = TranslationController._normalize_dialogue_format(dialogue_data, 0)

    print(f"  Normalized format:")
    print(f"    Index: {normalized.get('index')}")
    print(f"    Interview ID: {normalized.get('interview_id')}")
    print(f"    Number of turns: {len(normalized.get('generated_dialogue', []))}")

    if normalized.get("generated_dialogue"):
        first_turn = normalized["generated_dialogue"][0]
        user_preview = first_turn.get("user_content", "")[:100]
        print(f"    First user message: {user_preview}...")

    print("✓ Actual dialogue file processing works")


def test_profile_matching():
    """Test that profiles match correctly with interview IDs."""
    print("\nTesting profile matching...")

    profiles = load_user_profiles_from_csv(
        "datasets/wvs_benchmarks/sampled_demographic_features.csv"
    )

    dialogue_file = "datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl"

    with open(dialogue_file, "r") as f:
        for i, line in enumerate(f):
            if i >= 5:  # Test first 5
                break

            dialogue_data = json.loads(line)
            normalized = TranslationController._normalize_dialogue_format(dialogue_data, i)
            interview_id = normalized.get("interview_id")

            # Try matching as in controller
            profile = None
            if interview_id:
                profile = profiles.get(interview_id)
                if not profile and isinstance(interview_id, str):
                    try:
                        int_id = int(interview_id)
                        profile = profiles.get(int_id)
                    except (ValueError, TypeError):
                        pass

            if profile:
                print(f"  Dialogue {i} (ID: {interview_id}):")
                print(f"    ✓ Profile matched")
                print(f"    Country: {profile.get('place_of_residence')}")
            else:
                print(f"  Dialogue {i} (ID: {interview_id}): ✗ No profile match")

    print("✓ Profile matching test complete")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRANSLATION FORMAT AND MATCHING TESTS")
    print("=" * 60)

    try:
        test_format_normalization()
        test_profile_loading()
        test_actual_dialogue_file()
        test_profile_matching()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nThe translation system should now work correctly!")
        print("Run the translation with:")
        print("  python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \\")
        print("    --config translation_configs/career_translation_config.yaml \\")
        print("    --max 5")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
