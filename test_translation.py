"""
Test script for dialogue translation system.

This script tests the language selection logic without making API calls.
"""

from llm_behavior_adaptation.dialogue_dataset_creation.dialogue_translator import (
    select_target_language,
    is_common_language,
    get_country_language,
    COMMON_LANGUAGES,
    COUNTRY_LANGUAGE_MAP,
)


def test_common_language_check():
    """Test the common language checker."""
    print("Testing common language checker...")

    # Test common languages
    assert is_common_language("English") == True
    assert is_common_language("Spanish") == True
    assert is_common_language("Chinese") == True
    assert is_common_language("Japanese") == True

    # Test uncommon languages (not in the list)
    assert is_common_language("Greenlandic") == False
    assert is_common_language("Tagalog") == False
    assert is_common_language("") == False
    assert is_common_language(None) == False

    print("✓ Common language check tests passed")


def test_country_language_mapping():
    """Test country to language mapping."""
    print("\nTesting country to language mapping...")

    # Test major countries
    assert get_country_language("United States") == "English"
    assert get_country_language("France") == "French"
    assert get_country_language("Spain") == "Spanish"
    assert get_country_language("China") == "Chinese"
    assert get_country_language("Japan") == "Japanese"
    assert get_country_language("Germany") == "German"

    # Test countries with no mapping
    assert get_country_language("Micronesia") == None
    assert get_country_language("Greenland") == None
    assert get_country_language("Invalid Country") == None

    print("✓ Country language mapping tests passed")


def test_language_selection():
    """Test the language selection logic with various scenarios."""
    print("\nTesting language selection logic...")

    # Scenario 1: Common language from current residence
    profile1 = {
        "place_of_residence": "France",
        "place_of_birth": "United States",
    }
    lang1, reason1 = select_target_language(profile1)
    assert lang1 == "French"
    assert "France" in reason1
    print(f"  Scenario 1: {lang1} - {reason1}")

    # Scenario 2: Fallback to birth country
    profile2 = {
        "place_of_residence": "Greenland",
        "place_of_birth": "Denmark",
    }
    lang2, reason2 = select_target_language(profile2)
    assert lang2 == "Danish"
    assert "Denmark" in reason2
    print(f"  Scenario 2: {lang2} - {reason2}")

    # Scenario 3: Default to English (both uncommon)
    profile3 = {
        "place_of_residence": "Micronesia",
        "place_of_birth": "Greenland",
    }
    lang3, reason3 = select_target_language(profile3)
    assert lang3 == "English"
    assert "Defaulting to English" in reason3
    print(f"  Scenario 3: {lang3} - {reason3}")

    # Scenario 4: Only current residence available
    profile4 = {
        "place_of_residence": "Japan",
    }
    lang4, reason4 = select_target_language(profile4)
    assert lang4 == "Japanese"
    assert "Japan" in reason4
    print(f"  Scenario 4: {lang4} - {reason4}")

    # Scenario 5: Empty profile
    profile5 = {}
    lang5, reason5 = select_target_language(profile5)
    assert lang5 == "English"
    assert "no country information" in reason5
    print(f"  Scenario 5: {lang5} - {reason5}")

    # Scenario 6: Spanish-speaking countries
    for country in ["Spain", "Mexico", "Argentina", "Colombia"]:
        profile = {"place_of_residence": country}
        lang, reason = select_target_language(profile)
        assert lang == "Spanish", f"Expected Spanish for {country}, got {lang}"
    print(f"  Scenario 6: Spanish-speaking countries ✓")

    # Scenario 7: Chinese-speaking regions
    for country in ["China", "Taiwan", "Singapore", "Hong Kong"]:
        profile = {"place_of_residence": country}
        lang, reason = select_target_language(profile)
        assert lang == "Chinese", f"Expected Chinese for {country}, got {lang}"
    print(f"  Scenario 7: Chinese-speaking regions ✓")

    print("✓ Language selection tests passed")


def test_system_coverage():
    """Test coverage of the language system."""
    print("\nTesting system coverage...")

    # Check number of common languages
    print(f"  Common languages supported: {len(COMMON_LANGUAGES)}")
    assert len(COMMON_LANGUAGES) > 20, "Should support at least 20 common languages"

    # Check number of country mappings
    print(f"  Countries mapped: {len(COUNTRY_LANGUAGE_MAP)}")
    assert len(COUNTRY_LANGUAGE_MAP) > 50, "Should have mappings for at least 50 countries"

    # Verify all mapped languages are in common languages
    mapped_languages = set(COUNTRY_LANGUAGE_MAP.values())
    uncommon_mapped = mapped_languages - COMMON_LANGUAGES
    if uncommon_mapped:
        print(f"  WARNING: These mapped languages are not in COMMON_LANGUAGES: {uncommon_mapped}")
    else:
        print(f"  ✓ All mapped languages are marked as common")

    # Display language distribution
    from collections import Counter
    lang_dist = Counter(COUNTRY_LANGUAGE_MAP.values())
    print(f"\n  Most common languages in mapping:")
    for lang, count in lang_dist.most_common(10):
        print(f"    {lang}: {count} countries")

    print("\n✓ System coverage tests passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\nTesting edge cases...")

    # Case-sensitive country names
    profile1 = {"place_of_residence": "united states"}  # lowercase
    lang1, _ = select_target_language(profile1)
    # Should default to English since exact match not found
    assert lang1 == "English"
    print(f"  Case sensitivity: Handled correctly (defaults to English)")

    # Missing fields
    profile2 = {"age": 30, "gender": "Male"}  # no country fields
    lang2, _ = select_target_language(profile2)
    assert lang2 == "English"
    print(f"  Missing country fields: Defaults to English ✓")

    # None values
    profile3 = {"place_of_residence": None, "place_of_birth": None}
    lang3, _ = select_target_language(profile3)
    assert lang3 == "English"
    print(f"  None values: Defaults to English ✓")

    print("✓ Edge case tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("TRANSLATION SYSTEM TEST SUITE")
    print("=" * 60)

    try:
        test_common_language_check()
        test_country_language_mapping()
        test_language_selection()
        test_system_coverage()
        test_edge_cases()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
