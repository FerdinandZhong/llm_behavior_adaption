"""Unit tests for llm_behavior_adaptation.value_measurement.measurement_utils module."""

from llm_behavior_adaptation.value_measurement.measurement_utils import (
    get_continent,
    get_culture,
    get_development_level,
)


class TestGetContinent:
    """Test get_continent function."""

    def test_african_country(self):
        """Test mapping African countries to Africa."""
        assert get_continent("Nigeria") == "Africa"
        assert get_continent("Egypt") == "Africa"
        assert get_continent("South Africa") == "Africa"

    def test_asian_country(self):
        """Test mapping Asian countries to Asia."""
        assert get_continent("China") == "Asia"
        assert get_continent("Japan") == "Asia"
        assert get_continent("India") == "Asia"

    def test_european_country(self):
        """Test mapping European countries to Europe."""
        assert get_continent("Germany") == "Europe"
        assert get_continent("France") == "Europe"
        assert get_continent("United Kingdom") == "Europe"

    def test_north_american_country(self):
        """Test mapping North American countries to America."""
        assert get_continent("United States") == "America"
        assert get_continent("Canada") == "America"
        assert get_continent("Mexico") == "America"

    def test_south_american_country(self):
        """Test mapping South American countries to America."""
        assert get_continent("Brazil") == "America"
        assert get_continent("Argentina") == "America"
        assert get_continent("Chile") == "America"

    def test_oceania_mapped_to_unknown(self):
        """Test that Oceania countries are mapped to Unknown."""
        # According to the code, Oceania is mapped to "Unknown"
        assert get_continent("Australia") == "Unknown"
        assert get_continent("New Zealand") == "Unknown"

    def test_micronesia_special_case(self):
        """Test special handling of Micronesia."""
        assert get_continent("Micronesia") == "Unknown"

    def test_pitcairn_islands_special_case(self):
        """Test special handling of Pitcairn Islands."""
        assert get_continent("pitcairn islands") == "Unknown"

    def test_invalid_country_name(self):
        """Test handling of invalid country names."""
        assert get_continent("NotACountry") == "Unknown"
        assert get_continent("XYZ123") == "Unknown"

    def test_case_sensitivity(self):
        """Test that function handles different cases correctly."""
        # pycountry.lookup is case-insensitive
        assert get_continent("germany") == "Europe"
        assert get_continent("GERMANY") == "Europe"


class TestGetCulture:
    """Test get_culture function."""

    def test_known_country(self):
        """Test getting culture for known countries."""
        # Without knowing the exact mapping, test that it returns a string
        result = get_culture("United States")
        assert isinstance(result, str)
        assert result in [
            "Western",
            "Other",
        ]  # Could be in mapping or default

    def test_micronesia_special_case(self):
        """Test special handling of Micronesia."""
        assert get_culture("Micronesia") == "Oceania"

    def test_pitcairn_islands_special_case(self):
        """Test special handling of Pitcairn Islands."""
        assert get_culture("pitcairn islands") == "Oceania"

    def test_invalid_country_name(self):
        """Test handling of invalid country names."""
        assert get_culture("NotACountry") == "Other"
        assert get_culture("XYZ123") == "Other"

    def test_returns_string(self):
        """Test that function always returns a string."""
        countries = ["China", "Brazil", "Germany", "Kenya"]
        for country in countries:
            result = get_culture(country)
            assert isinstance(result, str)

    def test_various_countries(self):
        """Test that various countries return valid results."""
        countries = [
            "United Kingdom",
            "France",
            "China",
            "India",
            "Brazil",
            "Nigeria",
        ]
        for country in countries:
            result = get_culture(country)
            assert isinstance(result, str)
            assert len(result) > 0  # Should not be empty


class TestGetDevelopmentLevel:
    """Test get_development_level function."""

    def test_known_country(self):
        """Test getting development level for known countries."""
        # Test that function returns a valid string
        result = get_development_level("United States")
        assert isinstance(result, str)

    def test_invalid_country_name(self):
        """Test handling of invalid country names."""
        assert get_development_level("NotACountry") == "Unknown"
        assert get_development_level("XYZ123") == "Unknown"

    def test_returns_string(self):
        """Test that function always returns a string."""
        countries = ["China", "Brazil", "Germany", "Kenya"]
        for country in countries:
            result = get_development_level(country)
            assert isinstance(result, str)

    def test_various_countries(self):
        """Test that various countries return valid results."""
        countries = [
            "United Kingdom",
            "France",
            "China",
            "India",
            "Brazil",
            "Nigeria",
        ]
        for country in countries:
            result = get_development_level(country)
            assert isinstance(result, str)
            # Should return either a development level or "Unknown"
            assert len(result) > 0

    def test_case_handling(self):
        """Test that function handles different cases correctly."""
        # pycountry.lookup is case-insensitive
        result1 = get_development_level("germany")
        result2 = get_development_level("GERMANY")
        result3 = get_development_level("Germany")
        # All should return the same result
        assert result1 == result2 == result3

    def test_empty_string(self):
        """Test handling of empty string."""
        result = get_development_level("")
        assert result == "Unknown"


class TestJobClassifier:
    """Test JobClassifier class."""

    def test_job_classifier_attributes(self):
        """Test that JobClassifier has required attributes."""
        from llm_behavior_adaptation.value_measurement.measurement_utils import JobClassifier

        assert hasattr(JobClassifier, "position_levels")
        assert hasattr(JobClassifier, "sector_levels")
        assert hasattr(JobClassifier, "job_categories")

    def test_position_levels_list(self):
        """Test that position_levels is properly defined."""
        from llm_behavior_adaptation.value_measurement.measurement_utils import JobClassifier

        assert isinstance(JobClassifier.position_levels, list)
        assert len(JobClassifier.position_levels) == 5
        assert "Entry Level" in JobClassifier.position_levels
        assert "C-Suite" in JobClassifier.position_levels

    def test_sector_levels_list(self):
        """Test that sector_levels is properly defined."""
        from llm_behavior_adaptation.value_measurement.measurement_utils import JobClassifier

        assert isinstance(JobClassifier.sector_levels, list)
        assert "Tech" in JobClassifier.sector_levels
        assert "Finance" in JobClassifier.sector_levels

    def test_job_categories_list(self):
        """Test that job_categories is properly defined."""
        from llm_behavior_adaptation.value_measurement.measurement_utils import JobClassifier

        assert isinstance(JobClassifier.job_categories, list)
        assert "Tech" in JobClassifier.job_categories
        assert "Health" in JobClassifier.job_categories

    # Note: Testing the actual classifier would require loading the model
    # which is expensive and depends on transformers/torch being installed
    # For unit tests, we test the structure and attributes only
