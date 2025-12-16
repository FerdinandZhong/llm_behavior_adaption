"""Unit tests for llm_behavior_adaptation.value_measurement.distances module."""

import numpy as np
import pytest

from llm_behavior_adaptation.value_measurement.distances import (
    _get_answer,
    _normalize,
    _safe_isnan,
    emd_between_vectors,
)


class TestGetAnswer:
    """Test _get_answer helper function."""

    def test_get_answer_from_dict(self):
        """Test getting answer from dictionary."""
        v = {"q1": 3, "q2": 4, "q3": 5}
        assert _get_answer("q1", v) == 3
        assert _get_answer("q2", v) == 4
        assert _get_answer("q3", v) == 5

    def test_get_answer_missing_key(self):
        """Test getting answer for missing key returns None."""
        v = {"q1": 3, "q2": 4}
        assert _get_answer("q3", v) is None

    def test_get_answer_from_sequence_raises_error(self):
        """Test that passing a sequence raises TypeError."""
        v = [1, 2, 3, 4, 5]
        with pytest.raises(TypeError, match="If v_a/v_b are sequences, call emd_between_vectors"):
            _get_answer("q1", v)


class TestNormalize:
    """Test _normalize helper function."""

    def test_normalize_with_normalization_enabled(self):
        """Test normalization when enabled."""
        result = _normalize(4.0, 1, 5, normalize_range=True)
        assert result == 1.0  # 4 / (5-1) = 1.0

    def test_normalize_with_normalization_disabled(self):
        """Test that normalization is skipped when disabled."""
        result = _normalize(4.0, 1, 5, normalize_range=False)
        assert result == 4.0

    def test_normalize_with_equal_bounds(self):
        """Test normalization when lo == hi."""
        result = _normalize(2.0, 5, 5, normalize_range=True)
        assert result == 2.0  # Should return original value

    def test_normalize_zero_distance(self):
        """Test normalization with zero distance."""
        result = _normalize(0.0, 1, 5, normalize_range=True)
        assert result == 0.0


class TestSafeIsnan:
    """Test _safe_isnan helper function."""

    def test_isnan_with_nan(self):
        """Test detection of NaN."""
        assert _safe_isnan(float("nan")) is True
        assert _safe_isnan(np.nan) is True

    def test_isnan_with_regular_numbers(self):
        """Test that regular numbers are not NaN."""
        assert _safe_isnan(0) is False
        assert _safe_isnan(1.5) is False
        assert _safe_isnan(-10) is False

    def test_isnan_with_infinity(self):
        """Test that infinity is not considered NaN."""
        assert _safe_isnan(float("inf")) is False
        assert _safe_isnan(float("-inf")) is False

    def test_isnan_with_none(self):
        """Test handling of None."""
        assert _safe_isnan(None) is False

    def test_isnan_with_string(self):
        """Test handling of string (should not raise exception)."""
        assert _safe_isnan("not a number") is False


class TestEMDBetweenVectors:
    """Test emd_between_vectors function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.questions = ["q1", "q2", "q3"]
        self.question_metadata = {
            "q1": {"answer_scale_min": 1, "answer_scale_max": 5},
            "q2": {"answer_scale_min": 1, "answer_scale_max": 5},
            "q3": {"answer_scale_min": 1, "answer_scale_max": 5},
        }

    def test_identical_vectors_mapping_mode(self):
        """Test EMD between identical vectors in mapping mode is 0."""
        v_a = {"q1": 3, "q2": 4, "q3": 2}
        v_b = {"q1": 3, "q2": 4, "q3": 2}
        agg_emd, per_q = emd_between_vectors(self.questions, v_a, v_b, self.question_metadata, vector_mode="mapping")
        assert np.isclose(agg_emd, 0.0)
        assert np.allclose(per_q, [0.0, 0.0, 0.0])

    def test_different_vectors_mapping_mode(self):
        """Test EMD between different vectors in mapping mode."""
        v_a = {"q1": 1, "q2": 1, "q3": 1}
        v_b = {"q1": 5, "q2": 5, "q3": 5}
        agg_emd, per_q = emd_between_vectors(self.questions, v_a, v_b, self.question_metadata, vector_mode="mapping")
        assert agg_emd > 0
        # With normalize=True, each question contributes |5-1|/(5-1) = 1.0
        assert np.allclose(per_q, [1.0, 1.0, 1.0])
        assert np.isclose(agg_emd, 1.0)

    def test_identical_vectors_sequence_mode(self):
        """Test EMD between identical vectors in sequence mode is 0."""
        v_a = [3, 4, 2]
        v_b = [3, 4, 2]
        agg_emd, per_q = emd_between_vectors(self.questions, v_a, v_b, self.question_metadata, vector_mode="sequence")
        assert np.isclose(agg_emd, 0.0)
        assert np.allclose(per_q, [0.0, 0.0, 0.0])

    def test_different_vectors_sequence_mode(self):
        """Test EMD between different vectors in sequence mode."""
        v_a = [1, 2, 3]
        v_b = [5, 4, 1]
        agg_emd, per_q = emd_between_vectors(self.questions, v_a, v_b, self.question_metadata, vector_mode="sequence")
        assert agg_emd > 0
        # q1: |1-5|/(5-1) = 1.0
        # q2: |2-4|/(5-1) = 0.5
        # q3: |3-1|/(5-1) = 0.5
        assert np.isclose(per_q[0], 1.0)
        assert np.isclose(per_q[1], 0.5)
        assert np.isclose(per_q[2], 0.5)

    def test_missing_values_in_mapping(self):
        """Test handling of missing values in mapping mode."""
        v_a = {"q1": 3, "q2": 4}  # q3 missing
        v_b = {"q1": 3, "q2": 5, "q3": 2}
        agg_emd, per_q = emd_between_vectors(self.questions, v_a, v_b, self.question_metadata, vector_mode="mapping")
        # q1: same, q2: different, q3: missing
        assert np.isclose(per_q[0], 0.0)  # q1
        assert per_q[1] > 0  # q2
        assert np.isnan(per_q[2])  # q3 missing

    def test_nan_values(self):
        """Test handling of NaN values."""
        v_a = [3, float("nan"), 2]
        v_b = [3, 4, 2]
        agg_emd, per_q = emd_between_vectors(self.questions, v_a, v_b, self.question_metadata, vector_mode="sequence")
        # q1: same, q2: nan, q3: same
        assert np.isclose(per_q[0], 0.0)
        assert np.isnan(per_q[1])
        assert np.isclose(per_q[2], 0.0)
        # Average should only consider valid values
        assert np.isfinite(agg_emd)

    def test_out_of_range_values(self):
        """Test handling of out-of-range values."""
        v_a = [0, 2, 6]  # q1 and q3 out of range
        v_b = [3, 3, 3]
        agg_emd, per_q = emd_between_vectors(self.questions, v_a, v_b, self.question_metadata, vector_mode="sequence")
        # q1 and q3 should be skipped (NaN)
        assert np.isnan(per_q[0])
        assert np.isfinite(per_q[1])
        assert np.isnan(per_q[2])

    def test_with_weights(self):
        """Test EMD with custom weights."""
        v_a = [1, 1, 1]
        v_b = [5, 5, 5]
        weights = [2.0, 1.0, 1.0]  # q1 has double weight
        agg_emd, per_q = emd_between_vectors(
            self.questions,
            v_a,
            v_b,
            self.question_metadata,
            vector_mode="sequence",
            weights=weights,
        )
        # All per_q should be 1.0
        assert np.allclose(per_q, [1.0, 1.0, 1.0])
        # With weights [2,1,1] normalized to [0.5, 0.25, 0.25], agg = 1.0
        assert np.isclose(agg_emd, 1.0)

    def test_without_normalization(self):
        """Test EMD without range normalization."""
        v_a = [1, 1, 1]
        v_b = [5, 5, 5]
        agg_emd, per_q = emd_between_vectors(
            self.questions,
            v_a,
            v_b,
            self.question_metadata,
            vector_mode="sequence",
            normalize_range=False,
        )
        # Without normalization, each distance is |5-1| = 4
        assert np.allclose(per_q, [4.0, 4.0, 4.0])
        assert np.isclose(agg_emd, 4.0)

    def test_no_valid_answers(self):
        """Test when no questions have valid answers in both vectors."""
        v_a = {"q1": float("nan")}
        v_b = {"q1": float("nan")}
        agg_emd, per_q = emd_between_vectors(
            ["q1"],
            v_a,
            v_b,
            {"q1": self.question_metadata["q1"]},
            vector_mode="mapping",
        )
        # Should return 0.0 when no comparable questions
        assert agg_emd == 0.0
        assert np.isnan(per_q[0])

    def test_invalid_vector_mode(self):
        """Test error with invalid vector_mode."""
        v_a = {"q1": 3}
        v_b = {"q1": 3}
        with pytest.raises(ValueError, match="vector_mode must be"):
            emd_between_vectors(
                ["q1"],
                v_a,
                v_b,
                {"q1": self.question_metadata["q1"]},
                vector_mode="invalid",
            )

    def test_sequence_mode_length_mismatch(self):
        """Test error when sequence length doesn't match questions length."""
        v_a = [1, 2]  # Only 2 elements
        v_b = [3, 4]
        with pytest.raises(ValueError, match="must match len\\(questions\\)"):
            emd_between_vectors(
                self.questions,  # 3 questions
                v_a,
                v_b,
                self.question_metadata,
                vector_mode="sequence",
            )

    def test_sequence_mode_with_dict_raises_error(self):
        """Test error when using sequence mode with dict inputs."""
        v_a = {"q1": 3, "q2": 4, "q3": 2}
        v_b = {"q1": 3, "q2": 4, "q3": 2}
        with pytest.raises(TypeError, match="v_a and v_b must be sequences"):
            emd_between_vectors(self.questions, v_a, v_b, self.question_metadata, vector_mode="sequence")

    def test_weights_length_mismatch(self):
        """Test error when weights length doesn't match questions length."""
        v_a = [1, 2, 3]
        v_b = [3, 4, 2]
        weights = [1.0, 1.0]  # Only 2 weights for 3 questions
        with pytest.raises(ValueError, match="weights must have the same length"):
            emd_between_vectors(
                self.questions,
                v_a,
                v_b,
                self.question_metadata,
                vector_mode="sequence",
                weights=weights,
            )

    def test_all_zero_weights(self):
        """Test handling of all-zero weights."""
        v_a = [1, 2, 3]
        v_b = [5, 4, 1]
        weights = [0.0, 0.0, 0.0]
        agg_emd, per_q = emd_between_vectors(
            self.questions,
            v_a,
            v_b,
            self.question_metadata,
            vector_mode="sequence",
            weights=weights,
        )
        # Should fall back to unweighted mean
        assert np.isfinite(agg_emd)

    def test_partial_matching_questions(self):
        """Test with only some questions present in both vectors."""
        v_a = {"q1": 1, "q3": 3}
        v_b = {"q1": 5, "q2": 4, "q3": 3}
        agg_emd, per_q = emd_between_vectors(self.questions, v_a, v_b, self.question_metadata, vector_mode="mapping")
        # q1: different, q2: missing, q3: same
        assert per_q[0] > 0
        assert np.isnan(per_q[1])
        assert np.isclose(per_q[2], 0.0)
