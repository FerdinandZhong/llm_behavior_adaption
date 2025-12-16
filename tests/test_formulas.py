"""Unit tests for llm_behavior_adaptation.value_measurement.formulas module."""

import numpy as np
import pytest

from llm_behavior_adaptation.value_measurement.formulas import (
    componentwise_centroid,
    componentwise_centroid_vsm,
    compute_emd,
    compute_js_centroid,
    compute_js_centroid_and_avg,
    emd_distance,
    emd_distance_vsm,
    emd_medoid_skip_nan,
    filter_rows,
    hellinger_distance,
    jensen_shannon_divergence,
    softmax,
)


class TestJensenShannonDivergence:
    """Test jensen_shannon_divergence function."""

    def test_identical_distributions(self):
        """Test JSD between identical distributions is 0."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        jsd = jensen_shannon_divergence(p, q)
        assert np.isclose(jsd, 0.0, atol=1e-10)

    def test_different_distributions(self):
        """Test JSD between different distributions is positive."""
        p = np.array([0.5, 0.5, 0.0, 0.0])
        q = np.array([0.0, 0.0, 0.5, 0.5])
        jsd = jensen_shannon_divergence(p, q)
        assert jsd > 0
        assert jsd <= 1.0  # JSD with base 2 is bounded by 1

    def test_invalid_distribution_p(self):
        """Test with invalid probability distribution p (doesn't sum to 1)."""
        p = np.array([0.3, 0.3, 0.3])  # Sums to 0.9
        q = np.array([0.5, 0.3, 0.2])
        jsd = jensen_shannon_divergence(p, q)
        assert jsd == 1  # Returns 1 for invalid distributions

    def test_invalid_distribution_q(self):
        """Test with invalid probability distribution q (doesn't sum to 1)."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.3, 0.3, 0.3])  # Sums to 0.9
        jsd = jensen_shannon_divergence(p, q)
        assert jsd == 1

    def test_different_base(self):
        """Test JSD with different logarithm base."""
        p = np.array([0.5, 0.5])
        q = np.array([0.3, 0.7])
        jsd_base2 = jensen_shannon_divergence(p, q, base=2)
        jsd_base_e = jensen_shannon_divergence(p, q, base=np.e)
        # Different bases should give different results
        assert not np.isclose(jsd_base2, jsd_base_e)

    def test_symmetry(self):
        """Test that JSD is symmetric."""
        p = np.array([0.6, 0.4])
        q = np.array([0.3, 0.7])
        jsd_pq = jensen_shannon_divergence(p, q)
        jsd_qp = jensen_shannon_divergence(q, p)
        assert np.isclose(jsd_pq, jsd_qp)


class TestHellingerDistance:
    """Test hellinger_distance function."""

    def test_identical_distributions(self):
        """Test Hellinger distance between identical distributions is 0."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])
        hd = hellinger_distance(p, q)
        assert np.isclose(hd, 0.0, atol=1e-10)

    def test_orthogonal_distributions(self):
        """Test Hellinger distance for completely different distributions."""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 1.0, 0.0])
        hd = hellinger_distance(p, q)
        assert np.isclose(hd, 1.0, atol=1e-10)

    def test_invalid_distribution_p(self):
        """Test with invalid probability distribution p."""
        p = np.array([0.3, 0.3, 0.3])
        q = np.array([0.5, 0.3, 0.2])
        hd = hellinger_distance(p, q)
        assert hd == 1

    def test_invalid_distribution_q(self):
        """Test with invalid probability distribution q."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.3, 0.3, 0.3])
        hd = hellinger_distance(p, q)
        assert hd == 1

    def test_symmetry(self):
        """Test that Hellinger distance is symmetric."""
        p = np.array([0.6, 0.4])
        q = np.array([0.3, 0.7])
        hd_pq = hellinger_distance(p, q)
        hd_qp = hellinger_distance(q, p)
        assert np.isclose(hd_pq, hd_qp)

    def test_bounded(self):
        """Test that Hellinger distance is bounded between 0 and 1."""
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.1, 0.3, 0.6])
        hd = hellinger_distance(p, q)
        assert 0 <= hd <= 1


class TestSoftmax:
    """Test softmax function."""

    def test_softmax_output_sums_to_one(self):
        """Test that softmax output sums to 1."""
        z = np.array([1.0, 2.0, 3.0])
        result = softmax(z)
        assert np.isclose(np.sum(result), 1.0)

    def test_softmax_all_positive(self):
        """Test that all softmax outputs are positive."""
        z = np.array([1.0, 2.0, 3.0])
        result = softmax(z)
        assert np.all(result > 0)

    def test_softmax_monotonic(self):
        """Test that softmax preserves order."""
        z = np.array([1.0, 2.0, 3.0])
        result = softmax(z)
        assert result[0] < result[1] < result[2]

    def test_softmax_numerical_stability(self):
        """Test numerical stability with large values."""
        z = np.array([1000, 1001, 1002])
        result = softmax(z)
        assert np.isfinite(result).all()
        assert np.isclose(np.sum(result), 1.0)


class TestFilterRows:
    """Test filter_rows function."""

    def test_all_valid_rows(self):
        """Test filtering with all valid rows."""
        arr = np.array([[0.25, 0.25, 0.25, 0.25], [0.5, 0.3, 0.2, 0.0]])
        filtered, removed = filter_rows(arr)
        assert filtered.shape[0] == 2
        assert removed == 0

    def test_some_invalid_rows(self):
        """Test filtering with some invalid rows."""
        arr = np.array([[0.25, 0.25, 0.25, 0.25], [0.5, 0.3, 0.2, 0.1]])  # Invalid
        filtered, removed = filter_rows(arr)
        assert filtered.shape[0] == 1
        assert removed == 1

    def test_all_invalid_rows(self):
        """Test filtering with all invalid rows."""
        arr = np.array([[0.3, 0.3, 0.3], [0.4, 0.4, 0.1]])
        filtered, removed = filter_rows(arr)
        assert filtered.shape[0] == 0
        assert removed == 2

    def test_custom_tolerance(self):
        """Test filtering with custom tolerance."""
        arr = np.array([[0.25, 0.25, 0.25, 0.24999999]])
        filtered, removed = filter_rows(arr, rtol=1e-5, atol=1e-8)
        assert filtered.shape[0] == 1
        assert removed == 0


class TestComputeJSCentroid:
    """Test compute_js_centroid function."""

    def test_single_distribution(self):
        """Test centroid of a single distribution is itself."""
        distributions = np.array([[0.25, 0.25, 0.25, 0.25]])
        centroid, failed = compute_js_centroid(distributions)
        assert np.allclose(centroid, distributions[0], atol=1e-3)
        assert failed == 0

    def test_identical_distributions(self):
        """Test centroid of identical distributions."""
        distributions = np.array([[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
        centroid, failed = compute_js_centroid(distributions)
        assert np.allclose(centroid, [0.25, 0.25, 0.25, 0.25], atol=1e-3)
        assert failed == 0

    def test_different_distributions(self):
        """Test centroid of different distributions."""
        distributions = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
        centroid, failed = compute_js_centroid(distributions)
        assert np.isclose(np.sum(centroid), 1.0)
        assert failed == 0

    def test_with_invalid_distributions(self):
        """Test with some invalid distributions."""
        distributions = np.array([[0.25, 0.25, 0.25, 0.25], [0.3, 0.3, 0.3, 0.0]])  # Invalid
        centroid, failed = compute_js_centroid(distributions)
        assert np.isclose(np.sum(centroid), 1.0)
        assert failed == 1


class TestComputeJSCentroidAndAvg:
    """Test compute_js_centroid_and_avg function."""

    def test_returns_three_values(self):
        """Test that function returns centroid, failed count, and avg divergence."""
        distributions = np.array([[0.25, 0.25, 0.25, 0.25], [0.3, 0.2, 0.3, 0.2]])
        centroid, failed, avg_div = compute_js_centroid_and_avg(distributions)
        assert len(centroid) == 4
        assert isinstance(failed, (int, np.integer))
        assert isinstance(avg_div, (float, np.floating))

    def test_avg_divergence_non_negative(self):
        """Test that average divergence is non-negative."""
        distributions = np.array([[0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]])
        _, _, avg_div = compute_js_centroid_and_avg(distributions)
        assert avg_div >= 0


class TestComputeEMD:
    """Test compute_emd function."""

    def test_identical_vectors(self):
        """Test EMD between identical vectors is 0."""
        a = np.array([1, 2, 3, 4])
        b = np.array([1, 2, 3, 4])
        emd = compute_emd(a, b)
        assert np.isclose(emd, 0.0, atol=1e-10)

    def test_different_vectors(self):
        """Test EMD between different vectors is positive."""
        a = np.array([1, 1, 1, 1])
        b = np.array([4, 4, 4, 4])
        emd = compute_emd(a, b)
        assert emd > 0

    def test_emd_range(self):
        """Test EMD is within valid range."""
        a = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        b = np.array([4, 3, 2, 1, 4, 3, 2, 1])
        emd = compute_emd(a, b)
        assert 0 <= emd <= 4  # Maximum possible EMD for bins 1-5


class TestEMDDistance:
    """Test emd_distance function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.question_metadata = {
            "q1": {"answer_scale_min": 1, "answer_scale_max": 5},
            "q2": {"answer_scale_min": 1, "answer_scale_max": 5},
            "q3": {"answer_scale_min": 1, "answer_scale_max": 5},
        }

    def test_identical_answers(self):
        """Test EMD between identical answer vectors is 0."""
        a = {"q1": 3, "q2": 4, "q3": 2}
        b = {"q1": 3, "q2": 4, "q3": 2}
        dist = emd_distance(a, b, self.question_metadata)
        assert np.isclose(dist, 0.0)

    def test_different_answers(self):
        """Test EMD between different answer vectors is positive."""
        a = {"q1": 1, "q2": 1, "q3": 1}
        b = {"q1": 5, "q2": 5, "q3": 5}
        dist = emd_distance(a, b, self.question_metadata)
        assert dist > 0

    def test_normalized_distance(self):
        """Test normalized distance is bounded."""
        a = {"q1": 1, "q2": 2, "q3": 3}
        b = {"q1": 5, "q2": 4, "q3": 1}
        dist = emd_distance(a, b, self.question_metadata, normalize=True)
        assert dist >= 0

    def test_missing_keys(self):
        """Test handling of missing keys."""
        a = {"q1": 3, "q2": 4}
        b = {"q1": 3, "q2": 5, "q3": 2}
        dist = emd_distance(a, b, self.question_metadata)
        # Should only compute distance for shared keys
        assert dist >= 0

    def test_nan_values(self):
        """Test handling of NaN values."""
        a = {"q1": 3, "q2": float("nan"), "q3": 2}
        b = {"q1": 3, "q2": 4, "q3": 2}
        dist = emd_distance(a, b, self.question_metadata)
        # Should skip NaN values
        assert np.isfinite(dist)

    def test_out_of_range_skip(self):
        """Test skipping out of range values."""
        a = {"q1": 0, "q2": 3, "q3": 2}  # q1 out of range
        b = {"q1": 3, "q2": 3, "q3": 2}
        dist = emd_distance(a, b, self.question_metadata, skip_out_of_range=True, normalize=True)
        # q1 should be skipped, only q2 and q3 contribute
        assert dist >= 0


class TestEMDDistanceVSM:
    """Test emd_distance_vsm function."""

    def test_identical_vectors(self):
        """Test EMD between identical vectors is 0."""
        a = [1, 2, 3, 4, 5]
        b = [1, 2, 3, 4, 5]
        dist = emd_distance_vsm(a, b)
        assert np.isclose(dist, 0.0)

    def test_different_vectors(self):
        """Test EMD between different vectors is positive."""
        a = [1, 1, 1, 1, 1]
        b = [5, 5, 5, 5, 5]
        dist = emd_distance_vsm(a, b)
        assert dist > 0

    def test_normalized_output(self):
        """Test that normalized output is bounded."""
        a = [1, 2, 3, 4, 5]
        b = [5, 4, 3, 2, 1]
        dist = emd_distance_vsm(a, b, normalize=True)
        assert dist >= 0


class TestComponentwiseCentroid:
    """Test componentwise_centroid function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.question_metadata = {
            "q1": {"answer_scale_min": 1, "answer_scale_max": 5},
            "q2": {"answer_scale_min": 1, "answer_scale_max": 5},
            "q3": {"answer_scale_min": 1, "answer_scale_max": 5},
        }

    def test_single_vector(self):
        """Test centroid of single vector is itself."""
        vectors = [{"q1": 3, "q2": 4, "q3": 2}]
        centroid = componentwise_centroid(vectors, self.question_metadata)
        assert centroid == {"q1": 3, "q2": 4, "q3": 2}

    def test_odd_number_of_vectors(self):
        """Test centroid with odd number of vectors (median is middle value)."""
        vectors = [
            {"q1": 1, "q2": 2, "q3": 3},
            {"q1": 3, "q2": 4, "q3": 5},
            {"q1": 5, "q2": 2, "q3": 1},
        ]
        centroid = componentwise_centroid(vectors, self.question_metadata)
        assert centroid["q1"] == 3  # median of [1, 3, 5]
        assert centroid["q2"] == 2  # median of [2, 4, 2] = [2, 2, 4]
        assert centroid["q3"] == 3  # median of [3, 5, 1] = [1, 3, 5]

    def test_even_number_of_vectors(self):
        """Test centroid with even number of vectors (median is average of two middle values)."""
        vectors = [
            {"q1": 1, "q2": 2},
            {"q1": 3, "q2": 4},
            {"q1": 5, "q2": 2},
            {"q1": 2, "q2": 5},
        ]
        centroid = componentwise_centroid(vectors, self.question_metadata)
        # q1: [1, 2, 3, 5] -> median = (2+3)/2 = 2.5 -> round(2.5) = 2
        # q2: [2, 2, 4, 5] -> median = (2+4)/2 = 3
        assert centroid["q1"] in [2, 3]  # Could be 2 or 3 depending on rounding
        assert centroid["q2"] == 3

    def test_with_nan_values(self):
        """Test handling of NaN values."""
        vectors = [
            {"q1": 1, "q2": 2, "q3": 3},
            {"q1": 3, "q2": float("nan"), "q3": 5},
            {"q1": 5, "q2": 4, "q3": 1},
        ]
        centroid = componentwise_centroid(vectors, self.question_metadata)
        # q2 should skip the NaN value
        assert "q1" in centroid
        assert "q2" in centroid
        assert "q3" in centroid

    def test_clamping_to_range(self):
        """Test that values are clamped to valid range."""
        vectors = [
            {"q1": 6, "q2": 2},  # q1 out of range (max is 5)
            {"q1": 0, "q2": 4},  # q1 out of range (min is 1)
        ]
        centroid = componentwise_centroid(vectors, self.question_metadata)
        # After clamping: q1 = [5, 1] -> median = 3
        assert 1 <= centroid["q1"] <= 5
        assert 1 <= centroid["q2"] <= 5


class TestComponentwiseCentroidVSM:
    """Test componentwise_centroid_vsm function."""

    def test_single_vector(self):
        """Test centroid of single vector is itself."""
        vectors = [[3, 4, 2, 1, 5]]
        centroid = componentwise_centroid_vsm(vectors)
        assert centroid == [3, 4, 2, 1, 5]

    def test_multiple_vectors(self):
        """Test centroid of multiple vectors."""
        vectors = [[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [3, 3, 3, 3, 3]]
        centroid = componentwise_centroid_vsm(vectors)
        # Each position should be median: [1,5,3], [2,4,3], [3,3,3], [4,2,3], [5,1,3]
        assert centroid == [3, 3, 3, 3, 3]

    def test_clamping(self):
        """Test that values are within valid range."""
        vectors = [[1, 1, 1], [5, 5, 5], [3, 3, 3]]
        centroid = componentwise_centroid_vsm(vectors)
        for val in centroid:
            assert 1 <= val <= 5


class TestEMDMedoidSkipNan:
    """Test emd_medoid_skip_nan function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.question_metadata = {
            "q1": {"answer_scale_min": 1, "answer_scale_max": 5},
            "q2": {"answer_scale_min": 1, "answer_scale_max": 5},
        }

    def test_single_vector(self):
        """Test medoid of single vector is itself."""
        vectors = [{"q1": 3, "q2": 4}]
        medoid, dist, excluded, indices = emd_medoid_skip_nan(vectors, self.question_metadata)
        assert medoid == {"q1": 3, "q2": 4}
        assert dist == 0.0
        assert excluded == 0
        assert indices == []

    def test_multiple_vectors_no_nan(self):
        """Test medoid with multiple valid vectors."""
        vectors = [{"q1": 1, "q2": 1}, {"q1": 3, "q2": 3}, {"q1": 5, "q2": 5}]
        medoid, dist, excluded, indices = emd_medoid_skip_nan(vectors, self.question_metadata)
        # The middle vector should be the medoid
        assert medoid == {"q1": 3, "q2": 3}
        assert dist >= 0
        assert excluded == 0

    def test_with_nan_vectors(self):
        """Test medoid excluding vectors with NaN."""
        vectors = [
            {"q1": 1, "q2": 1},
            {"q1": float("nan"), "q2": 3},  # Should be excluded
            {"q1": 5, "q2": 5},
        ]
        medoid, dist, excluded, indices = emd_medoid_skip_nan(vectors, self.question_metadata)
        assert excluded == 1
        assert 1 in indices
        # Medoid should be one of the valid vectors
        assert medoid in [vectors[0], vectors[2]]

    def test_all_nan_vectors(self):
        """Test error when all vectors have NaN."""
        vectors = [
            {"q1": float("nan"), "q2": 1},
            {"q1": 1, "q2": float("nan")},
        ]
        with pytest.raises(ValueError, match="all vectors contain NaN"):
            emd_medoid_skip_nan(vectors, self.question_metadata)

    def test_empty_vectors(self):
        """Test error with empty vectors."""
        vectors = []
        with pytest.raises(ValueError, match="empty"):
            emd_medoid_skip_nan(vectors, self.question_metadata)
