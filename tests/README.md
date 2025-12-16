# Unit Tests

This directory contains unit tests for the `llm_behavior_adaptation` package.

## Test Structure

- `test_utils.py` - Tests for logging utilities (`llm_behavior_adaptation.utils`)
- `test_formulas.py` - Tests for mathematical formulas and computations (`value_measurement.formulas`)
- `test_distances.py` - Tests for distance calculation functions (`value_measurement.distances`)
- `test_measurement_utils.py` - Tests for measurement utilities including country mapping (`value_measurement.measurement_utils`)

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=llm_behavior_adaptation --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_utils.py
```

### Run specific test class
```bash
pytest tests/test_formulas.py::TestJensenShannonDivergence
```

### Run specific test method
```bash
pytest tests/test_formulas.py::TestJensenShannonDivergence::test_identical_distributions
```

### Run tests with verbose output
```bash
pytest -v
```

### Run tests and show print statements
```bash
pytest -s
```

## Test Coverage

The test suite covers:

- Logger functionality and formatting
- Statistical measures (Jensen-Shannon Divergence, Hellinger Distance)
- Distance calculations (EMD, medoid computation)
- Probability distribution operations (softmax, filtering, centroids)
- Country/continent mapping utilities
- Edge cases (NaN handling, empty inputs, invalid ranges)

## Requirements

Tests require:
- pytest >= 6.0
- numpy
- scipy
- pandas
- pycountry
- pycountry_convert

Install test dependencies:
```bash
pip install -e ".[dev]"
```
