# Evaluating LLM Adaptation to Sociodemographic Factors

Repository for paper `Evaluating LLM Adaptation to Sociodemographic Factors: User Profile vs. Dialogue History`.


## Directory Structure

```bash
├── LICENSE
├── Makefile
├── README.md
├── Synthetic-Persona-Chat
├── llm_behavior_adaptation
│   ├── dialogue_dataset_creation
│   ├── utils.py
│   └── value_measurement
├── requirements.txt
├── scripts
├── setup.cfg
├── setup.py
├── understanding
├── values_results
```

## Dataset Generation

The 2 datasets consisting of 1000 generated dialogues each [datasets](https://github.com/FerdinandZhong/llm_behavior_adaptation/datasets/wvs_generated_dialogues) is open for usage.

Dataset is generated through a multi-agent mechanism based on the [seed dataset](https://github.com/FerdinandZhong/llm_behavior_adaptation/datasets/wvs_benchmarks/sampled_demographic_features.csv)

![Figure: Dataset Generation](https://github.com/FerdinandZhong/llm_behavior_adaptation/blob/main/images/DataGen.png)


Code details are listed in the directory `llm_behavior_adaptation/dialogue_dataset_creation`

## Behavior Adaptation Evaluation

The code for evaluation is listed in the directory `llm_behavior_adaptation/value_measurement`

* Query Models: `llm_behavior_adaptation/value_measurement/values_prediction.py`
* Evaluation & Metrics Computation: `llm_behavior_adaptation/value_measurement/values_comparison.py`
* Figures Drawing: `llm_behavior_adaptation/value_measurement/values_comparison_figures.py`

## Development

### Environment Setup

```bash
# Create conda environment
conda create -n llm_behavior_test python=3.10 -y
conda activate llm_behavior_test

# Install package in development mode
pip install -e .

# Install development dependencies
pip install pre-commit pytest pytest-cov
```

### Pre-commit Hooks

This project uses pre-commit hooks for automated code quality checks:

```bash
# Install hooks (first time only)
pre-commit install

# Or use Makefile
make pre-commit-install

# Hooks will now run automatically on commit
git commit -m "Your message"
```

**Quick Reference:**
- 📖 [Pre-commit Guide](PRE_COMMIT_GUIDE.md) - Detailed documentation
- 📋 [Setup Summary](PRE_COMMIT_SETUP_SUMMARY.md) - Installation overview
- 🚀 [Quick Reference](.pre-commit-quickref.md) - Common commands

**What's checked:**
- Code formatting (Black)
- Import sorting (isort)
- Code quality (flake8)
- Security issues (bandit)
- Common issues (trailing whitespace, file endings, etc.)

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llm_behavior_adaptation --cov-report=html

# Or use Makefile
make test
```

See [tests/README.md](tests/README.md) for more details on testing.

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Run all checks (lint + pre-commit + tests)
make check
```
