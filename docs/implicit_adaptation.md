# Implicit Adaptation Experiment

This experiment tests whether language models adapt their value predictions when given implicit feedback from users.

## Overview

The implicit adaptation experiment follows this workflow:

1. **Initial Prediction**: The model makes a value prediction based on a user's demographic profile
2. **Implicit Feedback**: The model receives one of two types of implicit feedback:
   - **Implicit Feedback** (`implicit_feedback`): User expresses uncertainty ("I'm not entirely sure that answer fits me")
   - **Willing to Change** (`willing_to_change`): User suggests an alternative option might fit better
3. **Adapted Prediction**: The model reconsiders and makes a new prediction
4. **Analysis**: Compare initial vs. adapted predictions to measure adaptation behavior

## Files

### Main Script
- `llm_behavior_adaptation/value_measurement/wvs_implicit_adaptation.py` - Main experiment controller

### Prompt Templates
- `llm_behavior_adaptation/value_measurement/prompts/direct_question.json` - Initial prediction prompt
- `llm_behavior_adaptation/value_measurement/prompts/implicit_feedback.json` - Uncertainty feedback prompt
- `llm_behavior_adaptation/value_measurement/prompts/willing_to_change.json` - Alternative suggestion prompt

### Configuration
- `configs/implicit_adaptation_example.yaml` - Example configuration file

## Usage

### 1. Configure the Experiment

Create or edit a YAML configuration file (see `configs/implicit_adaptation_example.yaml`):

```yaml
# Dataset paths
user_profile_dataset_path: "datasets/wvs_benchmarks/sampled_demographic_features.csv"
picked_questions_path: "datasets/wvs_benchmarks/picked_questions.json"

# Output path
output_file_path: "results/implicit_adaptation/gpt4_implicit_feedback.jsonl"

# Prompts folder
prompts_folder: "llm_behavior_adaptation/value_measurement/prompts"

# Model configuration
evaluated_model: "gpt-4"
llm_server: "llm_platform"

# Feedback type
feedback_type: "implicit_feedback"  # or "willing_to_change"

# Optional settings
starting_row: 0
ending_row: 10  # Test with 10 users first
storage_step: 5
verbose: 1
```

### 2. Set Environment Variables

```bash
export api_key="your-api-key"
export base_url="http://localhost:8000/v1"  # If using local model server
```

### 3. Run the Experiment

```bash
python -m llm_behavior_adaptation.value_measurement.wvs_implicit_adaptation \
    --config configs/your_config.yaml
```

## Output Format

The script outputs JSONL files where each line contains results for one user:

```json
{
  "user_id": {
    "initial": {
      "category_1": [
        {"question_id_1": {"option_id": 5, "reason": "...", "confidence": "high"}},
        {"question_id_2": {"option_id": 3, "reason": "...", "confidence": "medium"}}
      ]
    },
    "adapted": {
      "category_1": [
        {"question_id_1": {"option_id": 4, "reason": "...", "confidence": "high"}},
        {"question_id_2": {"option_id": 3, "reason": "...", "confidence": "high"}}
      ]
    }
  }
}
```

## Feedback Types

### Implicit Feedback
- **Prompt**: "Hmm, I'm not entirely sure that answer fits me. Could you think about it again based on what we've discussed?"
- **Purpose**: Tests if the model reconsiders without explicit direction
- **Expected Behavior**: Model may adjust confidence or reconsider its reasoning

### Willing to Change
- **Prompt**: "Hmm, I'm not entirely sure about that. When I think about it, option X feels more aligned with how I actually see things. What do you think?"
- **Purpose**: Tests if the model adapts when user suggests an alternative
- **Expected Behavior**: Model may shift toward the suggested option or explain why it maintains its original choice

## Analysis

To analyze the results, you can:

1. **Adaptation Rate**: Calculate how often the model changes its prediction
2. **Direction of Change**: When using "willing_to_change", measure how often the model shifts toward the suggested option
3. **Confidence Changes**: Compare confidence levels between initial and adapted predictions
4. **Alignment Impact**: Measure whether adapted predictions are closer to ground truth values

Example analysis script structure:

```python
import json

def analyze_adaptation(jsonl_path):
    changes = 0
    total = 0

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            for user_id, results in data.items():
                initial = results["initial"]
                adapted = results["adapted"]

                for category in initial:
                    for i, init_q in enumerate(initial[category]):
                        adapt_q = adapted[category][i]

                        init_id = list(init_q.values())[0]["option_id"]
                        adapt_id = list(adapt_q.values())[0]["option_id"]

                        total += 1
                        if init_id != adapt_id:
                            changes += 1

    print(f"Adaptation rate: {changes/total:.2%}")

analyze_adaptation("results/implicit_adaptation/output.jsonl")
```

## Related Scripts

- `wvs_values_prediction.py` - Standard value prediction (no adaptation)
- `wvs_values_comparison.py` - Compare model predictions to human values
- `wvs_willing_to_change.py` - Analysis script for willing-to-change experiments
