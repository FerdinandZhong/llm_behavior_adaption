# Gap Analysis Script

This script analyzes gaps between BA (Behavior Adaptation) dialogue results and actual human values, then tests if the model can adapt when given soft user feedback.

## Quick Start

**1. Set API key**:

```bash
export api_key="your-api-key"
```

**2. Run test with 10 users**:

```bash
python llm_behavior_adaptation/value_measurement/wvs_gap_analysis.py \
    --config wvs_values_gap_analysis/gap_analysis_test_config.yaml
```

**3. Check results**:

- Results: `wvs_values_gap_analysis/qwen3_30b_a3b_career_gap_analysis_test.jsonl`
- Statistics: `wvs_values_gap_analysis/qwen3_30b_a3b_career_gap_analysis_test_statistics.json`

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
  - [What This Tests](#what-this-tests)
- [Files](#files)
- [Usage](#usage)
  - [Basic Usage](#1-basic-usage)
  - [Configuration File](#2-configuration-file)
  - [Testing with Data Slices](#3-testing-with-data-slices-experimental)
  - [Environment Variables](#4-environment-variables)
  - [Reasoning Model Configuration](#5-reasoning-model-configuration)
  - [Running the Script](#6-running-the-script)
- [Input File Formats](#input-file-formats)
- [Output Format](#output-format)
- [Interpreting Results](#interpreting-results)
- [Features](#features)
- [Examples](#examples)
- [Prompt Engineering](#prompt-engineering)
- [Error Handling](#error-handling)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

## Overview

The script performs the following steps:

1. **Load BA dialogue results**: Reads predictions from the BA dialogue system
2. **Load human results**: Reads actual human value responses from WVS dataset
3. **Load dialogue history**: Retrieves the conversation context for each user
4. **Match users**: For each user in BA results, finds corresponding human data
5. **Detect gaps**: For each question, compares predicted vs actual option_id
6. **Test adaptation**: When gaps exist, presents the model with:
   - The full dialogue context
   - The model's original answer (with reasoning)
   - A soft hint from the user suggesting they prefer the human's choice
7. **Measure adaptation**: Tracks whether the model changes to match the human response or maintains its original position

### What This Tests

This script measures **behavior adaptation capability** through a natural interaction pattern:

1. **Soft Feedback**: Instead of forcing change, the user gives a gentle hint ("Hmm, I'm not entirely sure about that...")
2. **Context Integration**: The model sees the full conversation history to make an informed decision
3. **Genuine Adaptation**: The model can either:
   - **Adapt**: Change to align with the user's preference (high flexibility)
   - **Maintain**: Stick with its original reasoning (high consistency)

**Key Metrics**:

- **Adaptation Rate**: % of gaps where the model changed to match human response
- **Remaining Gap Rate**: % of gaps where the model maintained its original position
- **Accuracy Improvement**: How much accuracy increases after giving the model a chance to adapt

## Files

- **`wvs_gap_analysis.py`**: Main script for gap analysis
- **`prompts/gap_rationale.json`**: Prompt template for generating gap rationales
- **`configs/gap_analysis_config.yaml`**: Example configuration file for full dataset
- **`wvs_values_gap_analysis/gap_analysis_test_config.yaml`**: Test configuration for small data slices

## Usage

### 1. Basic Usage

```bash
python llm_behavior_adaptation/value_measurement/wvs_gap_analysis.py \
    --config configs/gap_analysis_config.yaml
```

### 2. Configuration File

Create a YAML config file with the following structure:

```yaml
# Model configuration
evaluated_model: "gpt-4o"
llm_server: "llm_platform"
reasoning: false

# Input paths
ba_dialogue_results_path: "results/ba_dialogue_career_results.jsonl"
human_results_path: "datasets/wvs_benchmarks/sampled_values_df.csv"
generated_dialogues_path: "results/generated_dialogues_career.jsonl"
picked_questions_path: "datasets/wvs_benchmarks/picked_questions.json"
prompts_folder: "llm_behavior_adaptation/value_measurement/prompts"

# Output
output_file_path: "results/gap_analysis_results.jsonl"

# Optional
storage_step: 10
verbose: 1

# Experimental (for testing/debugging)
starting_row: null  # Process subset of data
ending_row: null    # null or -1 for all
```

### 3. Testing with Data Slices (Experimental)

For testing or debugging, you can process only a subset of the data:

```yaml
# Process only first 10 users
starting_row: 0
ending_row: 10

# Process users 10-19
starting_row: 10
ending_row: 20

# Process from user 5 to end
starting_row: 5
ending_row: -1  # or null
```

This is useful for:

- Quick testing with small datasets
- Debugging specific users
- Incremental processing

### 4. Environment Variables

Set the following environment variables:

```bash
# For OpenAI models (GPT-4o, etc.)
export api_key="your-openai-api-key"

# For OpenRouter models (Qwen, etc.)
export api_key="your-openrouter-api-key"

# For local models:
export api_key="dummy-key"
export base_url="http://localhost:8000/v1"
```

### 5. Reasoning Model Configuration

For reasoning models (e.g., Qwen3-30B-A3B-Instruct), you need to:

1. Set `reasoning: true` in the config
2. Provide `model_base_url` (e.g., OpenRouter URL)
3. Configure `extra_body` with reasoning parameters:

```yaml
reasoning: true
model_base_url: "https://openrouter.ai/api/v1"
extra_body:
  reasoning:
    effort: "low"  # Options: "low", "medium", "high"
  provider:
    only:
      - chutes  # Specify preferred provider
```

**Reasoning Effort Levels**:

- **low**: Faster responses, less thorough reasoning
- **medium**: Balanced speed and reasoning depth
- **high**: Most thorough reasoning, slower responses

### 6. Running the Script

**Quick test (10 users)**:

```bash
python llm_behavior_adaptation/value_measurement/wvs_gap_analysis.py \
    --config wvs_values_gap_analysis/gap_analysis_test_config.yaml
```

**Full dataset**:

```bash
python llm_behavior_adaptation/value_measurement/wvs_gap_analysis.py \
    --config configs/gap_analysis_config.yaml
```

## Input File Formats

### BA Dialogue Results (JSONL)

Each line contains predictions for one user:

```json
{
  "user_id_123": {
    "Social Values, Norms, Stereotypes": [
      {
        "Q2": {
          "option_id": 2,
          "reason": "Reason for selection..."
        }
      }
    ]
  }
}
```

### Human Results (CSV)

CSV file with columns:
- `D_INTERVIEW`: User ID
- `Q1`, `Q2`, ...: Question IDs with selected option_id values

### Generated Dialogues (JSONL)

Each line contains dialogue history for one user:

```json
{
  "user_id_123": [
    {
      "role": "user",
      "content": "I need career advice..."
    },
    {
      "role": "chatbot",
      "content": "I'd be happy to help..."
    }
  ]
}
```

## Output Format

### Gap Analysis Results (JSONL)

The script outputs a JSONL file where each line contains gap analysis for one user:

```json
{
  "user_id_123": {
    "Q2": {
      "predicted_option_id": 2,
      "human_option_id": 3,
      "gap_rationale": {
        "option_id": 3,
        "reason": "You're absolutely right. After considering your feedback and our conversation context, option 3 better represents your values..."
      }
    },
    "Q5": {
      "predicted_option_id": 1,
      "human_option_id": 4,
      "gap_rationale": {
        "option_id": 1,
        "reason": "I understand your concern, but based on our earlier discussion about your work priorities, I still believe option 1 is most aligned..."
      }
    }
  }
}
```

**Note**: The `gap_rationale.option_id` may differ from `human_option_id` if the model decides not to adapt based on the user's feedback.

### Statistics File (JSON)

The script also generates a statistics file (`*_statistics.json`) with adaptation metrics:

```json
{
  "summary": {
    "total_questions": 1500,
    "original_gaps": 450,
    "adapted_gaps": 320,
    "remaining_gaps": 130,
    "adaptation_rate_percent": 71.11,
    "remaining_gap_rate_percent": 28.89,
    "original_accuracy_percent": 70.00,
    "post_adaptation_accuracy_percent": 91.33
  }
}
```

**Metrics Explained**:

- **total_questions**: Total number of questions compared
- **original_gaps**: Number of questions where BA prediction ≠ human response
- **adapted_gaps**: Number of gaps where model changed to match human (successful adaptation)
- **remaining_gaps**: Number of gaps where model maintained its original answer
- **adaptation_rate_percent**: Percentage of gaps that were adapted
- **original_accuracy_percent**: BA model's original accuracy
- **post_adaptation_accuracy_percent**: Accuracy after giving model opportunity to adapt

## Interpreting Results

### Adaptation Behavior

The statistics reveal how well the model can adapt to user feedback:

- **High adaptation rate (>70%)**: Model is good at listening to user feedback and adjusting
- **Low adaptation rate (<30%)**: Model tends to maintain its original position even when user hints at disagreement
- **Mixed results**: Some questions adapted, others not - suggests selective adaptation based on confidence

### What Does This Measure?

This script tests **behavior adaptation capability**:

1. **User Feedback Responsiveness**: Can the model recognize when a user hints they prefer a different option?
2. **Context Integration**: Does the model use dialogue context to make informed decisions?
3. **Flexibility vs. Consistency**: Does the model change too easily or stick to its reasoning appropriately?

### Example Console Output

```text
============================================================
GAP ANALYSIS STATISTICS
============================================================
Total questions processed: 1500
Original gaps (model != human): 450 (30.00%)
Gaps adapted (model changed to human): 320 (71.11% of gaps)
Remaining gaps (model still != human): 130 (28.89% of gaps)
------------------------------------------------------------
Original accuracy: 70.00%
Post-adaptation accuracy: 91.33%
Accuracy improvement: 21.33%
============================================================
Statistics saved to: results/gap_analysis_results_statistics.json
```

## Features

- **Asynchronous processing**: Uses async/await for efficient API calls
- **Automatic retries**: Handles transient errors with exponential backoff
- **Periodic saving**: Saves results incrementally (configurable with `storage_step`)
- **Progress tracking**: Shows progress bar during execution
- **Flexible model support**: Works with OpenAI, local models, and reasoning models
- **Statistics tracking**: Automatically calculates adaptation rates and accuracy metrics

## Comparison with Existing Scripts

### vs. `wvs_values_comparison.py`

- **Purpose**: `wvs_values_comparison.py` computes metrics (EMD, correlation) between datasets
- **This script**: Focuses on individual gaps and generates explanatory rationales
- **Usage**: Use comparison script for aggregate analysis, use this script for detailed gap investigation

### vs. `wvs_values_prediction.py`

- **Purpose**: `wvs_values_prediction.py` generates predictions from user profiles or dialogues
- **This script**: Analyzes existing predictions against ground truth and explains discrepancies
- **Usage**: Use prediction script to generate BA results, then use this script to analyze gaps

## Examples

### Example 1: Quick Test with Small Data Slice

For initial testing, use the test configuration that processes only 10 users:

```bash
python llm_behavior_adaptation/value_measurement/wvs_gap_analysis.py \
    --config wvs_values_gap_analysis/gap_analysis_test_config.yaml
```

**Test Configuration** (`wvs_values_gap_analysis/gap_analysis_test_config.yaml`):

```yaml
# Model configuration
evaluated_model: "qwen/qwen3-30b-a3b-instruct-2507"
llm_server: "llm_platform"
reasoning: true

# Input paths
ba_dialogue_results_path: "wvs_values_results/Qwen3-30B-A3B-Instruct/career/BA_dialogue_values_results/total_1000.jsonl"
human_results_path: "datasets/wvs_benchmarks/sampled_values_df.csv"
generated_dialogues_path: "datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl"
picked_questions_path: "datasets/wvs_benchmarks/picked_questions.json"
prompts_folder: "llm_behavior_adaptation/value_measurement/prompts"

# Output
output_file_path: "wvs_values_gap_analysis/qwen3_30b_a3b_career_gap_analysis_test.jsonl"

# Test settings - process only first 10 users
starting_row: 0
ending_row: 10
storage_step: 5
verbose: 1

# API configuration
model_base_url: "https://openrouter.ai/api/v1"
extra_body:
  reasoning:
    effort: "low"
  provider:
    only:
      - chutes
```

### Example 2: Analyze GPT-4o Career Dialogues

```yaml
evaluated_model: "gpt-4o"
llm_server: "llm_platform"
reasoning: false

ba_dialogue_results_path: "results/ba_dialogue_career_results.jsonl"
human_results_path: "datasets/wvs_benchmarks/sampled_values_df.csv"
generated_dialogues_path: "results/generated_dialogues_career.jsonl"
picked_questions_path: "datasets/wvs_benchmarks/picked_questions.json"
prompts_folder: "llm_behavior_adaptation/value_measurement/prompts"
output_file_path: "results/gap_analysis_career.jsonl"

storage_step: 10
verbose: 1
```

### Example 3: Analyze Qwen Model with Reasoning

For reasoning models like Qwen3-30B-A3B-Instruct:

```yaml
evaluated_model: "qwen/qwen3-30b-a3b-instruct-2507"
llm_server: "llm_platform"
reasoning: true

ba_dialogue_results_path: "wvs_values_results/Qwen3-30B-A3B-Instruct/career/BA_dialogue_values_results/total_1000.jsonl"
human_results_path: "datasets/wvs_benchmarks/sampled_values_df.csv"
generated_dialogues_path: "datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl"
picked_questions_path: "datasets/wvs_benchmarks/picked_questions.json"
prompts_folder: "llm_behavior_adaptation/value_measurement/prompts"
output_file_path: "wvs_values_gap_analysis/qwen3_career_gap_analysis_full.jsonl"

storage_step: 50
verbose: 1

model_base_url: "https://openrouter.ai/api/v1"
extra_body:
  reasoning:
    effort: "low"
  provider:
    only:
      - chutes
```

### Example 4: Process Specific User Range

To process users 100-200 from the dataset:

```yaml
evaluated_model: "gpt-4o"
# ... other config ...
starting_row: 100
ending_row: 200
```

## Prompt Engineering

The gap rationale prompt ([prompts/gap_rationale.json](prompts/gap_rationale.json)) can be customized to:

- Change the tone or style of rationales
- Add domain-specific context
- Request specific types of explanations
- Adjust the level of detail

## Error Handling

The script handles:

- **Missing users**: Skips users not found in human results
- **Missing dialogues**: Skips users without dialogue history
- **Missing questions**: Skips questions not in metadata
- **API errors**: Retries with exponential backoff
- **Invalid responses**: Logs warnings and continues

## Performance

- Processing time depends on:
  - Number of users
  - Number of gaps per user
  - Model latency
  - Number of concurrent requests

- Use `storage_step` to save progress periodically
- Adjust `max_attempts` in retry logic if needed

## Troubleshooting

### Issue: "Missing required config key"

**Solution**: Ensure all required fields are in your config YAML

Required fields:

- `evaluated_model`
- `ba_dialogue_results_path`
- `human_results_path`
- `generated_dialogues_path`
- `picked_questions_path`
- `prompts_folder`
- `output_file_path`

### Issue: "No human results found for user"

**Solution**: Check that user IDs match between BA results and human CSV

The user IDs in the BA dialogue results should match the `D_INTERVIEW` column in the human results CSV.

### Issue: "File not found" errors

**Solution**: Verify all paths are correct relative to the project root

Common issues:

- Check that BA dialogue results file exists at the specified path
- Verify generated dialogues file path is correct
- Ensure datasets are in the expected locations

### Issue: API timeout errors

**Solution**: Increase retry delays or reduce concurrent requests

For reasoning models, consider:

- Using lower reasoning effort (`effort: "low"`)
- Increasing `max_attempts` in retry logic
- Processing smaller batches with `ending_row`

### Issue: Out of memory

**Solution**: Process users in batches using `storage_step` and data slicing

```yaml
storage_step: 10  # Save more frequently
starting_row: 0
ending_row: 100   # Process in chunks of 100 users
```

### Issue: Reasoning model not working

**Solution**: Check reasoning model configuration

Ensure:

1. `reasoning: true` is set
2. `model_base_url` is provided (e.g., OpenRouter URL)
3. `extra_body` has proper reasoning configuration
4. API key is valid for the provider

### Issue: Empty statistics or low gap count

**Solution**: This is expected if the BA model is very accurate

- If `original_gaps` is 0, the model already matches human responses perfectly
- Consider testing with a different model or dataset
- Check if data slicing is limiting the sample size too much

## Future Enhancements

Potential improvements:
- Parallel processing of multiple users
- Caching of LLM responses
- Support for multiple dialogue types in one run
- Gap severity classification
- Automated analysis of gap patterns
