# Scam Adaptation - Qwen3 Production Run Guide

## Overview

This guide provides instructions for running the scam adaptation experiment with Qwen3-30B-A3B-Instruct model on the full WVS dataset.

## Configuration

**Production Config File:**
```
llm_behavior_adaptation/value_measurement/values_prediction_configs/qwen3-30b-a3b-instruct/qwen3-30b-a3b-scam-career.yaml
```

### Configuration Details

```yaml
# Model
evaluated_model: "qwen/qwen3-30b-a3b-instruct-2507"
llm_server: "llm_platform"
model_base_url: "https://openrouter.ai/api/v1"

# Dataset
user_profile_dataset_path: "datasets/wvs_benchmarks/sampled_demographic_features.csv"
picked_questions_path: "datasets/wvs_benchmarks/picked_questions.json"
human_results_path: "datasets/wvs_benchmarks/sampled_values_df.csv"

# Processing
starting_row: 0
ending_row: -1  # All users
storage_step: 50  # Save every 50 users

# Performance
reasoning: false  # Faster processing
verbose: 1  # Detailed logs

# API
provider: atlas-cloud/bf16  # Optimized inference
```

## Output Structure

### Results Location
```
wvs_values_results/Qwen3-30B-A3B-Instruct/scam_adaptation/career/
├── results.jsonl           # Main results (JSONL format)
├── results_statistics.json # Summary statistics
└── results.log            # Detailed execution logs
```

### Output Format

**Main Results (results.jsonl)** - Per-user results with:
- `initial`: Initial model predictions
- `scam_response`: Model responses after scam suggestion
- `scam_info`: Detailed scam experiment information
- `summary_results`: Clear per-question summaries with:
  - `model_initial_choice`: Model's initial prediction
  - `human_choice`: Human's actual choice
  - `scam_option`: Scam option presented (farthest from both)
  - `model_scam_response`: Model's response to scam
  - `switched_to_scam`: Boolean flag
  - `switched_to_human`: Boolean flag
  - `maintained_initial`: Boolean flag

**Statistics (results_statistics.json)** - Summary metrics:
- `scam_vulnerability_rate`: % switched to scam
- `human_acceptance_rate`: % switched to human
- `maintenance_rate`: % maintained initial choice
- Pearson correlations before/after

## Running the Experiment

### Prerequisites

1. **Environment Setup**
```bash
conda activate llm_behavior_test
# Ensure api_key is set in environment
echo $api_key  # Should show your API key
```

2. **Verify Data Files Exist**
```bash
ls -la datasets/wvs_benchmarks/
# Should have:
# - sampled_demographic_features.csv
# - picked_questions.json
# - sampled_values_df.csv
```

### Run Command

```bash
python -m llm_behavior_adaptation.value_measurement.wvs_scam_adaption \
  --config llm_behavior_adaptation/value_measurement/values_prediction_configs/qwen3-30b-a3b-instruct/qwen3-30b-a3b-scam-career.yaml
```

### Monitoring Progress

The experiment will show:
- Progress bar with user count
- Periodic saves every 50 users
- Detailed logs in `results.log`

To monitor in real-time:
```bash
tail -f wvs_values_results/Qwen3-30B-A3B-Instruct/scam_adaptation/career/results.log
```

## Expected Results

### Timing Estimates

- **Per user**: ~2-5 seconds (varies with API latency)
- **Full run**: ~1-2 hours for all users
- **Storage**: ~50-100 MB for results.jsonl

### Sample Output Structure

```json
{
  "user_id": {
    "summary_results": {
      "Social Values, Norms, Stereotypes": [
        {
          "question_id": "Q3",
          "model_initial_choice": 3,
          "human_choice": 1,
          "scam_option": 10,
          "tested": true,
          "model_scam_response": 10,
          "switched_to_scam": true,
          "switched_to_human": false,
          "maintained_initial": false
        }
      ]
    }
  }
}
```

## Analysis

### Quick Statistics

```bash
# View summary statistics
cat wvs_values_results/Qwen3-30B-A3B-Instruct/scam_adaptation/career/results_statistics.json | jq .summary
```

### Extract Key Metrics

```python
import json

with open('wvs_values_results/Qwen3-30B-A3B-Instruct/scam_adaptation/career/results_statistics.json') as f:
    stats = json.load(f)
    summary = stats['summary']

print(f"Scam Vulnerability Rate: {summary['scam_vulnerability_rate_percent']}%")
print(f"Human Acceptance Rate: {summary['human_acceptance_rate_percent']}%")
print(f"Maintenance Rate: {summary['maintenance_rate_percent']}%")
print(f"Questions Tested: {summary['scam_tested_questions']}")
```

### Analyze Individual User

```python
import json

with open('wvs_values_results/Qwen3-30B-A3B-Instruct/scam_adaptation/career/results.jsonl') as f:
    for line in f:
        data = json.loads(line)
        user_id = list(data.keys())[0]
        summary = data[user_id]['summary_results']

        for category, questions in summary.items():
            vulnerable = sum(1 for q in questions if q.get('switched_to_scam'))
            robust = sum(1 for q in questions if q.get('maintained_initial'))

            print(f"{user_id} / {category}:")
            print(f"  Vulnerable: {vulnerable}")
            print(f"  Robust: {robust}")
```

## Troubleshooting

### API Key Issues

```bash
# Verify API key is accessible
conda activate llm_behavior_test
python -c "import os; print('api_key set:', bool(os.environ.get('api_key')))"
```

### Interrupted Run

The experiment saves every 50 users. If interrupted:
1. Last complete set of results is safe
2. Verify results.jsonl is valid JSONL
3. Re-run to continue or start fresh

### Low Scores

High scam vulnerability is normal for language models. Factors affecting vulnerability:
- Model architecture
- Fine-tuning data
- Prompt engineering
- Model size and capacity

## Configuration Options

### For Faster Testing

To test on subset before full run:

```yaml
starting_row: 0
ending_row: 100  # First 100 users
storage_step: 10
```

### For Different Question Categories

Modify `picked_questions_path` to use different question sets:
- Career questions (current)
- Investment questions: Use `qwen3-30b-a3b-dialogue-investment.yaml`
- Other categories: Create custom config

### Adjust Storage Frequency

```yaml
storage_step: 25  # Save more frequently
# vs
storage_step: 100  # Save less frequently (use less I/O)
```

## Advanced Usage

### Custom Scam Selection Strategy

The implementation uses "farthest distance" strategy:
- Calculates distance from model's initial choice
- Calculates distance from human's choice
- Selects option maximizing minimum distance
- Ensures scam is far from BOTH reference points

To modify: Edit `_pick_farthest_scam_option()` in `wvs_scam_adaption.py`

### API Provider Selection

Current config uses `atlas-cloud/bf16`. Other options:
```yaml
extra_body:
  provider:
    only:
      - deepinfra/fp4
      - novita/bf16
      - inference-net
```

## Performance Optimization

### Parallel Processing (Future)

The async implementation supports concurrent API calls. Current:
- Sequential per user
- Parallel per question within user

### Cost Optimization

- Using bf16 precision reduces cost
- Batch processing every 50 users
- Model selection (30B vs larger)

## References

- **Implementation Details**: [SCAM_IMPLEMENTATION_INDEX.md](SCAM_IMPLEMENTATION_INDEX.md)
- **Technical Guide**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Updates**: [SCAM_ADAPTION_UPDATES.md](SCAM_ADAPTION_UPDATES.md)

---

**Last Updated**: 2026-01-11
**Config File**: `qwen3-30b-a3b-scam-career.yaml`
**Status**: Production Ready
