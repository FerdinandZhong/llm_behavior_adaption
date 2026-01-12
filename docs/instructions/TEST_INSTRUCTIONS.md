# Running the Scam Adaptation Test for Qwen3

## Setup

The scam adaptation test has been created and configured for testing with Qwen3-30B-A3B-Instruct model.

## Files Created

1. **Main Script**: `llm_behavior_adaptation/value_measurement/wvs_scam_adaption.py`
   - Core implementation of the scam adaptation experiment

2. **Test Configuration**: `llm_behavior_adaptation/value_measurement/values_prediction_configs/qwen3-30b-a3b-instruct/qwen3-30b-a3b-scam-career-test.yaml`
   - Small test run (10 users) for quick validation
   - Configured for career questions
   - Output to: `wvs_values_results/Qwen3-30B-A3B-Instruct/scam_adaptation/career/scam_test.jsonl`

3. **Documentation**: `docs/SCAM_ADAPTATION.md`
   - Complete documentation of the experiment

## Running the Test

### Option 1: Direct Run (from your conda environment)

```bash
# Activate the conda environment if not already activated
conda activate llm_behavior_test

# Run the test
python -m llm_behavior_adaptation.value_measurement.wvs_scam_adaption \
  --config llm_behavior_adaptation/value_measurement/values_prediction_configs/qwen3-30b-a3b-instruct/qwen3-30b-a3b-scam-career-test.yaml
```

### Option 2: Using the Helper Script

```bash
# Make script executable
chmod +x run_scam_test.py

# Run with your conda environment
conda activate llm_behavior_test
python run_scam_test.py
```

## Expected Output

The test will generate:

1. **Main Results**: `wvs_values_results/Qwen3-30B-A3B-Instruct/scam_adaptation/career/scam_test.jsonl`
   - Detailed per-user, per-question results

2. **Statistics**: `wvs_values_results/Qwen3-30B-A3B-Instruct/scam_adaptation/career/scam_test_statistics.json`
   - Summary statistics including:
     - Scam vulnerability rate (% of times model switched to scam option)
     - Human acceptance rate (% of times model switched to human answer)
     - Maintenance rate (% of times model maintained initial choice)
     - Pearson correlation before/after

3. **Logs**: `wvs_values_results/Qwen3-30B-A3B-Instruct/scam_adaptation/career/scam_test.log`
   - Detailed execution logs

## Quick Test Parameters

Current configuration (`qwen3-30b-a3b-scam-career-test.yaml`):
- **Users**: First 10 (rows 0-9)
- **Model**: `qwen/qwen3-30b-a3b-instruct-2507`
- **Server**: OpenRouter (`https://openrouter.ai/api/v1`)
- **Storage Step**: Save every 2 users
- **Verbose**: Detailed logging enabled

## For Full Test Run

To run a full test across all users, create a new config file or modify ending_row:

```yaml
starting_row: 0
ending_row: -1  # -1 means all rows
storage_step: 50  # Save every 50 users
```

## Troubleshooting

### API Key Not Found

If you get `RuntimeError: Missing OpenAI API key`:

```bash
# Make sure api_key environment variable is set in your conda environment
conda activate llm_behavior_test
echo $api_key  # Should show your API key

# If empty, set it:
export api_key="your-api-key-here"
```

### Module Not Found

Make sure you have the required dependencies installed:

```bash
conda activate llm_behavior_test
pip install -e .  # Install the package in development mode
```

## Interpreting Results

See `docs/SCAM_ADAPTATION.md` for detailed interpretation of:
- Scam vulnerability rate
- Human acceptance rate
- Maintenance rate
- Correlation metrics
