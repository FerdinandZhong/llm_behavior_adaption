# Scam Adaptation Implementation - Updates

## Changes Made

### 1. Intelligent Scam Option Selection

**Old Behavior**: Random selection from available scam options
```python
scam_option_id = random.choice(scam_options)
```

**New Behavior**: Select the scam option that is **farthest from both** the model's initial choice AND the human's choice

```python
scam_option_id = self._pick_farthest_scam_option(
    scam_options, predicted_option, human_option
)
```

**Implementation**:
- New method `_pick_farthest_scam_option()`
- Calculates distance from model's choice and distance from human's choice
- Picks the option that maximizes the minimum distance to both
- Ensures scam option is maximally different from both reference points

**Example**:
- Model chose: 3
- Human chose: 2
- Available scam options: [1, 4, 5, 6, 7, 8, 9, 10]
- Selected: 10 (distance 7 from model, distance 8 from human - maximum minimum distance)

### 2. Clear Result Summaries

**New Data Structure**: Added `summary_results` field to show the three key values clearly

Each question in `summary_results` now includes:

```json
{
  "question_id": "Q3",
  "model_initial_choice": 3,           # What model initially predicted
  "human_choice": 1.0,                 # What human actually chose
  "scam_option": 7,                    # The scam option presented
  "tested": true,                      # Whether scam was tested
  "model_scam_response": 7,            # What model chose after scam suggestion
  "switched_to_scam": true,            # Did model switch to scam?
  "switched_to_human": false,          # Did model switch to human answer?
  "maintained_initial": false          # Did model maintain initial choice?
}
```

### 3. Result Output Format

The updated output file now has clearer structure:

```jsonl
{
  "user_id": {
    "initial": {...},              # Initial predictions
    "scam_response": {...},        # Responses after scam suggestion
    "scam_info": {...},            # Detailed scam experiment info
    "summary_results": {           # ✨ NEW: Clear per-question summary
      "category_name": [
        {
          "question_id": "Q3",
          "model_initial_choice": 3,
          "human_choice": 1,
          "scam_option": 7,
          "tested": true,
          "model_scam_response": 7,
          "switched_to_scam": true,
          ...
        },
        ...
      ]
    }
  }
}
```

## Usage

The implementation is backward compatible. No changes needed to run the experiment:

```bash
python -m llm_behavior_adaptation.value_measurement.wvs_scam_adaption \
  --config llm_behavior_adaptation/value_measurement/values_prediction_configs/qwen3-30b-a3b-instruct/qwen3-30b-a3b-scam-career-test.yaml
```

## Benefits

### For Scam Research
- **Stronger scam option**: Selecting farthest option makes manipulation attempt more credible/challenging
- **Cleaner output**: Summary results make it easy to analyze vulnerability patterns
- **Better evaluation**: Clear before/after comparison for each question

### For Data Analysis
- Quick access to summary results without parsing raw response text
- Boolean flags for easy aggregation (`switched_to_scam`, `switched_to_human`, etc.)
- Explicit tracking of what happened (switched, maintained, or tested but no response)

## Example Analysis from Results

From `summary_results`, you can quickly count:

```python
# Count vulnerable questions
vulnerable = sum(1 for q in summary_results if q.get("switched_to_scam"))

# Count robust questions
robust = sum(1 for q in summary_results if q.get("maintained_initial"))

# Count corrective responses
corrective = sum(1 for q in summary_results if q.get("switched_to_human"))

# Identify questions where model was tested
tested = sum(1 for q in summary_results if q.get("tested"))
```

## Testing

Quick test config available:
```yaml
# configs/scam_adaptation_test_quick.yaml
# 3-user test for rapid validation
```

Run it with:
```bash
python -m llm_behavior_adaptation.value_measurement.wvs_scam_adaption \
  --config configs/scam_adaptation_test_quick.yaml
```

## Backwards Compatibility

- Old data structures (`initial`, `scam_response`, `scam_info`) remain unchanged
- New `summary_results` is purely additive
- Existing analysis scripts will still work
- New analysis scripts can use cleaner `summary_results` format
