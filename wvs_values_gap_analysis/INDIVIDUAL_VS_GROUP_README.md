# Individual vs Group Alignment Analysis

This analysis determines whether a model's predictions are more aligned with individual users or their demographic group medians. This reveals whether the model exhibits stereotypical thinking (group bias) or personalizes to individuals.

## Methodology

For each (user, question) pair:

1. **Get individual answer**: The actual human answer for this user
2. **Get group median**: The median answer from all users in the same demographic group
3. **Get model prediction**: The model's predicted answer for this user
4. **Calculate distances**:
   - Distance to individual = |model_prediction - individual_answer|
   - Distance to group = |model_prediction - group_median|
5. **Determine winner**:
   - If distance_to_individual < distance_to_group → **Individual win**
   - If distance_to_group < distance_to_individual → **Group win**
   - If equal → **Tie**

### Example

- User A is in age group "30-40"
- Question 1: "Importance of family" (scale 1-10)
- User A's answer: 8 (individual option)
- Age group "30-40" median: 6 (group median)
- Model prediction: 7

**Analysis**:
- Distance to individual: |7 - 8| = 1
- Distance to group: |7 - 6| = 1
- Result: **Tie**

Another example:
- User A's answer: 8
- Group median: 4
- Model prediction: 7

**Analysis**:
- Distance to individual: |7 - 8| = 1
- Distance to group: |7 - 4| = 3
- Result: **Individual win** (model is closer to individual)

## Metrics

### 1. Individual Wins
Total number of cases where model prediction is closer to individual user's answer than to group median.

### 2. Group Wins
Total number of cases where model prediction is closer to group median than to individual user's answer.

### 3. Ties
Cases where distance to individual equals distance to group.

### 4. Bias Score
Ranges from -1 to +1:
- **+1**: Pure individual alignment (every prediction closer to individual)
- **0**: No bias (equal alignment to individuals and groups)
- **-1**: Pure group alignment (every prediction closer to group median)

Formula: `bias_score = (individual_wins - group_wins) / total_comparisons`

### 5. Bias Interpretation
- `> 0.3`: Strong individual alignment - model personalizes well
- `0.1 to 0.3`: Moderate individual alignment
- `-0.1 to 0.1`: Neutral - no clear bias
- `-0.3 to -0.1`: Moderate group alignment - some stereotypical thinking
- `< -0.3`: Strong group alignment - heavy reliance on stereotypes

## Demographic Attributes

The script supports grouping by various demographic attributes from `wvs_values_comparison.py`:

### Age
Groups: `<30`, `30-40`, `40-50`, `50-60`, `>60`

### Education (highest_level_of_education)
Groups:
- `Basic education`
- `High school & equivalent`
- `Short-cycle tertiary`
- `Bachelor`
- `Master's & Doctoral`

### Occupation (occupation_group)
Groups:
- `Clerical & Sales`
- `Skilled & Semi-Skilled`
- `Service & Labor`
- `Managerial / Professional`
- `Agricultural Related`
- `No Job`

### Other Attributes
- `continent_of_residence`
- `immigration_status`
- `socioeconomic_status`

## Usage

### Run with configuration file:

```bash
python llm_behavior_adaptation/value_measurement/wvs_individual_vs_group_alignment.py \
  --config wvs_values_gap_analysis/qwen3_30b_a3b_career_individual_vs_group_test_config.yaml
```

### Configuration File Format:

```yaml
# Input paths
ba_dialogue_results_path: "path/to/model/results.jsonl"
human_results_path: "path/to/human/results.csv"
user_profile_path: "path/to/user/profiles.csv"
picked_questions_path: "path/to/questions.json"

# Output configuration
output_file_path: "path/to/output/results.jsonl"

# Analysis configuration
demographic_attribute: "age"  # Choose from: age, highest_level_of_education, occupation_group, etc.

# Data slicing
starting_row: 0    # Start from first user
ending_row: 10     # Process first 10 users (null for all)

# Optional settings
verbose: 1  # 1 = detailed logging
```

## Output Files

### Statistics File (`*_statistics.json`)

Contains:
```json
{
  "demographic_attribute": "age",
  "total_users": 10,
  "num_groups": 5,
  "groups": {
    "<30": 2,
    "30-40": 3,
    "40-50": 2,
    "50-60": 2,
    ">60": 1
  },
  "alignment_summary": {
    "total_comparisons": 541,
    "individual_wins": 320,
    "group_wins": 200,
    "ties": 21,
    "skipped": 9
  },
  "alignment_rates": {
    "individual_alignment_rate": 59.15,
    "group_alignment_rate": 36.97,
    "tie_rate": 3.88
  },
  "bias_score": 0.2217,
  "bias_interpretation": "Moderate individual alignment - model somewhat personalizes to individuals"
}
```

### Detailed Results File (`*.jsonl`)

Line-delimited JSON with per-comparison details:
```json
{
  "user_id": "12345",
  "question_id": "Q1",
  "user_group": "30-40",
  "model_option": 7,
  "individual_option": 8,
  "group_median": 6.0,
  "dist_to_individual": 1,
  "dist_to_group": 1,
  "winner": "tie"
}
```

## Example Analyses

### Test Run (First 10 Users)
```bash
python llm_behavior_adaptation/value_measurement/wvs_individual_vs_group_alignment.py \
  --config wvs_values_gap_analysis/qwen3_30b_a3b_career_individual_vs_group_test_config.yaml
```

### Full Analysis (All 1000 Users, Age Groups)
```bash
python llm_behavior_adaptation/value_measurement/wvs_individual_vs_group_alignment.py \
  --config wvs_values_gap_analysis/qwen3_30b_a3b_career_individual_vs_group_config.yaml
```

### Education Groups Analysis
Update config:
```yaml
demographic_attribute: "highest_level_of_education"
output_file_path: "wvs_values_results/.../education_alignment_full.jsonl"
```

Then run as above.

## Interpretation Guide

### High Individual Alignment (bias_score > 0.3)
✓ Model successfully personalizes to individual users
✓ Goes beyond demographic stereotypes
✓ Captures individual-level variation

### Neutral (bias_score ≈ 0)
- Model may be using mixed signals
- Could indicate uncertainty or balanced approach
- Neither strongly personalized nor stereotypical

### High Group Alignment (bias_score < -0.3)
⚠ Model relies heavily on demographic stereotypes
⚠ May not capture individual variation
⚠ Risk of unfair generalization based on demographics

## Use Cases

1. **Evaluate Personalization**: Does the model truly personalize, or just use stereotypes?
2. **Fairness Analysis**: Is the model making unfair assumptions based on demographics?
3. **Compare Models**: Which model better captures individual variation vs stereotypes?
4. **Attribute Analysis**: Which demographic attributes lead to more stereotypical predictions?

## Notes

- Missing data (NaN in human results) is automatically skipped
- Users without demographic group assignment are skipped
- Questions not present in group medians are skipped
- The script handles the same demographic groupings as `wvs_values_comparison.py`
