# Gap Analysis: Logs and Data Quality

## Quick Reference

### 📊 Data Quality Status

- **Ground Truth**: 1000 users ✅
- **Generated Dialogues**: 997 users ⚠️ (3 missing)
- **BA Results**: 1000 users ⚠️ (3 with wrong demographics)
- **Gap Analysis**: 997 users ✅ (3 correctly skipped)

### 🚨 Known Issues

**3 users have missing dialogues and invalid BA predictions**:
- `417070392` - Male, 57, Kyrgyzstan (predicted as "61yo woman, Brazil")
- `364072334` - Female, 39, Iran (predicted as "61yo woman, Brazil")
- `124074929` - Male, 48, Canada (predicted as "61yo woman, Brazil")

**Impact**: 0.3% of dataset, automatically handled by gap analysis

### 📝 Log Files

When you run gap analysis, the following files are created:

```
wvs_values_gap_analysis/
├── qwen3_30b_a3b_career_gap_analysis_full.jsonl          # Results
├── qwen3_30b_a3b_career_gap_analysis_full.log            # Structured logs
├── qwen3_30b_a3b_career_gap_analysis_full_statistics.json # Summary stats
└── qwen3_30b_a3b_career_gap_analysis_full_console.log    # Console output (optional)
```

**Structured Log** (`.log`):
- Timestamps, line numbers, function names
- All INFO, WARNING, ERROR messages
- Permanent record of the analysis run

**Console Log** (`.console.log`, optional):
- Raw console output with ANSI colors
- Created by running: `./scripts/capture_gap_analysis_logs.sh <config>`

### 🔍 What to Look For in Logs

**Success indicators**:
```
INFO: Gap analysis logging started
INFO: Loaded BA dialogue results: rows 180 to end (820 users)
INFO: Loaded 817 generated dialogues
```

**Expected warnings** (known data quality issues):
```
WARNING: Found 3 user(s) in BA results but not in generated dialogues: ['124074929', '364072334', '417070392']
WARNING: No dialogue found for user: 417070392 (skipping)
WARNING: No dialogue found for user: 364072334 (skipping)
WARNING: No dialogue found for user: 124074929 (skipping)
```

**Statistics output**:
```
User Statistics:
  Total users in BA results: 820
  Successfully processed users: 817
  Skipped users (missing data): 3
```

## Quick Commands

### View Logs

```bash
# View structured log file
less wvs_values_gap_analysis/qwen3_30b_a3b_career_gap_analysis_full.log

# Search for warnings
grep WARNING wvs_values_gap_analysis/*.log

# Search for specific user
grep "417070392" wvs_values_gap_analysis/*.log

# Count skipped users
grep "skipping" wvs_values_gap_analysis/*.log | wc -l
```

### Run with Log Capture

```bash
# Automatic logging (always enabled)
python llm_behavior_adaptation/value_measurement/wvs_gap_analysis.py \
    --config wvs_values_gap_analysis/qwen3_30b_a3b_career_full_config.yaml

# Capture console output too (optional)
./scripts/capture_gap_analysis_logs.sh \
    wvs_values_gap_analysis/qwen3_30b_a3b_career_full_config.yaml
```

### Check Data Quality

```bash
# Count users in each file
echo "Demographics: $(tail -n +2 datasets/wvs_benchmarks/sampled_demographic_features.csv | wc -l)"
echo "Dialogues: $(wc -l < datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl)"
echo "BA Results: $(wc -l < wvs_values_results/Qwen3-30B-A3B-Instruct/career/BA_dialogue_values_results/total_1000.jsonl)"

# Find missing users
python3 -c "
import json, csv

demo_users = set()
with open('datasets/wvs_benchmarks/sampled_demographic_features.csv') as f:
    for row in csv.DictReader(f):
        demo_users.add(row['D_INTERVIEW'])

dialogue_users = set()
with open('datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl') as f:
    for line in f:
        dialogue_users.update(json.loads(line).keys())

print(f'Missing: {sorted(demo_users - dialogue_users)}')
"
```

## Documentation Files

### Investigation Reports

1. **[INVESTIGATION_SUMMARY.md](INVESTIGATION_SUMMARY.md)** ⭐ **START HERE**
   - Complete findings and answers
   - What happened, why, and what to do
   - Recommended first read

2. **[CRITICAL_DATA_ISSUE.md](CRITICAL_DATA_ISSUE.md)** 🚨
   - Detailed analysis of wrong demographics in BA results
   - Profile substitution bug explanation
   - Impact assessment and recommendations

3. **[MISSING_DIALOGUES_ANALYSIS.md](MISSING_DIALOGUES_ANALYSIS.md)** 📊
   - Detailed profiles of the 3 missing users
   - Root cause analysis
   - Verification commands

4. **[CONSOLE_LOG_ANALYSIS.md](CONSOLE_LOG_ANALYSIS.md)** 📝
   - Explanation of all logging updates
   - How to use the new logging system
   - Log format and structure

### Usage Guides

5. **[GAP_ANALYSIS_README.md](../llm_behavior_adaptation/value_measurement/GAP_ANALYSIS_README.md)** 📖
   - Complete user guide for gap analysis script
   - Configuration, examples, troubleshooting

## Statistics Output

The `*_statistics.json` file now includes user tracking:

```json
{
  "summary": {
    "gap_threshold": 0.5,
    "total_users": 820,
    "processed_users": 817,
    "skipped_users": 3,
    "total_questions": 12255,
    "original_gaps": 3678,
    "queried_gaps": 2456,
    "skipped_gaps": 1222,
    "adapted_gaps": 1845,
    "remaining_gaps_after_adaptation": 611,
    "total_remaining_gaps": 1833,
    "adaptation_rate_percent": 75.12,
    "remaining_gap_rate_percent": 24.88,
    "original_accuracy_percent": 69.98,
    "post_adaptation_accuracy_percent": 85.04
  }
}
```

**Key metrics**:
- `skipped_users`: Users with missing dialogues (expected: 3)
- `processed_users`: Successfully analyzed users
- `total_users`: Total users in BA results

## Important Notes

### For Gap Analysis Users ✅

- **Safe to run**: Missing users are automatically detected and skipped
- **Minimal impact**: 0.3% of data affected
- **Well logged**: All issues captured in log files
- **Statistics accurate**: Skipped users properly tracked

### For BA Accuracy Analysis ⚠️

- **MUST exclude 3 users**: `417070392`, `364072334`, `124074929`
- **Predictions invalid**: Wrong demographics used
- **Document in papers**: Note 0.3% exclusion for data quality

```python
# Example: Exclude invalid users
INVALID_USERS = ['417070392', '364072334', '124074929']

valid_predictions = {
    uid: pred for uid, pred in ba_results.items()
    if uid not in INVALID_USERS
}
```

### For Pipeline Developers 🔧

- **Fix dialogue generation**: Add validation and retry logic
- **Fix BA prediction**: Never use fallback profiles without validation
- **Add alignment checks**: Verify user IDs match across files
- **See CRITICAL_DATA_ISSUE.md** for detailed recommendations

## Quick Diagnosis

### "Are there any data quality issues?"

```bash
# Check skipped users count
grep "Skipped users" wvs_values_gap_analysis/*_statistics.json
# Expected: "skipped_users": 3
```

### "Why are users skipped?"

```bash
# See which users and why
grep "No dialogue found" wvs_values_gap_analysis/*.log
# Expected: 3 users (417070392, 364072334, 124074929)
```

### "Is this normal?"

Yes! These 3 users are a **known data quality issue**:
- Dialogue generation failed for rows 607-609
- Gap analysis correctly detects and skips them
- Impact is minimal (0.3%)
- No action needed for gap analysis
- Just document in papers and exclude from BA accuracy metrics

## Need More Info?

1. **For gap analysis usage**: See [GAP_ANALYSIS_README.md](../llm_behavior_adaptation/value_measurement/GAP_ANALYSIS_README.md)
2. **For missing users details**: See [INVESTIGATION_SUMMARY.md](INVESTIGATION_SUMMARY.md)
3. **For BA accuracy concerns**: See [CRITICAL_DATA_ISSUE.md](CRITICAL_DATA_ISSUE.md)
4. **For logging details**: See [CONSOLE_LOG_ANALYSIS.md](CONSOLE_LOG_ANALYSIS.md)
