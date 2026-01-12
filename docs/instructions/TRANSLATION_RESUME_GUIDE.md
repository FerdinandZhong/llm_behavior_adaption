# Translation Resume Guide

This guide shows you how to resume interrupted translations or process specific ranges of dialogues.

## Quick Start: Resuming an Interrupted Translation

### Step 1: Check Current Progress

```bash
python check_translation_progress.py \
    wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl \
    datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl
```

**Output example:**
```
======================================================================
TRANSLATION PROGRESS CHECK
======================================================================

📊 Translation Progress:
   Translated file: wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl
   Dialogues completed: 247
   Last processed index: 246
   First 5 indices: [0, 1, 2, 3, 4]
   Last 5 indices: [242, 243, 244, 245, 246]

📁 Input File:
   Input file: datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl
   Total dialogues: 1000
   Remaining: 753
   Progress: 247/1000 (24.7%)

✅ To resume translation:

   Using CLI:
   python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
       --config <your_config.yaml> \
       --starting-row 247

   Or update config YAML:
   starting_row: 247
   ending_row: null  # or specify end index
```

### Step 2: Resume Translation

**Option A: Using CLI (Recommended for one-time resume)**

```bash
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config llm_behavior_adaptation/dialogue_dataset_creation/translation_configs/career_translation_config.yaml \
    --starting-row 247
```

**Option B: Update Config File (Better for persistent changes)**

Edit your config file:
```yaml
# Range control (for resuming interrupted translations)
starting_row: 247  # Start from where you left off
ending_row: null   # null means translate till end
```

Then run:
```bash
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config llm_behavior_adaptation/dialogue_dataset_creation/translation_configs/career_translation_config.yaml
```

## Use Cases

### Use Case 1: Resume After Interruption

**Scenario:** Your translation was interrupted at dialogue 247 out of 1000.

```bash
# Check progress
python check_translation_progress.py \
    wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl \
    datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl

# Resume from dialogue 247
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --starting-row 247
```

### Use Case 2: Process in Batches

**Scenario:** You want to process 1000 dialogues in batches of 200 to monitor progress.

```bash
# Batch 1: Dialogues 0-199
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --starting-row 0 \
    --ending-row 200

# Batch 2: Dialogues 200-399
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --starting-row 200 \
    --ending-row 400

# Batch 3: Dialogues 400-599
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --starting-row 400 \
    --ending-row 600

# ... and so on
```

### Use Case 3: Parallel Processing

**Scenario:** You want to speed up translation by running multiple processes in parallel.

**Terminal 1:**
```bash
# Process dialogues 0-250
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --output wvs_generated_dialogues/translated_dialogues/career/part1.jsonl \
    --starting-row 0 \
    --ending-row 250
```

**Terminal 2:**
```bash
# Process dialogues 250-500
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --output wvs_generated_dialogues/translated_dialogues/career/part2.jsonl \
    --starting-row 250 \
    --ending-row 500
```

**Terminal 3:**
```bash
# Process dialogues 500-750
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --output wvs_generated_dialogues/translated_dialogues/career/part3.jsonl \
    --starting-row 500 \
    --ending-row 750
```

**Terminal 4:**
```bash
# Process dialogues 750-1000
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --output wvs_generated_dialogues/translated_dialogues/career/part4.jsonl \
    --starting-row 750 \
    --ending-row 1000
```

Then merge the files:
```bash
cat wvs_generated_dialogues/translated_dialogues/career/part*.jsonl > \
    wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl
```

### Use Case 4: Test Specific Range

**Scenario:** You want to test translation on dialogues 100-105 before processing everything.

```bash
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --starting-row 100 \
    --ending-row 105 \
    --output test_translation.jsonl \
    --verbose 2
```

### Use Case 5: Re-translate Failed Dialogues

**Scenario:** Some dialogues failed during translation and you want to re-translate just those.

First, identify the failed range (e.g., dialogues 300-320 failed):

```bash
# Re-translate just the failed range
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --starting-row 300 \
    --ending-row 320 \
    --output failed_retranslated.jsonl
```

Then manually merge or replace those lines in the main file.

## Configuration Options

### YAML Config

```yaml
# Range control (for resuming interrupted translations)
starting_row: 0      # Starting row index (0-indexed, inclusive)
ending_row: null     # Ending row index (exclusive, null means till end)

# Other useful options
max_dialogues: null  # Limit total dialogues (applied after range selection)
batch_size: 10       # Flush to disk every N dialogues
verbose: 1           # 0=quiet, 1=info, 2=debug
```

### CLI Arguments

```bash
--starting-row N     # Start from dialogue N (0-indexed, inclusive)
--ending-row N       # Stop at dialogue N (exclusive)
--max M              # Process at most M dialogues (within the range)
--verbose V          # Set verbosity level
```

### Priority and Interaction

The parameters work together as follows:

1. **starting_row** sets the beginning (inclusive)
2. **ending_row** sets the end (exclusive)
3. **max_dialogues** further limits the count within the range

**Example:**
```bash
--starting-row 100 --ending-row 500 --max 50
```
This processes dialogues 100-149 (50 dialogues starting from 100, stopping before 150).

## Index vs. Row Number

**Important distinction:**

- **Row Number**: Physical line number in the file (0-indexed)
  - Row 0 = first line in the file
  - Row 247 = 248th line in the file

- **Index**: The `index` field in the normalized dialogue data
  - Usually matches row number (index 0 = row 0)
  - But can differ if dialogues are reordered or filtered

**Rule of thumb:** Use row numbers for `--starting-row` and `--ending-row` since these control which lines to read from the file.

## Log Output

When you use starting/ending row, you'll see helpful logs:

```
[INFO] Total dialogues in file: 1000
[INFO] Processing range: [247, 1000) (753 dialogues)
[INFO] Input: datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl
[INFO] Output: wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl

Translating Dialogues:   0%|                    | 0/753 [00:00<?, ?dialogue/s]
```

This confirms:
- Total dialogues in the input file
- The exact range being processed [start, end)
- How many dialogues will be processed

## Best Practices

### 1. Always Check Progress First

Before resuming, always run the progress checker:
```bash
python check_translation_progress.py <translated_file> <input_file>
```

### 2. Use Append Mode (Automatic)

The controller automatically appends to existing output files, so you won't lose previous work.

### 3. Backup Before Re-running Same Range

If re-translating a range, backup first:
```bash
cp wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl \
   wvs_generated_dialogues/translated_dialogues/career/career_translated.backup.jsonl
```

### 4. Monitor Progress Regularly

For long translations, check progress periodically:
```bash
# In another terminal, run every 10 minutes
watch -n 600 "python check_translation_progress.py translated_file.jsonl input_file.jsonl"
```

### 5. Use Batches for Very Large Datasets

For 10,000+ dialogues, process in batches of 500-1000:
- Easier to monitor and debug
- Can resume from batch boundaries
- Reduces impact of failures

### 6. Test First

Before translating everything, test with a small range:
```bash
--starting-row 0 --ending-row 5 --verbose 2
```

## Troubleshooting

### "No dialogues being processed"

**Symptom:**
```
Processing range: [1000, 1000) (0 dialogues)
Translation completed successfully. Translated 0 dialogues.
```

**Cause:** starting_row >= total dialogues in file

**Solution:** Check the file size first:
```bash
wc -l datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl
```

### "Starting from wrong index"

**Symptom:** Progress checker says last index is 246, but translation starts from 247 and skips dialogues.

**Cause:** Correct behavior! Index 246 is complete, so resume from 247.

**Solution:** This is working as intended. The system correctly resumes after the last completed dialogue.

### "Output file not updating"

**Symptom:** Translation runs but output file doesn't grow.

**Cause:** May be buffering or batch_size is large.

**Solution:**
- Set `batch_size: 1` in config for immediate writes
- Check verbose output to confirm processing

### "Indices are out of order"

**Symptom:** Progress checker shows gaps or out-of-order indices.

**Cause:** Parallel processing or interrupted runs with different ranges.

**Solution:**
- This is usually fine - the system handles it
- If problematic, re-sort the output file by index

## Examples

### Example 1: Basic Resume

```bash
# Step 1: Check where you left off
$ python check_translation_progress.py \
    wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl \
    datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl

# Output: Last processed index: 246

# Step 2: Resume from 247
$ python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --starting-row 247
```

### Example 2: Process Middle Section

```bash
# Translate dialogues 500-600 only
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --starting-row 500 \
    --ending-row 600 \
    --output middle_section.jsonl
```

### Example 3: Resume in Config File

Edit `career_translation_config.yaml`:
```yaml
starting_row: 247
ending_row: null
```

Then run:
```bash
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config llm_behavior_adaptation/dialogue_dataset_creation/translation_configs/career_translation_config.yaml
```

## Summary

The resume functionality gives you fine-grained control over translation ranges:

✅ **Resume interrupted translations** - Continue from where you left off
✅ **Process in batches** - Break large jobs into manageable chunks
✅ **Parallel processing** - Speed up translation with multiple processes
✅ **Test specific ranges** - Debug or validate specific dialogues
✅ **Re-translate failures** - Retry failed sections without redoing everything

Always use `check_translation_progress.py` to determine your resume point, and remember that row indices are 0-indexed and inclusive on the start, exclusive on the end.
