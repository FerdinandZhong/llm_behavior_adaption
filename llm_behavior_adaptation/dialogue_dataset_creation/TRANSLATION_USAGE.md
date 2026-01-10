# Quick Start Guide: Dialogue Translation

This guide shows you how to translate your generated dialogues to appropriate languages based on user profiles.

## Prerequisites

1. Generated dialogue JSONL file (from dialogue generation)
2. User profiles CSV with country information
3. OpenAI API key set in environment

## Quick Start

### Step 1: Set up your environment

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Step 2: Translate career dialogues

```bash
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config llm_behavior_adaptation/dialogue_dataset_creation/translation_configs/career_translation_config.yaml
```

### Step 3: Translate investment dialogues

```bash
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config llm_behavior_adaptation/dialogue_dataset_creation/translation_configs/investment_translation_config.yaml
```

## Configuration Files

Edit the YAML config files to customize paths and settings:

**Career Config:** `llm_behavior_adaptation/dialogue_dataset_creation/translation_configs/career_translation_config.yaml`
**Investment Config:** `llm_behavior_adaptation/dialogue_dataset_creation/translation_configs/investment_translation_config.yaml`

Key settings to adjust:
```yaml
input_file_path: "datasets/generated_dialogues/career_dialogues.jsonl"  # Your input
output_file_path: "datasets/translated_dialogues/career_translated.jsonl"  # Your output
user_profiles_path: "datasets/wvs_benchmarks/sampled_demographic_features.csv"  # Profiles
max_dialogues: null  # null = translate all, or set a number for testing
batch_size: 10  # How many to process before writing to disk
verbose: 1  # 0=quiet, 1=info, 2=debug
```

## Testing with Small Sample

Before translating all dialogues, test with a small sample:

```bash
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --max 5 \
    --verbose 2
```

## How Language Selection Works

The system automatically selects the best language for each user:

1. **Current Country First**: If user lives in France → translates to French
2. **Birth Country Fallback**: If current country's language is uncommon (e.g., Greenland), uses birth country language
3. **English Default**: If both countries have uncommon languages, defaults to English

### Supported Languages (32 total)

**Major Languages:**
- English, Spanish, French, German, Italian, Portuguese, Russian, Arabic
- Japanese, Korean, Chinese, Hindi, Bengali, Urdu, Turkish, Vietnamese, Thai, Indonesian, Malay

**European Languages:**
- Dutch, Polish, Ukrainian, Romanian, Czech, Greek, Swedish, Norwegian, Danish, Finnish, Hungarian, Hebrew, Persian

## User Profile Format

Your CSV should have these columns:
- `place_of_residence`: Current living country (e.g., "France", "Japan", "Brazil")
- `place_of_birth` or `country_of_birth`: Birth country (fallback)

Example CSV:
```csv
D_INTERVIEW,gender,age,place_of_residence,place_of_birth
123456,Male,35,France,United States
234567,Female,42,Japan,Japan
345678,Male,28,Greenland,Denmark
```

## Output Format

Translated dialogues include:
- Original text (preserved)
- Translated text
- Target language selected
- Reason for language selection

Example output:
```json
{
  "index": 0,
  "target_language": "French",
  "language_selection_reason": "Selected French based on current residence: France",
  "translated_dialogue": [
    {
      "user_content": "Quels sont mes prochains pas de carrière?",
      "chatbot_content": "Basé sur votre profil, je suggère...",
      "original_user_content": "What are my next career steps?",
      "original_chatbot_content": "Based on your profile, I suggest..."
    }
  ]
}
```

## Common Issues

### "Missing OpenAI API key"
Set the environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### "Input file not found"
Check that your `input_file_path` in the config points to the correct file.

### Testing the System
Run unit tests to verify language selection logic:
```bash
python test_translation.py
```

## Resuming Interrupted Translations

If your translation is interrupted, you can easily resume from where you left off:

### Step 1: Check Progress

```bash
python check_translation_progress.py \
    wvs_generated_dialogues/translated_dialogues/career/career_translated.jsonl \
    datasets/wvs_generated_dialogues/career_advice/all_samples.jsonl
```

This shows how many dialogues are complete and tells you where to resume.

### Step 2: Resume

```bash
# Resume from dialogue 247 (where you left off)
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --starting-row 247
```

See [TRANSLATION_RESUME_GUIDE.md](../../../TRANSLATION_RESUME_GUIDE.md) for detailed examples and use cases.

## Advanced Usage

### Override Config Settings via CLI

```bash
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config career_config.yaml \
    --input my_dialogues.jsonl \
    --output my_translated.jsonl \
    --max 100 \
    --verbose 2
```

### Create Custom Config

```bash
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --create-config my_custom_config.yaml \
    --topic career
```

## Model Selection

Choose translation model based on your needs:

- **gpt-4.1-mini** (default): Fast, cost-effective, good quality
- **gpt-5-mini**: Higher quality, slightly slower
- **gpt-4o**: Best quality, most expensive

Edit in config:
```yaml
translation_model: "gpt-4.1-mini"
```

## Monitoring Progress

Use verbose mode to see detailed progress:

```bash
--verbose 1  # Shows language selection and progress every 10 dialogues
--verbose 2  # Shows every translation call (detailed)
```

Example output:
```
[INFO] Loading user profiles from: datasets/profiles.csv
[INFO] Loaded 500 user profiles
[INFO] Starting translation of 500 dialogues
[INFO] [DIALOGUE 0] Selected French based on current residence: France
[INFO] [TRANSLATION] French | user: Quels sont les... | chatbot: Je vous conseille...
[INFO] Translated 10 dialogues...
...
[INFO] Translation completed successfully. Translated 500 dialogues.
```

## Full Documentation

For complete details, see [TRANSLATION_README.md](./TRANSLATION_README.md)

## Getting Help

1. Check configuration file format
2. Run unit tests: `python test_translation.py`
3. Enable verbose mode: `--verbose 2`
4. Review the full documentation
