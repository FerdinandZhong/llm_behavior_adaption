# Dialogue Translation System

This system translates generated dialogues (both career and investment advice) into appropriate languages based on user profiles.

## Overview

The translation system intelligently selects target languages based on user profile information, with a priority system that ensures the best language match while maintaining LLM translation quality.

## Language Selection Logic

The system follows a three-tier priority for language selection:

### Priority 1: Current Living Country
- First checks the user's `place_of_residence` field
- Maps the country to its primary language
- Uses this language if it's considered "common" (good LLM support)

### Priority 2: Born Country
- If current residence language is uncommon or unmapped, checks `place_of_birth`/`country_of_birth`
- Maps birth country to its primary language
- Uses this language if it's considered "common"

### Priority 3: English Default
- If both countries have uncommon languages or no mapping, defaults to English
- Ensures consistent, high-quality translations

## Common Language Standards

A language is considered "common" for LLM translation if it meets these criteria:

1. **Widespread Use**: Spoken by significant global population
2. **Digital Presence**: Strong representation in online content
3. **LLM Training Data**: Well-represented in LLM training datasets
4. **Translation Quality**: LLMs can produce high-quality, culturally appropriate translations

### Supported Common Languages

The system currently supports these common languages:

**Major International Languages:**
- English, Spanish, French, German, Italian, Portuguese, Russian, Arabic, Japanese, Korean, Chinese

**Regional Languages with Strong LLM Support:**
- Hindi, Bengali, Turkish, Vietnamese, Thai, Indonesian, Malay, Dutch, Polish, Ukrainian, Romanian, Czech, Greek, Hebrew, Persian, Swedish, Norwegian, Danish, Finnish, Hungarian

### Country-Language Mappings

The system includes comprehensive mappings for 100+ countries. Some examples:

- United States → English
- Spain, Mexico, Argentina → Spanish
- France, Belgium → French
- Germany, Austria → German
- China, Taiwan, Singapore → Chinese
- Japan → Japanese
- India → Hindi
- Brazil → Portuguese

*Note: For countries with multiple official languages, the system maps to the most widely spoken or internationally used language.*

## Components

### 1. Core Translation Module (`dialogue_translator.py`)

Contains the core translation logic:

- `DialogueTranslator`: Main translation class
- `select_target_language()`: Language selection logic
- `is_common_language()`: Language commonality checker
- Country-to-language mappings

**Key Features:**
- Translates both user messages and chatbot responses
- Preserves tone, formality, and technical terms
- Handles translation failures gracefully (falls back to original)
- Supports batch processing

### 2. Translation Controller (`translation_controller.py`)

Orchestrates the translation process:

- YAML-based configuration
- Batch processing with progress tracking
- User profile integration
- Error handling and logging

**Key Features:**
- CLI interface with argument overrides
- Automatic output directory creation
- Incremental file writing (batch flush)
- Comprehensive error logging

### 3. Configuration Files

Located in `translation_configs/`:

- `career_translation_config.yaml`: Career advice translation settings
- `investment_translation_config.yaml`: Investment advice translation settings

## Usage

### Basic Usage

#### 1. Using Configuration Files (Recommended)

```bash
# Translate career dialogues
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config llm_behavior_adaptation/dialogue_dataset_creation/translation_configs/career_translation_config.yaml

# Translate investment dialogues
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config llm_behavior_adaptation/dialogue_dataset_creation/translation_configs/investment_translation_config.yaml
```

#### 2. Using CLI Arguments

```bash
# Translate with CLI overrides
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config translation_configs/career_translation_config.yaml \
    --max 100 \
    --verbose 2
```

#### 3. Create New Config

```bash
# Create a sample config file
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --create-config my_translation_config.yaml \
    --topic career
```

### Advanced Usage

#### Programmatic Usage

```python
import asyncio
from openai import AsyncOpenAI
from llm_behavior_adaptation.dialogue_dataset_creation.dialogue_translator import DialogueTranslator

# Initialize client and translator
client = AsyncOpenAI()
translator = DialogueTranslator(
    client=client,
    model="gpt-4.1-mini",
    temperature=0.3,
    verbose=1
)

# Translate a single dialogue
dialogue_data = {
    "index": 0,
    "generated_dialogue": [
        {
            "user_content": "Hello, I need career advice.",
            "chatbot_content": "I'd be happy to help with your career questions."
        }
    ]
}

user_profile = {
    "place_of_residence": "France",
    "place_of_birth": "United States"
}

# Run translation
translated = await translator.translate_dialogue(
    dialogue_data=dialogue_data,
    user_profile=user_profile
)

print(f"Target Language: {translated.target_language}")  # French
print(f"Reason: {translated.language_selection_reason}")
```

#### Batch Processing with Custom Profiles

```python
from llm_behavior_adaptation.dialogue_dataset_creation.dialogue_translator import load_user_profiles_from_csv

# Load user profiles
profiles = load_user_profiles_from_csv("datasets/user_profiles.csv")

# Translate dialogue file
await translator.translate_dialogue_file(
    input_file="datasets/generated_dialogues/career_dialogues.jsonl",
    output_file="datasets/translated_dialogues/career_translated.jsonl",
    user_profiles=profiles,
    max_dialogues=1000
)
```

## Configuration Options

### YAML Configuration Fields

```yaml
# Required fields
input_file_path: "path/to/input.jsonl"
output_file_path: "path/to/output.jsonl"

# Optional fields
user_profiles_path: "path/to/profiles.csv"  # For language selection
translation_model: "gpt-4.1-mini"  # Translation model
translation_temperature: 0.3  # 0.0-1.0, lower = more consistent
max_dialogues: 100  # null = translate all
batch_size: 10  # Dialogues per batch flush
verbose: 1  # 0=quiet, 1=info, 2=debug
openai_api_key: "sk-..."  # Or use env variable
```

### CLI Arguments

- `--config`: Path to YAML config file
- `--input`: Override input dialogue file path
- `--output`: Override output file path
- `--profiles`: Override user profiles CSV path
- `--max`: Override max dialogues to translate
- `--verbose`: Override verbosity level

## Input/Output Formats

### Input Format (Dialogue JSONL)

```json
{
  "index": 0,
  "generated_dialogue": [
    {
      "user_content": "What career path should I pursue?",
      "chatbot_content": "Based on your background, I suggest..."
    },
    {
      "user_content": "What skills should I develop?",
      "chatbot_content": "You should focus on..."
    }
  ]
}
```

### User Profiles CSV Format

Required columns:
- `place_of_residence`: Current living country
- `place_of_birth` or `country_of_birth`: Birth country (fallback)

Optional columns (preserved in output):
- `gender`, `age`, `occupation_group`, etc.

### Output Format (Translated JSONL)

```json
{
  "index": 0,
  "target_language": "Spanish",
  "language_selection_reason": "Selected Spanish based on current residence: Spain",
  "translated_dialogue": [
    {
      "user_content": "¿Qué carrera profesional debo seguir?",
      "chatbot_content": "Basándome en tu experiencia, sugiero...",
      "original_user_content": "What career path should I pursue?",
      "original_chatbot_content": "Based on your background, I suggest..."
    }
  ],
  "user_profile": {
    "place_of_residence": "Spain",
    "place_of_birth": "United States",
    "age": 35,
    ...
  }
}
```

## Examples

### Example 1: User from France

**Input Profile:**
```python
{
    "place_of_residence": "France",
    "place_of_birth": "United States",
    "age": 35
}
```

**Selection Logic:**
- Current residence: France → French (common) ✓
- **Result:** Translates to French

**Output:**
```json
{
  "target_language": "French",
  "language_selection_reason": "Selected French based on current residence: France"
}
```

### Example 2: User from Micronesia

**Input Profile:**
```python
{
    "place_of_residence": "Micronesia",
    "place_of_birth": "Philippines",
    "age": 42
}
```

**Selection Logic:**
- Current residence: Micronesia → No mapping ✗
- Birth country: Philippines → Tagalog (uncommon) ✗
- **Result:** Defaults to English

**Output:**
```json
{
  "target_language": "English",
  "language_selection_reason": "Defaulting to English because current residence (Micronesia) has no mapped language and birth country (Philippines) has no mapped language"
}
```

### Example 3: User from Greenland, born in Denmark

**Input Profile:**
```python
{
    "place_of_residence": "Greenland",
    "place_of_birth": "Denmark",
    "age": 28
}
```

**Selection Logic:**
- Current residence: Greenland → Greenlandic (uncommon) ✗
- Birth country: Denmark → Danish (common) ✓
- **Result:** Translates to Danish

**Output:**
```json
{
  "target_language": "Danish",
  "language_selection_reason": "Selected Danish from birth country (Denmark) because current residence language (Greenlandic) is uncommon"
}
```

## Error Handling

The system includes robust error handling:

1. **Translation Failures**: Falls back to original content
2. **Missing Profiles**: Defaults to English
3. **Invalid Input**: Logs error and continues with next dialogue
4. **API Errors**: Retries and logs detailed error information

## Logging

Verbosity levels:

- `0`: Quiet mode (only errors)
- `1`: Info mode (progress, selections, errors)
- `2`: Debug mode (detailed API calls, full stack traces)

Example log output:
```
[INFO] Loading user profiles from: datasets/profiles.csv
[INFO] Loaded 500 user profiles
[INFO] Starting translation of 500 dialogues
[INFO] [DIALOGUE 0] Selected French based on current residence: France
[INFO] [TRANSLATION] French | user: Quels sont les... | chatbot: Je vous conseille...
[INFO] Translated 10 dialogues...
[INFO] Translation completed successfully. Translated 500 dialogues.
```

## Performance Considerations

- **Batch Size**: Larger batches reduce I/O but increase memory usage
- **Translation Model**:
  - `gpt-4.1-mini`: Fast, cost-effective, good quality
  - `gpt-5-mini`: Higher quality, slightly slower
  - `gpt-4o`: Best quality, most expensive
- **Temperature**: 0.3 recommended for consistent translations
- **Max Dialogues**: Use for testing before full runs

## Testing

### Unit Tests

Test the language selection logic:

```python
from llm_behavior_adaptation.dialogue_dataset_creation.dialogue_translator import (
    select_target_language,
    is_common_language,
    get_country_language
)

# Test common language check
assert is_common_language("English") == True
assert is_common_language("Greenlandic") == False

# Test country mapping
assert get_country_language("France") == "French"
assert get_country_language("Japan") == "Japanese"

# Test language selection
profile = {"place_of_residence": "Spain", "place_of_birth": "USA"}
lang, reason = select_target_language(profile)
assert lang == "Spanish"
```

### Integration Test

Test with a small sample:

```bash
# Create test config with max_dialogues: 5
python -m llm_behavior_adaptation.dialogue_dataset_creation.translation_controller \
    --config test_config.yaml \
    --max 5 \
    --verbose 2
```

## Troubleshooting

### Common Issues

**Issue: "Missing OpenAI API key"**
- Set `OPENAI_API_KEY` environment variable
- Or add `openai_api_key` to YAML config

**Issue: "Input file not found"**
- Check `input_file_path` in config
- Ensure path is relative to execution directory

**Issue: "Translation quality is poor"**
- Try a higher-quality model (gpt-5-mini or gpt-4o)
- Lower temperature (0.1-0.2) for more literal translations
- Check if language is in COMMON_LANGUAGES list

**Issue: "Wrong language selected"**
- Verify user profile has correct country names
- Check COUNTRY_LANGUAGE_MAP for your countries
- Use `force_language` parameter to override

## Extending the System

### Adding New Languages

```python
# In dialogue_translator.py

# 1. Add to COMMON_LANGUAGES
COMMON_LANGUAGES.add("Swahili")

# 2. Add country mappings
COUNTRY_LANGUAGE_MAP.update({
    "Kenya": "Swahili",
    "Tanzania": "Swahili",
})
```

### Custom Language Selection Logic

```python
def custom_select_language(user_profile):
    """Custom language selection based on additional criteria."""
    # Your custom logic here
    age = user_profile.get("age", 0)
    if age < 30:
        # Younger users: prioritize English
        return "English", "Selected English for younger demographic"
    else:
        # Default logic
        return select_target_language(user_profile)
```

## Best Practices

1. **Test First**: Always test with `max_dialogues: 5-10` before full runs
2. **Version Config**: Keep different configs for testing and production
3. **Monitor Costs**: Track API usage, especially with larger models
4. **Validate Output**: Spot-check translated dialogues for quality
5. **Backup Original**: Always preserve original dialogue files
6. **Incremental Processing**: Use `batch_size` for large datasets to allow interruption
7. **Log Everything**: Use `verbose: 1` or `2` to track progress and issues

## Future Enhancements

Potential improvements to consider:

- [ ] Support for bilingual users (translate to multiple languages)
- [ ] Cultural adaptation beyond translation (idioms, examples)
- [ ] Caching frequently translated phrases
- [ ] Quality scoring for translations
- [ ] A/B testing different translation approaches
- [ ] Support for regional language variants (e.g., Spain Spanish vs. Mexico Spanish)
- [ ] Integration with professional translation services for critical content
- [ ] Automatic language detection for input validation

## License

[Add your license information here]

## Contact

For questions or issues, please [add contact information or issue tracker link].
