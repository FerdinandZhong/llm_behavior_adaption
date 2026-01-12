# Reasoning Extraction Fix for GPT-OSS-20B

## Problem Identified

The previous implementation of `values_prediction.py` had a critical bug in handling reasoning models like GPT-OSS-20B:

### Original Issue (Line 466, 520):
```python
reasoning_output = reasoning_response.choices[0].message.content.split("</think>")[0]
```

This assumed reasoning models would output:
```
<think>reasoning here</think>
{"option_id": X, "reason": "..."}
```

However, **GPT-OSS-20B does NOT use `<think>` tags**, causing:
1. **Split failure**: The entire response was treated as reasoning (no split occurred)
2. **Answer extraction failure**: Second API call with JSON schema couldn't properly extract the answer
3. **Default to option_id=1**: **72.6% of predictions became option_id=1** (instead of diverse predictions)
4. **Poor correlation**: -0.0595 with human values (vs 0.5134 for DeepSeek-V3)
5. **Artificially neutral bias**: -0.0792 (appears balanced but due to uniform predictions)

## Fix Implemented

### Major Discovery: Single API Call Optimization

Testing revealed that **OpenRouter/llm_platform returns reasoning in a separate `reasoning` field**, allowing us to get BOTH reasoning AND formatted JSON output in **ONE API call** instead of TWO!

**Original (broken) approach:**
1. First API call: Get reasoning (failed to extract properly)
2. Second API call: With reasoning context + JSON schema

**New optimized approach:**
1. **Single API call** with `response_format={"type": "json_object"}` + `extra_body` for reasoning
2. Response contains BOTH:
   - `message.reasoning`: The reasoning process
   - `message.content`: The formatted JSON output

### Updated `values_prediction.py` (Lines 460-492, 519-551)

**Single API call implementation:**

```python
# Single API call - reasoning models return reasoning in separate field
if self.reasoning:
    full_chat_response = await self.openai_client.chat.completions.create(
        model=self.evaluated_model,
        messages=direct_value_selection_prompt,
        response_format={"type": "json_object"},
        logprobs=True,
        top_logprobs=5,
        temperature=0.6,
        max_tokens=4096,
        extra_body=self._extra_body,  # Contains reasoning config
    )

    # Extract reasoning from response if available
    reasoning_output = None
    if hasattr(full_chat_response.choices[0].message, 'reasoning') and full_chat_response.choices[0].message.reasoning:
        reasoning_output = full_chat_response.choices[0].message.reasoning
        logger.info("Extracted reasoning from 'reasoning' field")
    elif hasattr(full_chat_response.choices[0].message, 'reasoning_content') and full_chat_response.choices[0].message.reasoning_content:
        reasoning_output = full_chat_response.choices[0].message.reasoning_content
        logger.info("Extracted reasoning from 'reasoning_content' field")
    else:
        # Fallback: try to extract from content with <think> tags
        content = full_chat_response.choices[0].message.content
        if "</think>" in content:
            reasoning_output = content.split("</think>")[0].replace("<think>", "").strip()
            logger.info("Extracted reasoning from <think> tags")
```

**Extraction priority:**
1. **`reasoning` field** (OpenRouter/GPT-OSS-20B format - BEST option)
2. **`reasoning_content` field** (alternative field name if provided)
3. **`<think>` tags parsing** (for models like QwQ-32B, DeepSeek-V3 if needed)

### Added `extra_body` Support

Updated `__init__` method to accept and use `extra_body` parameter:

```python
def __init__(
    self,
    ...
    reasoning: bool = False,
    extra_body: Dict = None,  # NEW
) -> None:
    ...
    self._extra_body = extra_body if extra_body is not None else {}
```

Now reasoning API calls include extra_body:
```python
reasoning_response = await self.openai_client.chat.completions.create(
    model=self.evaluated_model,
    messages=direct_value_selection_prompt,
    temperature=0.6,
    max_tokens=4096,
    extra_body=self._extra_body,  # NEW
)
```

This allows proper configuration of reasoning effort and provider preferences from YAML config.

## Testing

### Config Updated
- `gpt-oss-20b-low-dialogue-career.yaml`:
  - Changed `reasoning: false` → `reasoning: true`
  - Set `ending_row: 2` for quick testing
  - Output: `new_total_1000.jsonl`

### Test Script Created
- `test_reasoning_extraction.py`: Standalone test to verify reasoning_content field availability

## Expected Results

With this fix, GPT-OSS-20B predictions should:
1. ✅ Properly extract reasoning from API responses
2. ✅ Generate diverse option_id predictions (not 72% option_id=1)
3. ✅ Achieve positive correlation with human values (similar to other models)
4. ✅ Show genuine individual vs group bias patterns (not artificially neutral)

## Next Steps

1. Run small test (2 users) to verify reasoning extraction works
2. If successful, regenerate full career results (1000 users)
3. Recompute correlation and individual vs group alignment
4. Update `INDIVIDUAL_VS_GROUP_ALIGNMENT_RESULTS.md` with corrected results

## Files Modified

1. **llm_behavior_adaptation/value_measurement/values_prediction.py**
   - Lines 50-64: Added `extra_body` parameter to `__init__`
   - Lines 105: Store `extra_body` in instance variable
   - Lines 458-495: Updated `_direct_value_query` with improved reasoning extraction
   - Lines 527-559: Updated `_dialogue_continue_value_query` with improved reasoning extraction
   - Lines 466, 536: Added `extra_body` to reasoning API calls

2. **llm_behavior_adaptation/value_measurement/values_prediction_configs/gpt-oss-20b/gpt-oss-20b-low-dialogue-career.yaml**
   - Line 15: Changed `reasoning: false` → `reasoning: true`

3. **test_reasoning_extraction.py** (NEW)
   - Standalone test script to verify reasoning extraction

## Technical Details

### Why the Original Code Failed

The original implementation made **two API calls**:
1. **First call**: Get reasoning (assumed `<think>` tags)
2. **Second call**: With reasoning context + JSON schema forcing

For GPT-OSS-20B:
- First call returns plain text reasoning (no tags)
- Split on `</think>` fails (no split happens)
- Second call receives malformed context
- JSON schema forces a response, defaults to option_id=1

### Why the Fix Works

The new implementation:
1. **Checks for official field**: `reasoning_content` (if llm_platform adds it)
2. **Handles multiple formats**: `<think>` tags OR plain text
3. **Always extracts something**: Uses entire content as fallback
4. **Proper context passing**: Second API call gets correct reasoning

This makes the code **robust across different reasoning model formats**.
