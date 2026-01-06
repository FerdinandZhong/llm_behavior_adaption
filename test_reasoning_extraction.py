"""Test script to verify reasoning extraction from GPT-OSS-20B"""
import asyncio
import os
import sys
import yaml
from openai import AsyncOpenAI

async def test_reasoning_extraction():
    """Test if reasoning_content field is available from llm_platform"""

    # Load config to get extra_body settings
    config_path = "llm_behavior_adaptation/value_measurement/values_prediction_configs/gpt-oss-20b/gpt-oss-20b-low-dialogue-career.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    extra_body = config.get('extra_body', {})
    base_url = config.get('model_base_url', 'https://openrouter.ai/api/v1')

    # Try to get API key from environment or prompt user
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("api_key") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("ERROR: No API key found. Please set one of: OPENROUTER_API_KEY, api_key, or OPENAI_API_KEY")
        print("Or run: export OPENROUTER_API_KEY='your_key_here'")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    # Test message - asking for structured JSON output
    messages = [
        {
            "role": "system",
            "content": "You are evaluating a person's values. You must respond with JSON in the format: {\"option_id\": <number>, \"reason\": <string>}"
        },
        {
            "role": "user",
            "content": """You are a 30-year-old engineer from India.

Question: How important is leisure time in your life?
Options:
1. Not at all important
2. Not very important
3. Rather important
4. Very important

Select the option (1-4) that best matches this person's values and provide a brief reason."""
        }
    ]

    print("Testing GPT-OSS-20B with reasoning mode (low effort)...")
    print("Goal: Get both reasoning AND formatted JSON output in one request")
    print("-" * 60)

    try:
        print(f"\nUsing extra_body config: {extra_body}")

        # Test 1: Without JSON schema (just with reasoning)
        print("\n" + "="*60)
        print("TEST 1: Request with reasoning but NO JSON schema")
        print("="*60)

        response1 = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.6,
            max_tokens=4096,
            extra_body=extra_body
        )

        print("\n1. Response attributes:")
        message1 = response1.choices[0].message
        print(f"   - Has 'reasoning' field: {hasattr(message1, 'reasoning')}")
        if hasattr(message1, 'reasoning'):
            print(f"   - Reasoning: {message1.reasoning}")
        print(f"   - Content: {message1.content[:150]}...")

        # Test 2: With JSON schema + reasoning
        print("\n" + "="*60)
        print("TEST 2: Request with reasoning AND JSON schema")
        print("="*60)

        response2 = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            temperature=0.6,
            max_tokens=4096,
            response_format={"type": "json_object"},
            extra_body=extra_body
        )

        message2 = response2.choices[0].message
        print(f"   - Has 'reasoning' field: {hasattr(message2, 'reasoning')}")
        if hasattr(message2, 'reasoning'):
            print(f"   - Reasoning: {message2.reasoning}")
        print(f"   - Content: {message2.content[:200]}...")

        # Try to parse JSON
        try:
            import json
            json_data = json.loads(message2.content)
            print(f"\n2. Parsed JSON:")
            print(f"   - option_id: {json_data.get('option_id')}")
            print(f"   - reason: {json_data.get('reason')}")
        except json.JSONDecodeError as e:
            print(f"\n2. JSON parsing FAILED: {e}")

        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        if hasattr(message2, 'reasoning') and message2.reasoning:
            print("✓ Reasoning field is available in response")
            print(f"  Reasoning: {message2.reasoning}")
        else:
            print("✗ Reasoning field NOT available")

        try:
            json_data = json.loads(message2.content)
            if 'option_id' in json_data:
                print(f"✓ JSON output is properly formatted")
                print(f"  option_id: {json_data['option_id']}")
                print(f"  reason: {json_data.get('reason', 'N/A')[:100]}")
            else:
                print("✗ JSON missing 'option_id' field")
        except:
            print("✗ Content is not valid JSON")

        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)

        has_reasoning = hasattr(message2, 'reasoning') and message2.reasoning
        has_json = False
        try:
            json_data = json.loads(message2.content)
            has_json = 'option_id' in json_data
        except:
            pass

        if has_reasoning and has_json:
            print("✓✓✓ SUCCESS: Single API call can get BOTH reasoning AND JSON!")
            print("    We can simplify to ONE API call instead of TWO!")
            print("\n    Recommendation: Update values_prediction.py to use single-call approach")
            print("    with response_format={'type': 'json_object'} + extra_body for reasoning")
        elif has_reasoning:
            print("⚠ PARTIAL: Has reasoning but JSON needs work")
        elif has_json:
            print("⚠ PARTIAL: Has JSON but reasoning missing")
        else:
            print("✗ FAILED: Neither reasoning nor JSON working properly")

        print("\n" + "=" * 60)
        print("Test completed!")
        return has_reasoning and has_json

    except Exception as e:
        print(f"\nERROR during API call: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_reasoning_extraction())
    sys.exit(0 if result else 1)
