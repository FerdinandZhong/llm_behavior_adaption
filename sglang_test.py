from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI(base_url=f"http://127.0.0.1:30000/v1", api_key="None")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to find the weather for, e.g. 'San Francisco'",
                    },
                    "state": {
                        "type": "string",
                        "description": "the two-letter abbreviation for the state that the city is"
                        " in, e.g. 'CA' which would mean 'California'",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit to fetch the temperature in",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "state", "unit"],
            },
        },
    }
]

# Define the schema using Pydantic
class BestPlayer(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the player")
    reason: str = Field(..., description="The reason of choosing the player")
    # through: str = Field(..., description="The reasoning process for the question")
    # mayer: str = Field(..., description="Mayer of the capital city")

reasoning = client.chat.completions.create(
    model="Qwen/QwQ-32B",
    messages=[
        {
            "role": "user",
            "content": "Think who is the goat in history of football"
        },
    ],
    max_tokens=4096,
    temperature=0.6,
    top_p=0.95,
    # top_k=40,
    # min_p=0.0,
    # repetition_penalty=1.0
    # response_format={
    #     "type": "json_schema",
    #     "json_schema": {
    #         "name": "football_schema",
    #         # convert the pydantic model to json schema
    #         "schema": BestPlayer.model_json_schema(),
    #     },
    # },
).choices[0].message.content.split("</think>")[0]

print(reasoning)

response = client.chat.completions.create(
    model="Qwen/QwQ-32B",
    messages=[
        {
            "role": "user",
            "content": "who is the goat in history of football"
        },
        {
            "role": "assistant",
            "content": f"<think>\n{reasoning}\n</think>\n"
        }
        # {
        #     "role": "user",
        #     "content": "Who is the goat in history of football, given the reasoning."
        # },
        # {
        #     "role": "user",
        #     "content": "What's the weather like in Boston today? Please respond with the format: Today's weather is :{function call result}",
        # }
    ],
    max_tokens=1024,
    # tools=tools,
    temperature=0.6,
    top_p=0.95,
    # top_k=40,
    # min_p=0.0,
    # repetition_penalty=1.0,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "football_schema",
            # convert the pydantic model to json schema
            "schema": BestPlayer.model_json_schema(),
        },
    },
)

response_content = response.choices[0].message.content
print(response_content)
# validate the JSON response by the pydantic model
capital_info = BestPlayer.model_validate_json(response_content)
print(f"Validated response: {capital_info.model_dump_json()}")


