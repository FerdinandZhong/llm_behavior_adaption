from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI(base_url=f"http://127.0.0.1:30000/v1", api_key="None")

# Define the schema using Pydantic
class CapitalInfo(BaseModel):
    name: str = Field(..., pattern=r"^\w+$", description="Name of the capital city")
    # mayer: str = Field(..., description="Mayer of the capital city")


response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Please generate the information of the capital of France in the JSON format.",
        },
    ],
    temperature=0,
    max_tokens=1024,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "foo",
            # convert the pydantic model to json schema
            "schema": CapitalInfo.model_json_schema(),
        },
    },
)

response_content = response.choices[0].message.content
print(response_content)
# validate the JSON response by the pydantic model
capital_info = CapitalInfo.model_validate_json(response_content)
print(f"Validated response: {capital_info.model_dump_json()}")