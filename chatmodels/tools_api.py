import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

async def openai_example_tools():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found. Check your .env path and value.")

    model = OpenAIChatModel(
        model_name="gpt-4o-mini",
        api_key=api_key,
        stream=False,
    )

    formatter = OpenAIChatFormatter()

    json_schema = [
                {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search for a query on Google.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    user_msg = Msg(
        name="Aadhya",
        role="user",
        content="Search AgentScope release notes.",
    )

    formatted_messages = await formatter.format([user_msg])

    response = await model(
        messages=formatted_messages,
        tools=json_schema,
        tool_choice="auto",
    )

    print(response.content)

if __name__ == "__main__":
    asyncio.run(openai_example_tools())