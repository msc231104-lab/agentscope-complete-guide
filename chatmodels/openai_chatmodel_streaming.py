import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

async def openai_example_streaming():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found. Check your .env path and value.")

    model = OpenAIChatModel(
        model_name="gpt-4o-mini",
        api_key=api_key,
        stream=True)

    formatter = OpenAIChatFormatter()

    user_msg = Msg(
        name="Aadhya",
        role="user",
        content="Count from 1 to 5"
    )

    formatted_messages = await formatter.format([user_msg])

    generator = await model(formatted_messages)

    async for chunk in generator:
        print("Chunk:", chunk.content)


if __name__ == "__main__":
    asyncio.run(openai_example_streaming())
