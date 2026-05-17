import asyncio
import os
from pathlib import Path
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from dotenv import load_dotenv
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)


async def openai_model_call():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found. Check your .env path and value.")

    model = OpenAIChatModel(
        model_name="gpt-4o-mini",
        api_key=api_key,
        stream=False)

    formatter = OpenAIChatFormatter()

    user_msg = Msg(
        name="Aadhya",
        role="user",
        content="Hii !"
    )

    formatted_messages = await formatter.format([user_msg])

    res = await model(formatted_messages)

    print("response:", res.content)
    print("Usage:", res.usage)

if __name__ == "__main__":
    asyncio.run(openai_model_call())


