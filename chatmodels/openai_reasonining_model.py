import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

async def openai_example_reasoning():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found. Check your .env path and value.")

    model = OpenAIChatModel(
        # Use an OpenAI reasoning model to get ThinkingBlock output.
        model_name="o4-mini",
        api_key=api_key,
        stream=True,
        reasoning_effort="medium",

    )

    formatter = OpenAIChatFormatter()

    user_msg = Msg(
        name="Aadhya",
        role="user",
        content="what is 17*23 ?"
    )

    formatted_messages = await formatter.format([user_msg])

    res = await model(formatted_messages)


    last_chunk = None

    async for chunk in res:
        last_chunk = chunk

    if last_chunk is None:
        return

    saw_thinking = False
    for block in last_chunk.content:
        block_type = block["type"]
        if block_type == "thinking":
            saw_thinking = True
            content = block.get("thinking", "")
        else:
            content = block.get("text", "")
        print(f"[{block_type}] {content[:80]}...")

    if not saw_thinking:
        print("[info] No thinking block returned by this provider/model.")

if __name__ == "__main__":
    asyncio.run(openai_example_reasoning())

