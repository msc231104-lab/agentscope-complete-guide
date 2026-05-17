import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.formatter import OpenAIChatFormatter
from agentscope.token import CharTokenCounter


class CustomSummary(BaseModel):

    main_topic: str = Field(
        max_length=200,
        description="The main topic of the conversation"
    )

    key_points: str = Field(
        max_length=400,
        description="Important points discussed"
    )

    pending_tasks: str = Field(
        max_length=200,
        description="Tasks that remain to be done"
    )

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found")


model = OpenAIChatModel(
    model_name="gpt-4o-mini",
    api_key=api_key,
)

formatter = OpenAIChatFormatter()

agent = ReActAgent(
    name="Assistant",
    sys_prompt="You are a helpful assistant.",
    model=model,
    formatter=formatter,

    compression_config=ReActAgent.CompressionConfig(

        enable=True,

        agent_token_counter=CharTokenCounter(),

        # Small threshold for demo
        trigger_threshold=50,

        keep_recent=1,

        summary_schema=CustomSummary,

        compression_prompt=(
            "<system-hint>"
            "Summarize the conversation focusing on "
            "the main topic, key points, and pending tasks."
            "</system-hint>"
        ),

        summary_template=(
            "<system-info>Summary:\n"
            "Main Topic: {main_topic}\n\n"
            "Key Points:\n{key_points}\n\n"
            "Pending Tasks:\n{pending_tasks}"
            "</system-info>"
        ),
    ),
)

async def main():

    msg1 = Msg(
        name="Sai",
        role="user",
        content="Explain Artificial Intelligence.",
    )

    await agent(msg1)

    msg2 = Msg(
        name="Sai",
        role="user",
        content="Explain Machine Learning.",
    )

    await agent(msg2)


if __name__ == "__main__":
    asyncio.run(main())
