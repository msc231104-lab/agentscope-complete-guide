import asyncio
import os
from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.tool import Toolkit, ToolResponse

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

model = OpenAIChatModel(
    model_name="gpt-4o-mini",
    api_key=api_key,
)

formatter = OpenAIChatFormatter()


# --------------------------------------------------
# Agent-as-Tool
# --------------------------------------------------
async def create_worker(task: str) -> ToolResponse:
    """
    Create a worker agent and let it handle the task.
    """

    worker = ReActAgent(
        name="WorkerAgent",
        sys_prompt="You are a worker agent.",
        model=model,
        formatter=formatter,
    )

    response = await worker(
        Msg(
            name="user",
            content=task,
            role="user",
        )
    )

    return ToolResponse(
        content=f"WorkerAgent executed task: {task}"
    )


# --------------------------------------------------
# Register Tool
# --------------------------------------------------
toolkit = Toolkit()
toolkit.register_tool_function(create_worker)


# --------------------------------------------------
# Master Agent
# --------------------------------------------------
master = ReActAgent(
    name="MasterAgent",
    sys_prompt="""
You are a master agent.

Use create_worker whenever a task needs to be executed.

Do not solve the task yourself.
""",
    model=model,
    formatter=formatter,
    toolkit=toolkit,
)


# --------------------------------------------------
# Main
# --------------------------------------------------
async def main():

    user_msg = Msg(
        name="user",
        content="Write a Python hello world program",
        role="user",
    )

    await master(user_msg)


if __name__ == "__main__":
    asyncio.run(main())