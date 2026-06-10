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


async def programming_worker(query: str) -> ToolResponse:
    """Use this tool for coding, debugging, programming, or software tasks."""
    return ToolResponse(content="ROUTE_SELECTED: ProgrammingWorker")


async def content_worker(query: str) -> ToolResponse:
    """Use this tool for writing, drafting, summarizing, or content generation tasks."""
    return ToolResponse(content="ROUTE_SELECTED: ContentWorker")


async def search_worker(query: str) -> ToolResponse:
    """Use this tool for factual questions, research, people, places, or search-related tasks."""
    return ToolResponse(content="ROUTE_SELECTED: SearchWorker")


toolkit = Toolkit()

toolkit.register_tool_function(programming_worker)
toolkit.register_tool_function(content_worker)
toolkit.register_tool_function(search_worker)


router = ReActAgent(
    name="MasterAgent",
    sys_prompt="""
You are a master routing agent.

Your only job is to choose the correct worker tool.

Rules:
- For coding/programming tasks, call programming_worker.
- For writing/content tasks, call content_worker.
- For factual/search/research tasks, call search_worker.

Call exactly one tool.
Do not solve the user's task.
""",
    model=model,
    formatter=formatter,
    toolkit=toolkit,
    max_iters=1,
)


async def main():
    user_query = "Write a Python function to reverse a string"

    user_msg = Msg(
        name="user",
        content=user_query,
        role="user",
    )

    print(f"User Query: {user_query}")
    await router(user_msg)


if __name__ == "__main__":
    asyncio.run(main())