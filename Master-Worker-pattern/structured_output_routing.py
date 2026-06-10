import asyncio
import os
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")


model = OpenAIChatModel(
    model_name="gpt-4o-mini",
    api_key=api_key,
)

formatter = OpenAIChatFormatter()


class RoutingChoice(BaseModel):
    route: Literal["programming", "content", "search"]


router = ReActAgent(
    name="MasterAgent",
    sys_prompt="""
You are a routing classifier.

Choose ONLY one route:
- programming
- content
- search

Do not answer the user's question.
Only classify the query.
""",
    model=model,
    formatter=formatter,
)


async def main():
    user_query = "Tell me who is virat kholi"

    user_msg = Msg(
        name="user",
        content=user_query,
        role="user",
    )

    route = await router(
        user_msg,
        structured_model=RoutingChoice,
    )

    selected_route = route.metadata["route"]

    worker_names = {
        "programming": "ProgrammingWorker",
        "content": "ContentWorker",
        "search": "SearchWorker",
    }

    print(f"User Query: {user_query}")
    print(f"MasterAgent selected route: {selected_route}")
    print(f"Routing to: {worker_names[selected_route]}")


if __name__ == "__main__":
    asyncio.run(main())