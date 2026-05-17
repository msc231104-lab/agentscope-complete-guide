import asyncio
import os
from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.plan import PlanNotebook, SubTask
from agentscope.message import Msg

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

plan_notebook = PlanNotebook()

agent = ReActAgent(
    name="Aadhya",
    sys_prompt="You are a helpful assistant.",
    model=OpenAIChatModel(
        model_name="gpt-4o-mini",
        api_key=api_key,
    ),
    formatter=OpenAIChatFormatter(),
    plan_notebook=plan_notebook,
)


async def main():
    await plan_notebook.create_plan(
        name="Research on Agent",
        description=(
            "Conduct a comprehensive research "
            "on the LLM-empowered agent."
        ),
        expected_outcome=(
            "A Markdown report about AI agents."
        ),
        subtasks=[
            SubTask(
                name="Search papers",
                description="Search agent papers.",
                expected_outcome="Paper list",
            ),
            SubTask(
                name="Summarize papers",
                description="Summarize findings.",
                expected_outcome="Summary",
            ),
        ],
    )

    msg = await plan_notebook.get_current_hint()
    print(f"{msg.name}: {msg.content}")

    response = await agent(
        Msg(
            name="user",
            content="What is the first step to complete the plan?",
            role="user",
        ),
    )

    print("\n Assistant:")
    print(response.get_text_content())

if __name__ == "__main__":
    asyncio.run(main())