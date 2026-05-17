import asyncio
import os
from dotenv import load_dotenv
from agentscope.agent import ReActAgent , UserAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.plan import PlanNotebook

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

plan_notebook = PlanNotebook()

agent = ReActAgent(
    name="Aadhya",
    sys_prompt="Your name is Aadhya. You are a helpful assistant.",
    model= OpenAIChatModel(
        model_name="gpt-4o-mini",
        api_key=api_key,
    ),
    formatter=OpenAIChatFormatter(),
    plan_notebook= plan_notebook,
)

async def interact_with_agent() -> None:
    user = UserAgent(name="Sai")
    msg = None
    while True:
        msg = await user(msg)
        if msg.get_text_content() == "exit":
            print("Exiting the conversation.")
            break
        msg = await agent(msg)

if __name__ == "__main__":
    asyncio.run(interact_with_agent())