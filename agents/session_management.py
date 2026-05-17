import asyncio
import os
from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.session import JSONSession
from agentscope.memory import InMemoryMemory

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")


session =JSONSession(
    save_dir="./sessions",
)


agent = ReActAgent(
    name="Aadhya",
    sys_prompt="Your name is Aadhya, you are a helpful assistant.",
    model = OpenAIChatModel(
        model_name="gpt-3.5-turbo",
        api_key=api_key,
    ),
    formatter= OpenAIChatFormatter(),
    memory= InMemoryMemory()
)

async def main():
    await agent(
        Msg(
            name="Sai",
            role="user",
            content="My favorite color is blue."
        )

    )

    await session.save_session_state(
        session_id="user_1",
        agent=agent
    )

    print("\n Session Saved Successfully \n")

    await session.load_session_state(
        session_id="user_1",
        agent=agent
    )

    print("\n Session Loaded Successfully \n")

    await agent(
        Msg(
            name="Sai",
            role="user",
            content="What is my favorite color?"
        )
    )

if __name__ == "__main__":
    asyncio.run(main())