import asyncio
import os
from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.pipeline import MsgHub

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

model = OpenAIChatModel(
    model_name="gpt-4o-mini",
    api_key=api_key,
)

formatter = OpenAIChatFormatter()


def create_agent(name: str, role: str):
    return ReActAgent(
        name=name,
        sys_prompt=f"You are {name}. Your role is {role}. Introduce yourself briefly.",
        model=model,
        formatter=formatter,
    )


Aadhya = create_agent("Aadhya", "a Research scientist")
Saichaitanya = create_agent("Saichaitanya", "a software engineer")
Tara = create_agent("Tara", "a designer")


async def main():
    async with MsgHub(
        participants=[Aadhya, Saichaitanya, Tara],
        announcement=Msg(
            name="user",
            content="Introduce yourself in one sentence.",
            role="user",
        ),
    ):
        await Aadhya()
        await Saichaitanya()
        await Tara()


if __name__ == "__main__":
    asyncio.run(main())