import asyncio
import os
from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.pipeline import fanout_pipeline

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

model = OpenAIChatModel(
    model_name="gpt-4o-mini",
    api_key=api_key,
)

formatter = OpenAIChatFormatter()


Aadhya = ReActAgent(
    name="Aadhya",
    sys_prompt="Analyze the topic from a teacher's perspective in one sentence",
    model=model,
    formatter=formatter,
)

Saichaitanya = ReActAgent(
    name="Saichaitanya",
    sys_prompt="Analyze the topic from a software engineer's perspective in one sentence.",
    model=model,
    formatter=formatter,
)

Tara = ReActAgent(
    name="Tara",
    sys_prompt="Analyze the topic from a designer's perspective in one sentence.",
    model=model,
    formatter=formatter,
)


async def main():

    input_msg = Msg(
        name="user",
        content="Artificial Intelligence is transforming healthcare.",
        role="user",
    )

    results = await fanout_pipeline(
        agents=[Aadhya, Saichaitanya, Tara],
        msg=input_msg,
        enable_gather=True,
    )

    print("\nFanout Results:\n")

    for response in results:
        print(f"{response.name}: {response.content}\n")


if __name__ == "__main__":
    asyncio.run(main())