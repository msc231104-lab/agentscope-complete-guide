import asyncio
import os
from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.message import Msg
from agentscope.pipeline import sequential_pipeline

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
    sys_prompt="Summarize the given topic in one sentence.",
    model=model,
    formatter=formatter,
)

Saichaitanya = ReActAgent(
    name="Saichaitanya",
    sys_prompt="Expand the received summary into two sentences.",
    model=model,
    formatter=formatter,
)

Tara = ReActAgent(
    name="Tara",
    sys_prompt="Convert the received text into a concise final explanation.",
    model=model,
    formatter=formatter,
)


async def main():

    initial_msg = Msg(
        name="user",
        content="Artificial Intelligence is transforming healthcare.",
        role="user",
    )

    result = await sequential_pipeline(
        agents=[Aadhya, Saichaitanya, Tara],
        msg=initial_msg,
    )

    print("\nFinal Output:")
    print(result.content)


if __name__ == "__main__":
    asyncio.run(main())