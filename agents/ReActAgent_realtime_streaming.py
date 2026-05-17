import asyncio
import os
from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.formatter import OpenAIChatFormatter

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found")


model = OpenAIChatModel(
    model_name="gpt-4.1-mini",
    api_key=api_key,
)


formatter = OpenAIChatFormatter()


agent = ReActAgent(
    name="Assistant",
    model=model,
    formatter=formatter,
    sys_prompt=(
        "You are a helpful AI assistant. "
        "Think carefully before answering."
    )
)

async def main():
    msg = Msg(
        name= "Aadhya",
        role="user",
        content=(
            "write a detailed explaination about the evolution of AI to Agentic AI."
        )
    )


    print("Starting agent...\n")

    reply_task = asyncio.create_task(agent(msg))

    await asyncio.sleep(3)

    print("Interrupting agent...\n")

    await agent.interrupt()

    result = await reply_task

    print("Final result:\n")

    print(result.get_text_content())

if __name__ == "__main__":
    asyncio.run(main())