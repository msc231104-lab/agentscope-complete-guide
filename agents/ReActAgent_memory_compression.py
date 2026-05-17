import asyncio
import os
from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.formatter import OpenAIChatFormatter
from agentscope.token import CharTokenCounter


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found")



model = OpenAIChatModel(
    model_name="gpt-4o-mini",
    api_key=api_key,
)



formatter = OpenAIChatFormatter()



agent = ReActAgent(
    name="Assistant",
    model=model,
    formatter=formatter,
    sys_prompt="You are a helpful assistant.",

    compression_config=ReActAgent.CompressionConfig(
        enable=True,
        agent_token_counter=CharTokenCounter(),
        trigger_threshold=50,
        keep_recent=1,
    ),
)


async def main():

    msg1 = Msg(
        name="Sai",
        role="user",
        content="What is Artificial Intelligence?",
    )

    await agent(msg1)

    msg2 = Msg(
        name="Sai",
        role="user",
        content="Explain Machine Learning.",
    )

    await agent(msg2)



if __name__ == "__main__":
    asyncio.run(main())