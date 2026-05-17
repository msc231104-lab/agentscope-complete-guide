import asyncio
import os
from dotenv import load_dotenv
from typing import Any
from agentscope.agent import AgentBase , ReActAgent
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

agent = ReActAgent(
    name="Aadhya",
    sys_prompt= "Your name is Aadhya, you are a helpful assistant.",
    model = OpenAIChatModel(
        model_name="gpt-3.5-turbo",
        api_key=api_key,
    ),
    formatter= OpenAIChatFormatter()
)


def add_prefix_hook(
    self:AgentBase,
    kwargs:dict[str,Any]
) -> dict[str,Any] | None:
    msg = kwargs["msg"]
    msg.content = "[Reviewed] " + msg.content

    print("Modified input:", msg.content)


    return {
        **kwargs,
        "msg": msg
    }


agent.register_instance_hook(
    hook_type="pre_reply",
    hook_name="add_prefix",
    hook=add_prefix_hook
)

async def main():
    msg = Msg(
        name="Sai",
        role="user",
        content="Print this exact sentence only: test message"    )

    response = await agent(msg)

    agent.remove_instance_hook(
        hook_type="pre_reply",
        hook_name="add_prefix"
    )

if __name__ == "__main__":
    asyncio.run(main())