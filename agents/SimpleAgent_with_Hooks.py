import asyncio
from typing import Any
from agentscope.agent import AgentBase
from agentscope.message import Msg

class MyAgent(AgentBase):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    async def reply(self, msg: Msg) -> Msg:
        return Msg(
            name=self.name,
            role="assistant",
            content=f"Received: {msg.content}",
        )

def my_pre_hook(
    self:AgentBase,
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    print("Pre Hook Executed")
    return kwargs

def my_post_hook(
    self: AgentBase,
    kwargs: dict[str, Any],
    output: Any,
) -> Any | None:
    print("Post Hook Executed")
    return output

async def main():
    agent = MyAgent(name="Aadhya")

    agent.register_instance_hook(
        hook_type="pre_reply",
        hook_name="my_pre_hook",
        hook=my_pre_hook,
    )

    agent.register_instance_hook(
        hook_type="post_reply",
        hook_name="my_post_hook",
        hook=my_post_hook,
    )

    msg = Msg(
        name="Sai",
        role="user",
        content="Hello",
    )

    response = await agent.reply(msg)
    print(response.get_text_content())

if __name__ == "__main__":
    asyncio.run(main())