import asyncio
from agentscope.agent import AgentBase
from agentscope.message import Msg

class MyAgent(AgentBase):

    def __init__(self, name: str):

        super().__init__()

        self.name = name

    async def reply(
        self,
        msg: Msg
    ) -> Msg:

        return Msg(
            name=self.name,
            role="assistant",
            content="Hello !"
        )

    async def handle_interrupt(
        self,
        *args,
        **kwargs
    ) -> Msg:

        return Msg(
            name=self.name,
            role="assistant",
            content="Interrupted"
        )

async def main():

    agent = MyAgent(name="Aadhya")

    msg = Msg(
        name="Sai",
        role="user",
        content="Hi"
    )

    res = await agent(msg)

    print(res.get_text_content())

if __name__ == "__main__":
    asyncio.run(main())