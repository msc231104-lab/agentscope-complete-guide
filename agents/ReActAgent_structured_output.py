import asyncio
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.formatter import OpenAIChatFormatter

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

class personInfo(BaseModel):

    name: str = Field(
        description="The person's name"
    )

    description: str = Field(
        description="A one-sentence description"
    )

    age: int = Field(
        description="The person's age"
    )

    honors: list[str] = Field(
        description="A list of honors"
    )

model = OpenAIChatModel(
    model_name="gpt-4o-mini",
    api_key=api_key,
)

formatter = OpenAIChatFormatter()

# Create Agent
agent = ReActAgent(
       name="Assistant",
       model=model,
       formatter=formatter,
       sys_prompt="You are a helpful assistant.",
    )

async def main():
    response = await agent(

        Msg(
            name="user",
            content="Introduce Albert Einstein",
            role="user",
        ),

        structured_model=personInfo,
    )

    # Print Metadata
    print(response.metadata)

if __name__ == "__main__":
    asyncio.run(main())


