import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
from agentscope.memory import Mem0LongTermMemory
from agentscope.model import OpenAIChatModel
from agentscope.embedding import OpenAITextEmbedding
from agentscope.message import Msg

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY was not found. Check your .env path and value.")



long_term_memory = Mem0LongTermMemory(
    agent_name="Friday",
    user_name="user_123",
    model=OpenAIChatModel(
        model_name="gpt-4o-mini",
        api_key=api_key,
        stream=False,
    ),
    embedding_model=OpenAITextEmbedding(
        model_name="text-embedding-3-small",
        dimensions=1536,
        api_key=api_key,
    ),
    on_disk=False,
)

async def mem0_basic_usage():
    try:
        await long_term_memory.record(
            Msg(
                name="user",
                role="user",
                content="I like staying in homestays when traveling"
            )
        )

        results = await long_term_memory.retrieve(
            Msg(
                name="user",
                role="user",
                content="What are my accommodation preferences?"
            )
        )

        print(results)

    finally:
        try:
            long_term_memory.long_term_working_memory.close()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(mem0_basic_usage())