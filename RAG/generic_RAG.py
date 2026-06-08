import asyncio
import os
from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.rag import TextReader, SimpleKnowledge, QdrantStore
from agentscope.embedding import OpenAITextEmbedding


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

async def build_knowledge_base() -> SimpleKnowledge:

    reader = TextReader(
        chunk_size=512,
        split_by="paragraph",
    )

    documents = await reader(
        text=(
            "I'm Aadhya Shrivastava, 22 years old.\n\n"
            "I live in Bbsr. I work at NIC as a junior AI developer. "
            "I love to watch anime and read manga, and I love animals. "
            "I also love hiking and traveling. "
            "I have a pet dog named Xenn.\n\n"
            "My father's name is Rajesh Shrivastava, a retired employee of NMDC. "
            "I'm very proud of my father.\n\n"
            "My mother is a homemaker. "
            "She is very caring and loving.\n\n"
            "I wish to become a successful person in research and development "
            "in the field of geospace and space weather. "
            "I want to contribute to the field of space research and make a "
            "positive impact on the world.\n\n"
            "My best friend is Aishu. "
            "We have been friends since childhood."
        )
    )

    knowledge = SimpleKnowledge(
        embedding_model=OpenAITextEmbedding(
            api_key=openai_api_key,
            model_name="text-embedding-3-small",
            dimensions=1536,
        ),
        embedding_store=QdrantStore(
            location=":memory:",
            collection_name="knowledge_base",
            dimensions=1536,
        ),
    )

    await knowledge.add_documents(documents)

    return knowledge


async def example_generic_manner() -> None:

    knowledge = await build_knowledge_base()

    agent = ReActAgent(
        name="Friday",
        sys_prompt=(
            "You are a helpful assistant named Friday. "
            "Answer questions using the provided knowledge base when relevant."
        ),
        model=OpenAIChatModel(
            api_key=openai_api_key,
            model_name="gpt-4o-mini",
        ),
        formatter=OpenAIChatFormatter(),
        knowledge=knowledge,
    )

    response = await agent(
        Msg(
            "user",
            "Do you know who Aadhya Shrivastava's father is?",
            "user",
        )
    )

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(example_generic_manner())