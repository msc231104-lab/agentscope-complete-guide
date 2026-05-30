import asyncio
import os
from agentscope.rag import TextReader, SimpleKnowledge , QdrantStore
from agentscope.embedding import OpenAITextEmbedding
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

async def build_knowledge_base() -> SimpleKnowledge:
    reader = TextReader(chunk_size=512 , split_by="paragraph")
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
            api_key=api_key,
            model_name="text-embedding-3-small",
            dimensions= 1536
        ),
        embedding_store = QdrantStore(
            location=":memory:",
            collection_name="knowledge_base",
            dimensions= 1536
        )
    )

    await knowledge.add_documents(documents)

    results = await knowledge.retrieve(
        query="What is the name of Aadhya Shrivastava's father?",
        limit = 1 ,
        score_threshold =0.3
    )

    for doc in results:
        print(doc)
    return knowledge

if __name__ == "__main__":
    knowledge = asyncio.run(build_knowledge_base())



