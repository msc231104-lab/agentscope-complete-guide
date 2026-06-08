import asyncio
import os
from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg
from agentscope.tool import Toolkit
from agentscope.rag import TextReader, SimpleKnowledge, QdrantStore
from agentscope.embedding import OpenAITextEmbedding

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

async def build_knowledge_base() -> SimpleKnowledge:


    reader = TextReader(chunk_size=512, split_by="paragraph")

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


async def example_agentic_manner() -> None:

    knowledge = await build_knowledge_base()

    toolkit = Toolkit()

    agent = ReActAgent(
        name="Friday",
        sys_prompt=(
            "You are a helpful assistant named Friday. "
            "Use conversation history to resolve pronouns and vague references. "
            "When the user asks about Aadhya Shrivastava or someone related to her, "
            "use the knowledge retrieval tool."
        ),
        model=OpenAIChatModel(
            api_key=openai_api_key,
            model_name="gpt-4o-mini",
        ),
        formatter=OpenAIChatFormatter(),
        toolkit=toolkit,
    )

    # First turn — introduce context without RAG/tool usage.
    await agent(
        Msg(
            "user",
            "Aadhya Shrivastava is my best friend.",
            "user",
        )
    )

    # Register RAG tool after the agent already has conversational context.
    toolkit.register_tool_function(
        knowledge.retrieve_knowledge,
        func_description=(
            "Retrieve documents relevant to the given query. "
            "Use this tool when you need to find information about Aadhya Shrivastava, "
            "her family, her hobbies, her work, her goals, or people related to her. "
            "Before calling this tool, rewrite vague references and pronouns using "
            "conversation history. For example, if the user says 'her father', "
            "search for 'Aadhya Shrivastava father'."
        ),
    )

    # Second turn — agent should resolve "her" as Aadhya Shrivastava,
    # then call retrieve_knowledge with a rewritten query.
    response = await agent(
        Msg(
            "user",
            "Do you know who her father is?",
            "user",
        )
    )

    print("\nFinal response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(example_agentic_manner())