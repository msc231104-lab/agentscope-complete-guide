import asyncio
import os
import subprocess
import urllib.request
from dotenv import load_dotenv
from agentscope.agent import ReActAgent
from agentscope.embedding import OpenAITextEmbedding
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.rag import PDFReader, QdrantStore, SimpleKnowledge
from agentscope.tool import Toolkit


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")


def download_file(url: str, path: str) -> None:
    """Download a file, falling back to curl if urllib fails."""
    try:
        urllib.request.urlretrieve(url, path)
    except Exception:
        subprocess.run(
            ["curl", "-fSL", "-o", path, url],
            check=True,
        )


paper_url = "https://arxiv.org/pdf/1706.03762"
paper_path = os.path.join(
    os.path.dirname(__file__),
    "attention_is_all_you_need.pdf",
)

if not os.path.exists(paper_path):
    print("Downloading Transformer paper...")
    download_file(paper_url, paper_path)
    print(f"Download complete: {paper_path}")
else:
    print(f"Paper already exists: {paper_path}")


model = OpenAIChatModel(
    api_key=OPENAI_API_KEY,
    model_name="gpt-4o-mini",
    stream=True,
)

embedding_model = OpenAITextEmbedding(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
    dimensions=1024,
)


async def build_knowledge() -> SimpleKnowledge:
    """Parse PDF and build a vector knowledge base."""

    vector_store = QdrantStore(
        location=":memory:",
        collection_name="transformer_paper",
        dimensions=1024,
    )

    knowledge = SimpleKnowledge(
        embedding_model=embedding_model,
        embedding_store=vector_store,
    )

    pdf_reader = PDFReader(
        chunk_size=3000,
        split_by="paragraph",
    )

    docs = await pdf_reader(pdf_path=paper_path)
    print(f"Paper split into {len(docs)} chunks")

    await knowledge.add_documents(docs)
    print("Knowledge base construction complete")

    return knowledge


async def main() -> None:
    knowledge = await build_knowledge()

    rag_toolkit = Toolkit()
    rag_toolkit.register_tool_function(
        knowledge.retrieve_knowledge,
        func_description=(
            "Retrieve relevant information from the 'Attention Is All You Need' paper "
            "knowledge base. Use this when the user asks about technical "
            "details such as Transformer architecture, Self-Attention, "
            "Multi-Head Attention, Positional Encoding, encoder, decoder, "
            "training setup, or computational complexity."
        ),
    )

    rag_agent = ReActAgent(
        name="ResearchAssistant",
        sys_prompt=(
            "You are a research assistant. "
            "You have access to a knowledge base built from the "
            "'Attention Is All You Need' Transformer paper. "
            "When the user asks a technical question about the paper, "
            "use the retrieve_knowledge tool first. "
            "Answer only using the retrieved paper content. "
            "If the answer is not found in the knowledge base, say so clearly."
        ),
        model=model,
        formatter=OpenAIChatFormatter(),
        memory=InMemoryMemory(),
        toolkit=rag_toolkit,
    )

    print("\nPDF RAG Research Assistant is ready.")
    print("Ask questions about the 'Attention Is All You Need' paper.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = await rag_agent(
            Msg("User", user_input, "user")
        )

        print("\nAssistant:")
        print(response.get_text_content())
        print()


if __name__ == "__main__":
    asyncio.run(main())