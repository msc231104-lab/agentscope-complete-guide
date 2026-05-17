import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
from agentscope.embedding import OpenAITextEmbedding

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY was not found. Check your .env path and value.")

async def example_text_embedding():
    embedding_model = OpenAITextEmbedding(
        model_name="text-embedding-3-small",
        api_key=api_key
    )

    texts = [
        "Hello My name is Aadhya",
        "What is your name?"
    ]

    response = await embedding_model(texts)

    print(f'Generated {len(response.embeddings)} embeddings:')
    print(f"Embedding dimension: {len(response.embeddings[0])}")
    print(f"Tokens used: {response.usage.tokens}")

if __name__ == "__main__":
    asyncio.run(example_text_embedding())


