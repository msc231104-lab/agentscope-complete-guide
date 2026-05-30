import asyncio
import json
from agentscope.rag import TextReader, Document

async def example_text_reader() -> list[Document]:
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

    print("Number of chunks:",len(documents))
    for idx , doc in enumerate(documents):
        print(f"Document #{idx}")
        print("Score:",doc.score)
        print("Metadata:",json.dumps(doc.metadata, indent=2))
    return documents

if __name__ == "__main__":
    docs = asyncio.run(example_text_reader())

