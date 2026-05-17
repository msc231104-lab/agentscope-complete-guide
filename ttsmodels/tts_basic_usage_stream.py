import asyncio
import base64
import os
import wave
from dotenv import load_dotenv
from pathlib import Path
from agentscope.tts import OpenAITTSModel
from agentscope.message import Msg

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

async def openai_example_tts():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found. Check your .env path and value.")
    tts_model = OpenAITTSModel(
        model_name="gpt-4o-mini-tts",
        api_key=api_key,
        voice="coral",
        stream=True,
    )

    msg = Msg(
        name="Aadhya",
        role="assistant",
        content="this is a TTS demo. My name is Aadhya. How are you doing today ?"
    )

    async for tts_response in await tts_model.synthesize(msg):
        print("Received audio chunk:", len(tts_response.content["source"]["data"]))

if __name__ == "__main__":
    asyncio.run(openai_example_tts())