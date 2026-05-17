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
        stream=False,
    )

    msg = Msg(
        name="Aadhya",
        role="assistant",
        content="this is a TTS demo. My name is Aadhya. How are you doing today ?"
    )

    tts_response = await tts_model.synthesize(msg)
    if not tts_response.content:
        raise RuntimeError("TTS returned empty audio content.")

    audio_b64 = tts_response.content["source"]["data"]
    audio_bytes = base64.b64decode(audio_b64)
    print("Audio bytes length:", len(audio_bytes))

    # Save audio as WAV file
    output_path = Path(__file__).parent / "output.wav"
    with wave.open(str(output_path), 'wb') as wav_file:
        # OpenAI PCM is mono, 24kHz, 16-bit
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(audio_bytes)

    print(f"Audio saved to: {output_path}")

if __name__ == "__main__":
    asyncio.run(openai_example_tts())