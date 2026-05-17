import asyncio
import os
from dotenv import load_dotenv
from agentscope.agent import RealtimeAgent
from agentscope.realtime import OpenAIRealtimeModel

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")

agent = RealtimeAgent(
    name="Aadhya",
    sys_prompt="You are a helpful assistant named Aadhya.",
    model= OpenAIRealtimeModel(
        model_name="gpt-realtime-1.5",
        api_key=api_key,
        voice="alloy",  # Optional: specify a voice for text-to-speech
        enable_input_audio_transcription=True
    )
)


async def main():
    outgoing_queue = asyncio.Queue()

    async def handle_events():
        while True:
            event = await outgoing_queue.get()
            print(f'Event: {event.type}')

    asyncio.create_task(handle_events())

    await agent.start(outgoing_queue)

    print("\nRealtime agent Started \n")

    await asyncio.sleep(20)  # Keep the agent running for 20 seconds

    await agent.stop()

    print("\nRealtime agent Stopped \n")

if __name__ == "__main__":
    asyncio.run(main())