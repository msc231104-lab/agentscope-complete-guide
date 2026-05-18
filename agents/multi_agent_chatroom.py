import asyncio
import os
from dotenv import load_dotenv
from agentscope.pipeline import ChatRoom
from agentscope.agent import RealtimeAgent
from agentscope.realtime import GeminiRealtimeModel, ClientEvents

load_dotenv()

async def main():
    agent1 = RealtimeAgent(
        name="Agent1",
        sys_prompt=(
            "You are Agent1. "
            "Reply with only the final answer. "
            "Use one short sentence. "
            "Do not include headings, analysis, reasoning, markdown, or explanations."
        ),
        model=GeminiRealtimeModel(
            model_name="gemini-2.5-flash-native-audio-preview-09-2025",
            api_key=os.getenv("GEMINI_API_KEY"),
        ),
    )

    agent2 = RealtimeAgent(
        name="Agent2",
        sys_prompt=(
            "You are Agent2. "
            "Reply with only the final answer. "
            "Use one short sentence. "
            "Do not include headings, analysis, reasoning, markdown, or explanations."
        ),
        model=GeminiRealtimeModel(
            model_name="gemini-2.5-flash-native-audio-preview-09-2025",
            api_key=os.getenv("GEMINI_API_KEY"),
        ),
    )

    chat_room = ChatRoom(agents=[agent1, agent2])
    outgoing_queue = asyncio.Queue()

    await chat_room.start(outgoing_queue)

    async def print_events():
        responses = {}

        while True:
            event = await outgoing_queue.get()
            data = event.model_dump()

            event_type = str(data.get("type"))
            agent_name = data.get("agent_name", "Unknown Agent")
            delta = data.get("delta", "")

            if agent_name not in responses:
                responses[agent_name] = ""

            if "READY" in event_type or "ready" in event_type:
                print(f"{agent_name}: ready")

            elif "TRANSCRIPT_DELTA" in event_type or "transcript_delta" in event_type:
                text = delta.strip()

                if not text:
                    continue

                if text.startswith("**"):
                    continue

                if "reasoning" in text.lower():
                    continue

                responses[agent_name] += text + " "

            elif "TEXT_DELTA" in event_type or "text_delta" in event_type:
                text = delta.strip()

                if not text:
                    continue

                if text.startswith("**"):
                    continue

                if "reasoning" in text.lower():
                    continue

                responses[agent_name] += text + " "

            elif "RESPONSE_DONE" in event_type or "response_done" in event_type:
                final_response = " ".join(responses[agent_name].split())

                if final_response:
                    print(f"{agent_name}: {final_response}")

                print(f"{agent_name}: response completed")
                responses[agent_name] = ""

            elif "ERROR" in event_type or "error" in event_type:
                print(f"{agent_name}: ERROR -> {data}")

    printer_task = asyncio.create_task(print_events())

    text_event = ClientEvents.from_json({
        "type": "client_text_append",
        "session_id": "session1",
        "text": (
            "Hello everyone! Introduce yourself in one short sentence only. "
            "Do not include reasoning, headings, markdown, or explanations."
        )
    })

    await chat_room.handle_input(text_event)

    await asyncio.sleep(10)

    printer_task.cancel()
    await chat_room.stop()


if __name__ == "__main__":
    asyncio.run(main())