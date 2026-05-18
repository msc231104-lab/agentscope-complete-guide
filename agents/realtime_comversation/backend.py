import asyncio
import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from agentscope.agent import RealtimeAgent
from agentscope.realtime import GeminiRealtimeModel, ClientEvents

load_dotenv()

app = FastAPI()


@app.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    await websocket.accept()

    frontend_queue = asyncio.Queue()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY was not found in environment variables.")

    model = GeminiRealtimeModel(
        model_name="gemini-2.5-flash-native-audio-preview-12-2025",
        api_key=gemini_api_key,
    )

    agent = RealtimeAgent(
        name="NEXAI",
        sys_prompt="Your name is NEXAI and you are a helpful assistant.",
        model=model,
    )

    await agent.start(frontend_queue)

    async def send_to_frontend():
        while True:
            msg = await frontend_queue.get()
            await websocket.send_json(msg.model_dump())

    task = asyncio.create_task(send_to_frontend())

    try:
        while True:
            data = await websocket.receive_json()
            client_event = ClientEvents.from_json(data)
            await agent.handle_input(client_event)

    except WebSocketDisconnect:
        print(f"Client disconnected: {user_id}, {session_id}")

    finally:
        task.cancel()
        await agent.stop()