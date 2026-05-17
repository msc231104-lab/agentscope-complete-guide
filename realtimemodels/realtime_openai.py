import asyncio
import base64
import json
import os
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import sounddevice as sd
from agentscope.agent import RealtimeAgent
from agentscope.realtime import OpenAIRealtimeModel
from agentscope.realtime import ClientEvents, ServerEvents

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY was not found. Check your .env path and value.")

real_time_model = OpenAIRealtimeModel(
        model_name="gpt-4o-realtime-preview",
        api_key=api_key,
        voice='coral',
    )

agent = RealtimeAgent(
    name="VoiceAssistant",
    sys_prompt=(
        "You are a helpful voice assistant named Aadhya. "
        "Always respond in English unless the user explicitly asks for another language."
    ),
    model=real_time_model

)


def _install_event_name_compat(model: OpenAIRealtimeModel) -> None:
    original_parse = model.parse_api_message
    event_type_map = {
        "response.audio.delta": "response.output_audio.delta",
        "response.audio.done": "response.output_audio.done",
        "response.audio_transcript.delta": "response.output_audio_transcript.delta",
        "response.audio_transcript.done": "response.output_audio_transcript.done",
    }

    async def _patched_parse(message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return await original_parse(message)

        if isinstance(data, dict):
            event_type = data.get("type")
            remapped_type = event_type_map.get(event_type)
            if remapped_type:
                data["type"] = remapped_type
                message = json.dumps(data)

        return await original_parse(message)

    model.parse_api_message = _patched_parse


def _install_session_config_compat(model: OpenAIRealtimeModel) -> None:
    original_build = model._build_session_config

    def _patched_build(instructions: str, tools: list[dict] | None, **kwargs):
        payload = original_build(instructions, tools, **kwargs)
        session = payload.get("session", {})

        if isinstance(session, dict):
            session.pop("type", None)
            session.pop("audio", None)

            if "output_modalities" in session and "modalities" not in session:
                session["modalities"] = session.pop("output_modalities")

            if session.get("modalities") == ["audio"]:
                session["modalities"] = ["audio", "text"]

            if "voice" not in session:
                session["voice"] = model.voice

        return payload

    model._build_session_config = _patched_build


_install_session_config_compat(real_time_model)
_install_event_name_compat(real_time_model)

async def _response_consumer(queue: asyncio.Queue):
    audio_chunks = []
    audio_rate = None

    while True:
        event = await queue.get()

        if isinstance(event, ServerEvents.AgentResponseCreatedEvent):
            audio_chunks = []
            audio_rate = None

        elif isinstance(event, ServerEvents.AgentResponseAudioDeltaEvent):
            audio_chunks.append(base64.b64decode(event.delta))
            if audio_rate is None:
                audio_rate = event.format.rate

        elif isinstance(event, ServerEvents.AgentResponseDoneEvent):
            if audio_chunks and audio_rate:
                print("Speaking...", flush=True)
                pcm_bytes = b"".join(audio_chunks)
                audio_array = np.frombuffer(pcm_bytes, dtype=np.int16)
                sd.play(audio_array, samplerate=audio_rate)
                await asyncio.to_thread(sd.wait)

        elif isinstance(event, ServerEvents.AgentErrorEvent):
            print(f"[error] {event.code}: {event.message}")


async def _trigger_response() -> None:
    # Current RealtimeAgent forwarding handles append events, but does not
    # forward ClientResponseCreateEvent to the model. Trigger response
    # directly on the model websocket.
    websocket = getattr(agent.model, "_websocket", None)
    if websocket is None:
        raise RuntimeError("Model websocket is not connected.")

    await websocket.send(
        json.dumps(
            {
                "type": "response.create",
                "response": {
                    "modalities": ["audio", "text"],
                    "voice": "coral",
                },
            }
        )
    )


async def _commit_audio_buffer() -> None:
    websocket = getattr(agent.model, "_websocket", None)
    if websocket is None:
        raise RuntimeError("Model websocket is not connected.")

    await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))


async def _append_audio_buffer(encoded_audio: str) -> None:
    websocket = getattr(agent.model, "_websocket", None)
    if websocket is None:
        raise RuntimeError("Model websocket is not connected.")

    await websocket.send(
        json.dumps(
            {
                "type": "input_audio_buffer.append",
                "audio": encoded_audio,
            }
        )
    )


def _record_audio_base64(sample_rate: int, duration_sec: float) -> str:
    frames = int(sample_rate * duration_sec)
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    pcm_bytes = audio.tobytes()

    # 16-bit mono PCM -> 2 bytes per sample; require at least 100ms.
    min_bytes = int(sample_rate * 0.1) * 2
    if len(pcm_bytes) < min_bytes:
        return ""

    return base64.b64encode(pcm_bytes).decode("ascii")


async def _apply_session_update_fix() -> None:
    websocket = getattr(agent.model, "_websocket", None)
    if websocket is None:
        raise RuntimeError("Model websocket is not connected.")

    await websocket.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "voice": "coral",
                    "instructions": agent.sys_prompt,
                },
            }
        )
    )


async def main():
    outgoing_queue = asyncio.Queue()
    await agent.start(outgoing_queue)
    await _apply_session_update_fix()

    # Wait for agent session to be ready before sending input.
    while True:
        event = await outgoing_queue.get()
        if isinstance(event, ServerEvents.AgentReadyEvent):
            print("[Agent ready]")
            break

    consumer_task = asyncio.create_task(_response_consumer(outgoing_queue))
    input_sample_rate = int(getattr(real_time_model, "input_sample_rate", 24000))
    record_seconds = 4.0
    print("Press Enter to speak. Type /exit to quit.")

    try:
        while True:
            user_cmd = await asyncio.to_thread(input, "")
            if user_cmd.strip().lower() in {"/exit", "exit", "quit"}:
                break

            print("Recording...", flush=True)
            encoded_audio = await asyncio.to_thread(
                _record_audio_base64,
                input_sample_rate,
                record_seconds,
            )
            print("Recording ended.", flush=True)

            if not encoded_audio:
                print("[error] No audio captured. Check your microphone.")
                continue

            await _append_audio_buffer(encoded_audio)
            await _commit_audio_buffer()
            await _trigger_response()
    finally:
        consumer_task.cancel()
        await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())

