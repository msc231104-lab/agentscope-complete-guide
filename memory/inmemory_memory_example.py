import asyncio
import json
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg

async def in_memory_example():
    memory = InMemoryMemory()

    await memory.add(
        Msg(
            name="system",
            role="system",
            content="<system-hint>Create a plan first and then proceed step by step</system-hint>"

        ),
        marks="hint"
    )

    hint_msg = await memory.get_memory(mark="hint")
    print("Messages with mark 'hint':")
    for msg in hint_msg:
        print("-",msg)

    state = memory.state_dict()
    print(json.dumps(state, indent=2))

    deleted_count = await memory.delete_by_mark(mark="hint")
    print(f"Deleted {deleted_count} message(s).")

if __name__ == "__main__":
    asyncio.run(in_memory_example())

