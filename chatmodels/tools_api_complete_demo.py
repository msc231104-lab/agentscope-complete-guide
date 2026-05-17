import asyncio
import os
from dotenv import load_dotenv
from pathlib import Path
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.message import Msg

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

async def openai_example_tools():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY was not found. Check your .env path and value.")

    model = OpenAIChatModel(
        model_name="gpt-4o-mini",
        api_key=api_key,
        stream=False,
    )

    formatter = OpenAIChatFormatter()

    json_schema = [
                {
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "Search for a query on Google.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    user_msg = Msg(
        name="Aadhya",
        role="user",
        content="Search AgentScope release notes.",
    )

    formatted_messages = await formatter.format([user_msg])

    response = await model(
        messages=formatted_messages,
        tools=json_schema,
        tool_choice="auto",
    )

    print("Step 1 - Model response (tool request):")
    print(response.content)

    # Extract tool call from response
    tool_use_block = None
    for block in response.content:
        if block.get("type") == "tool_use":
            tool_use_block = block
            break

    if tool_use_block:
        print(f"\nStep 2 - Executing tool: {tool_use_block['name']}")
        tool_name = tool_use_block["name"]
        tool_input = tool_use_block["input"]

        # Execute the tool (mock implementation)
        if tool_name == "google_search":
            # Mock search result
            tool_result = f"Search results for '{tool_input['query']}':\n1. AgentScope v1.0 released - January 2024\n2. New features: Multi-agent coordination, Tool use\n3. Latest version: 1.0.19"
        else:
            tool_result = f"Tool {tool_name} not implemented"

        print(f"Tool result: {tool_result}\n")

        # Send tool result back to model for final answer
        print("Step 3 - Sending tool result back to model for final answer:")

        # Build messages for second model call with tool result
        # Format in OpenAI native format (not AgentScope blocks)
        messages_with_result = [
            {"role": "user", "content": "Search AgentScope release notes."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_use_block["id"],
                        "type": "function",
                        "function": {
                            "name": tool_use_block["name"],
                            "arguments": str(tool_use_block["input"]).replace("'", '"')
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": tool_use_block["id"],
                "content": tool_result
            }
        ]

        # Get final response from model
        final_response = await model(
            messages=messages_with_result,
            tools=json_schema,
        )

        print("Final answer:")
        for block in final_response.content:
            if block.get("type") == "text":
                print(block.get("text"))
    else:
        print("No tool call made by the model")

if __name__ == "__main__":
    asyncio.run(openai_example_tools())