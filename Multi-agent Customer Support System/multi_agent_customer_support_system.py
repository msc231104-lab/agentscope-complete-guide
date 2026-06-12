import asyncio
import os
from dotenv import load_dotenv
from typing import  Literal
from pydantic import BaseModel , Field
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter , OpenAIMultiAgentFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg , TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.pipeline import MsgHub
from agentscope.tool  import Toolkit , ToolResponse


load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

model = OpenAIChatModel(
    api_key=api_key,
    model_name="gpt-4o-mini",
    stream=False,
)

formatter = OpenAIChatFormatter()

memory = InMemoryMemory()

toolkit = Toolkit()

class RouteDecision(BaseModel):
    category: Literal["technical","order","complaint","general"] = Field(
        description=(
            "Issue category: technical (technical issues),"
            "order (order issues), complaint (complaints),"
            "general (general inquiries)"
        )
    )

    confidence : float = Field(
        description = "Classification confidence, between 0 and 1",
        ge=0,
        le=1,
    )

    summary : str = Field(
        description = "Brief summary of the issue"

    )

    priority : Literal["low","medium", "high"] = Field(
        description = "Issue Priority"
    )


def create_router_agent() -> ReActAgent:
    return ReActAgent(
        name="Router",
        sys_prompt="""You are an intelligent routing system responsible for \
            analyzing customer issues and deciding which specialized team should \
handle them.

Classification rules:
- technical: Product usage issues, technical failures, feature inquiries
- order: Order status, logistics, returns and exchanges
- complaint: Dissatisfaction, complaints, compensation requests
- general: Membership, promotions, general inquiries

Priority rules:
- high: Complaints, urgent order issues
- medium: General order and technical issues
- low: Inquiry-type questions""",
    formatter=formatter,
    memory=memory,
    toolkit=toolkit,
    model=model,
)

class ResolutionReport(BaseModel):
    resolved: bool = Field(description="Whether the issue was resolved")
    solution: str = Field(description="Final solution for the customer")
    follow_up: str = Field(description="Follow-up action if needed")


def query_order(order_id: str) -> ToolResponse:
    orders = {
        "12345": {"status": "shipped", "eta": "2024-01-20"},
        "67890": {"status": "processing", "eta": "2024-01-22"},
    }

    order = orders.get(order_id, {"status": "not_found"})

    return ToolResponse(
        content=[TextBlock(type="text", text=f"Order {order_id}: {order}")]
    )


class CustomerSupportSystem:
    def __init__(self):
        self.router = create_router_agent()

        self.tech_agent = self.create_specialist_agent(
            "TechSupport",
            "technical support",
        )

        self.order_agent = self.create_specialist_agent(
            "OrderSupport",
            "order services",
        )

        self.complaint_agent = self.create_specialist_agent(
            "ComplaintHandler",
            "complaint handling",
        )

        self.supervisor = self.create_supervisor_agent()

    def create_specialist_agent(self, name: str, specialty: str) -> ReActAgent:
        specialist_toolkit = Toolkit()

        if specialty == "order services":
            specialist_toolkit.register_tool_function(query_order)

        return ReActAgent(
            name=name,
            sys_prompt=f"""
You are a {specialty} specialist.

Handle customer issues professionally, clearly, and politely.
If order information is needed, use the available tools.
""",
            model=model,
            formatter=OpenAIMultiAgentFormatter(),
            memory=InMemoryMemory(),
            toolkit=specialist_toolkit,
        )

    def create_supervisor_agent(self) -> ReActAgent:
        return ReActAgent(
            name="Supervisor",
            sys_prompt="""
You are a customer service supervisor.

Review the specialist response and prepare the final customer-facing reply.
Be polite, concise, and solution-oriented.
""",
            model=model,
            formatter=OpenAIMultiAgentFormatter(),
            memory=InMemoryMemory(),
            toolkit=Toolkit(),
        )

    async def handle_customer(self, customer_id: str, issue: str) -> str:
        print(f"\nCustomer {customer_id}: {issue}")

        route_response = await self.router(
            Msg("Customer", issue, "user"),
            structured_model=RouteDecision,
        )

        decision = route_response.metadata
        category = decision.get("category", "general")
        priority = decision.get("priority", "medium")

        print(f"Routing → Category: {category}, Priority: {priority}")

        specialist_map = {
            "technical": self.tech_agent,
            "order": self.order_agent,
            "complaint": self.complaint_agent,
            "general": self.tech_agent,
        }

        specialist = specialist_map.get(category, self.tech_agent)

        async with MsgHub(participants=[specialist, self.supervisor]):
            await specialist(
                Msg(
                    "Customer",
                    f"Customer ID: {customer_id}\nIssue: {issue}",
                    "user",
                )
            )

            final_response = await self.supervisor(
                Msg(
                    "System",
                    "Review the specialist response and provide the final customer reply.",
                    "user",
                ),
                structured_model=ResolutionReport,
            )

        report = final_response.metadata

        return (
            f"Solution: {report.get('solution')}\n\n"
            f"Follow-up: {report.get('follow_up')}\n\n"
            f"Resolved: {report.get('resolved')}"
        )


async def main() -> None:
    system = CustomerSupportSystem()

    customer_issues = [
        ("C001", "Your app keeps crashing. I can't use it at all!"),
        ("C002", "Has my order 12345 shipped? When will it arrive?"),
        ("C003", "This is terrible! I demand a full refund!"),
        ("C004", "Do you have student discounts?"),
    ]

    for customer_id, issue in customer_issues:
        response = await system.handle_customer(customer_id, issue)
        print(f"\nFinal Response:\n{response}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())