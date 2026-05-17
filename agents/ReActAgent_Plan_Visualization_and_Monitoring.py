import asyncio
from agentscope.plan import PlanNotebook , Plan , SubTask

plan_notebook = PlanNotebook()

def on_plan_changed(
    self: plan_notebook,
    plan:Plan
) -> None:
    print("\nPlan Updated:\n")
    print(plan)

plan_notebook.register_plan_change_hook(
    hook_name="print_hook",
    hook=on_plan_changed,
)

async def main():
        await plan_notebook.create_plan(

        name="Study AgentScope",

        description="Learn AgentScope step by step.",

        expected_outcome="Understand basic AgentScope usage.",

        subtasks=[

            SubTask(
                name="Read basics",

                description="Read introductory concepts.",

                expected_outcome="Understand basics.",
            )
        ],
    )

if __name__ == "__main__":
    asyncio.run(main())

