"""Demo script to visualize Tax Agent node-by-node execution flow.

Usage:
    poetry run python scripts/demo_tax_agent.py
"""

from dotenv import load_dotenv

load_dotenv()  # Load .env BEFORE any LangChain imports (for LangSmith tracing)

from finance_ai.agents.llm_factory import create_chat_model
from finance_ai.agents.tax_agent import build_tax_agent_graph


def print_separator() -> None:
    """Print a visual separator line."""
    print("=" * 70)


def print_node_event(node_name: str, event: dict) -> None:  # type: ignore[type-arg]
    """Print a single node execution event.

    Args:
        node_name: Name of the node that executed.
        event: The event data from that node.
    """
    print(f"\n🔹 Node: {node_name}")
    print("-" * 40)
    messages = event.get("messages", [])
    for msg in messages:
        role = type(msg).__name__
        print(f"  [{role}]")
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"    🔧 Tool Call: {tc['name']}")
                print(f"       Args: {tc['args']}")
        if msg.content:
            content = str(msg.content)[:500]
            print(f"    Content: {content}")


def main() -> None:
    """Run the Tax Agent with streaming to show node flow."""
    query = "คำนวณภาษีปี 2024 เงินเดือน 1.2 ล้าน มีลูก 2 คน ซื้อ RMF 100000"

    print_separator()
    print("🚀 Tax Agent Demo - Node Flow Visualization")
    print_separator()
    print(f"\n📝 User Query: {query}\n")

    model = create_chat_model()
    graph = build_tax_agent_graph(chat_model=model)

    print_separator()
    print("📊 Execution Flow:")
    print_separator()

    step = 0
    for event in graph.stream({"messages": [("user", query)]}):
        step += 1
        print(f"\n{'🟢' if step == 1 else '🔄'} Step {step}")
        for node_name, node_data in event.items():
            print_node_event(node_name, node_data)

    print_separator()
    print("✅ Execution Complete")
    print_separator()


if __name__ == "__main__":
    main()
