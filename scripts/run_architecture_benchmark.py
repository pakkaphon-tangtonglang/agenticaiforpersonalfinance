"""Run multi-agent architecture benchmark and save comparison report.

Usage:
    python scripts/run_architecture_benchmark.py

Output:
    data/evaluation/results/architecture_comparison.md
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

from finance_ai.evaluation.architecture_benchmark import run_architecture_benchmark
from finance_ai.evaluation.architecture_reporter import save_architecture_report


def main() -> None:
    """Run benchmark and save report."""
    print("เริ่มต้น Architecture Benchmark...")
    print("จะเรียก LLM ~15 ครั้ง (5 queries × 3 architectures)\n")

    print("🔵 [1/3] รัน Hub-and-Spoke...")
    print("🔴 [2/3] รัน Peer-to-Peer...")
    print("🟠 [3/3] รัน Hierarchical...\n")

    result = run_architecture_benchmark()

    print("\n✅ ผลลัพธ์:")
    print(f"  Hub-and-Spoke : {result.hub_spoke.mean_latency_ms:.0f} ms, "
          f"{result.hub_spoke.routing_llm_calls_per_query:.1f} routing calls, "
          f"{result.hub_spoke.coupling_score} connections")
    print(f"  P2P           : {result.p2p.mean_latency_ms:.0f} ms, "
          f"{result.p2p.routing_llm_calls_per_query:.1f} routing calls, "
          f"{result.p2p.coupling_score} connections")
    print(f"  Hierarchical  : {result.hierarchical.mean_latency_ms:.0f} ms, "
          f"{result.hierarchical.routing_llm_calls_per_query:.1f} routing calls, "
          f"{result.hierarchical.coupling_score} connections")

    output_path = "data/evaluation/results/architecture_comparison.md"
    path = save_architecture_report(result, output_path)
    print(f"\n📄 บันทึกรายงานที่: {path}")


if __name__ == "__main__":
    main()
