"""Tests for the pure helpers in conv_search.js, ask_about.js and insights.js.

They run the real JS in node (like test_chat_format_js.py) and are skipped
when node is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

STATIC = Path(__file__).resolve().parents[1] / "src" / "finance_ai" / "static"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def call_js(module: str, function: str, *args: Any) -> Any:
    """Call module.function(*args) in node and return its JSON result.

    Args:
        module: File name inside static/, e.g. "conv_search.js".
        function: Exported function name.
        *args: JSON-serialisable arguments.

    Returns:
        The decoded return value.
    """
    script = (
        f"const m = require({json.dumps(str(STATIC / module))});"
        "let s = ''; process.stdin.on('data', (d) => (s += d));"
        f"process.stdin.on('end', () => process.stdout.write("
        f"JSON.stringify(m.{function}(...JSON.parse(s)) ?? null)));"
    )
    completed = subprocess.run(
        [str(NODE), "-e", script],
        input=json.dumps(list(args)),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


class TestConvSearch:
    """Sidebar chat search: titles plus guru names in Thai and English."""

    def test_matches_title_case_insensitively(self) -> None:
        """'ssf' finds 'ลดหย่อนภาษีด้วย SSF'."""
        assert call_js("conv_search.js", "matches", "ลดหย่อนภาษีด้วย SSF", "ssf")

    def test_matches_guru_alias_in_either_language(self) -> None:
        """'buffet' and 'บัฟเฟต' both find a Buffett chat via its search words."""
        text = "ถามเรื่องหุ้น Warren Buffett buffett วอร์เรน บัฟเฟตต์"
        assert call_js("conv_search.js", "matches", text, "buffet")
        assert call_js("conv_search.js", "matches", text, "บัฟเฟต")
        assert not call_js("conv_search.js", "matches", text, "dalio")

    def test_empty_query_matches_everything(self) -> None:
        """No query, no filter; extra spaces are ignored."""
        assert call_js("conv_search.js", "matches", "anything", "   ")
        assert call_js("conv_search.js", "matches", "ค่า  อาหาร", "ค่า อาหาร")

    def test_split_match_keeps_original_case(self) -> None:
        """The marked part keeps the title's own spelling."""
        assert call_js("conv_search.js", "splitMatch", "พอร์ต PTT กับ KBANK", "ptt") == {
            "before": "พอร์ต ",
            "match": "PTT",
            "after": " กับ KBANK",
        }
        assert call_js("conv_search.js", "splitMatch", "ภาษี", "หุ้น") is None
        assert call_js("conv_search.js", "splitMatch", "ภาษี", "") is None


class TestAskAbout:
    """'Ask about this' message building."""

    def test_question_quotes_each_line(self) -> None:
        """The picked text becomes a markdown quote under the question."""
        message = call_js(
            "ask_about.js",
            "composeQuestion",
            "อธิบายตรงนี้แบบง่ายๆ ",
            "ซื้อ SSF\n\n  เพิ่มอีก 30,000",
        )
        assert message == "อธิบายตรงนี้แบบง่ายๆ:\n> ซื้อ SSF\n> เพิ่มอีก 30,000"

    def test_selection_length_limits(self) -> None:
        """2–500 characters after trimming."""
        assert call_js("ask_about.js", "isAskable", "SSF")
        assert not call_js("ask_about.js", "isAskable", " a ")
        assert not call_js("ask_about.js", "isAskable", "x" * 501)


def trends(**overrides: Any) -> dict[str, Any]:
    """A /dashboard/trends payload with sensible defaults."""
    payload: dict[str, Any] = {
        "spending": [
            {"year": 2026, "month": m, "total": t}
            for m, t in [
                (5, "11800"),
                (6, "12400"),
                (7, "0"),
                (8, "13100"),
                (9, "12100"),
                (10, "15650"),
            ]
        ],
        "average_spending": "12350.00",
        "spikes": [],
        "assets": [],
        "benchmark": None,
    }
    payload.update(overrides)
    return payload


def asset(symbol: str, kind: str, change: str, values: list[str]) -> dict[str, Any]:
    """One asset line."""
    points = [{"day": f"2026-10-{i + 1:02d}", "value": v} for i, v in enumerate(values)]
    return {"symbol": symbol, "name": symbol, "kind": kind, "points": points, "change_pct": change}


class TestInsightPages:
    """buildPages turns the trends payload into carousel pages."""

    def test_spending_page_compares_with_average(self) -> None:
        """15,650 vs 12,350 is +27%; labels are Thai month abbreviations."""
        pages = call_js("insights.js", "buildPages", trends())
        assert [page["key"] for page in pages] == ["spending"]
        assert ["+27%", "bad"] in pages[0]["prose"]
        assert pages[0]["chart"]["labels"][-1] == "ต.ค."
        assert pages[0]["question"] == "ควรลดรายจ่ายตรงไหนก่อน?"

    def test_no_data_means_no_pages(self) -> None:
        """A new user (no spending, no holdings) gets no cards at all."""
        empty = trends(
            spending=[{"year": 2026, "month": 10, "total": "0"}], average_spending="0.00"
        )
        assert call_js("insights.js", "buildPages", empty) == []
        assert call_js("insights.js", "buildPages", None) == []

    def test_spike_page_uses_biggest_increase(self) -> None:
        """The category furthest above normal leads the card."""
        spikes = [
            {
                "category": "food",
                "label": "อาหาร",
                "current": "6480",
                "average": "4800.00",
                "increase_pct": 35,
                "months_compared": 3,
            },
            {
                "category": "shopping",
                "label": "ช้อปปิ้ง",
                "current": "4000",
                "average": "2000.00",
                "increase_pct": 100,
                "months_compared": 3,
            },
        ]
        pages = call_js("insights.js", "buildPages", trends(spikes=spikes))
        spike = next(page for page in pages if page["key"] == "spike")
        assert spike["question"] == "ลดค่าช้อปปิ้งยังไงดี?"
        assert spike["chart"]["labels"] == ["ช้อปปิ้ง", "อาหาร"]

    def test_asset_page_shows_best_worst_and_benchmark_as_percent(self) -> None:
        """Lines are % since day 1; the SET50 line is dashed."""
        assets = [
            asset("PTT.BK", "stock", "3.70", ["40.50", "42.00"]),
            asset("SCC.BK", "stock", "1.00", ["200", "202"]),
            asset("K-USA-A(A)", "mutual_fund", "-2.10", ["15.00", "14.685"]),
        ]
        benchmark = asset("TDEX.BK", "benchmark", "1.20", ["10.00", "10.12"]) | {
            "name": "SET50 (TDEX)"
        }
        pages = call_js("insights.js", "buildPages", trends(assets=assets, benchmark=benchmark))
        page = next(page for page in pages if page["key"] == "assets")
        labels = [series["label"] for series in page["chart"]["series"]]
        assert labels == ["PTT.BK", "K-USA-A(A)", "SET50 (TDEX)"]
        assert page["chart"]["series"][0]["values"] == [0, 3.7]
        assert page["chart"]["series"][2]["dashed"] is True
        assert page["chart"]["series"][2]["color"] == "--faint"
        assert page["chart"]["labels"] == ["1 ต.ค.", "2 ต.ค."]
