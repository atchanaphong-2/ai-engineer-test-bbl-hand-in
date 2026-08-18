"""CLI entry point: runs the two-agent RAG orchestrator for one query."""

import argparse
import asyncio
import logging
import sys
from collections.abc import Callable

from agentic_rag.bootstrap import AppContainer
from agentic_rag.errors import AgentInvocationError
from agentic_rag.orchestrator import RagOrchestrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Each CLI invocation is a fresh, single-turn process — MemorySaver's state
# dies with it either way, so the thread_id value itself is arbitrary here.
CLI_THREAD_ID = "cli"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question against the local knowledge base."
    )
    parser.add_argument("query", help="The question to ask.")
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    orchestrator_factory: Callable[[], RagOrchestrator] | None = None,
) -> int:
    """Run the CLI. `orchestrator_factory` is injectable so tests can supply
    a fake/failing orchestrator without a live LLM (defaults to AppContainer's)."""
    args = parse_args(argv)
    build_orchestrator = orchestrator_factory or (lambda: AppContainer().orchestrator)

    try:
        orchestrator = build_orchestrator()
        state = asyncio.run(orchestrator.run(args.query, CLI_THREAD_ID))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    except AgentInvocationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    report = state["final_report"]

    print(f"\nQuery: {args.query}\n")
    print("Retrieved snippets:")
    for chunk in state["retrieved_chunks"]:
        print(f"  - {chunk}")

    print("\nAnswer:")
    print(report.answer if report else "(no answer produced)")
    if report is not None and not report.grounded:
        print("\n[Note: answer may be incomplete relative to the knowledge base]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
