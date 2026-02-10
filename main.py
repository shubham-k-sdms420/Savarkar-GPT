"""
Savarkar GPT - Main Entry Point

Usage:
    python main.py serve       # Start web UI (FastAPI + frontend)
    python main.py ingest      # Ingest data into Pinecone (run once)
    python main.py query       # Interactive query mode (CLI)
    python main.py stats       # Show Pinecone index stats
    python main.py usage       # Show token usage statistics
    python main.py config      # Print current configuration
"""

import os
import sys

from app.config.settings import settings


def cmd_serve():
    """Start the web UI server (FastAPI + frontend)."""
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    print("=" * 60)
    print("SAVARKAR GPT - Web Server")
    print("=" * 60)
    settings.print_config()
    print(f"\n  Starting server at http://localhost:{port}")
    print("  Press Ctrl+C to stop.\n")

    uvicorn.run(
        "app.api.server:app",
        host=host,
        port=port,
        reload=False,
    )


def cmd_ingest():
    """Run the data ingestion pipeline."""
    from app.pipeline.ingest import IngestionPipeline

    pipeline = IngestionPipeline()
    pipeline.run()


def cmd_query():
    """Interactive query mode via CLI."""
    from app.rag.chain import RAGChain

    print("=" * 60)
    print("SAVARKAR GPT - Interactive Query Mode")
    print("=" * 60)
    print("Ask any question about Savarkar. Type 'quit' to exit.\n")

    rag = RAGChain()

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print("\nSearching knowledge base...")
        result = rag.query(question)

        print(f"\nSavarkar GPT: {result['answer']}")
        print(f"\n--- Sources ({result['num_passages_retrieved']} passages) ---")
        for i, src in enumerate(result["sources"], 1):
            print(
                f"  {i}. \"{src['title']}\" by {src['author']} "
                f"| Ch: {src['chapter']} | Score: {src['relevance_score']:.2f}"
            )


def cmd_stats():
    """Show Pinecone index statistics."""
    from app.vectorstore.pinecone_store import PineconeStore

    store = PineconeStore()
    stats = store.get_index_stats()
    print("Pinecone Index Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def cmd_usage():
    """Show token usage statistics from logged requests."""
    from app.monitoring.token_logger import get_usage_summary

    summary = get_usage_summary()

    print("=" * 60)
    print("SAVARKAR GPT - Token Usage Statistics")
    print("=" * 60)

    if summary["total_requests"] == 0:
        print("\n  No requests logged yet. Ask some questions first!\n")
        return

    print(f"\n  Total requests:        {summary['total_requests']}")
    print(f"  Total input tokens:    {summary['total_input_tokens']:,}")
    print(f"  Total output tokens:   {summary['total_output_tokens']:,}")
    print(f"  Total tokens:          {summary['total_tokens']:,}")
    print(f"  Avg input tokens/req:  {summary['avg_input_tokens']:,}")
    print(f"  Avg output tokens/req: {summary['avg_output_tokens']:,}")
    print(f"  Avg latency:           {summary['avg_latency_ms']:,} ms")

    recent = summary.get("recent_requests", [])
    if recent:
        print(f"\n  --- Last {len(recent)} Requests ---")
        for i, r in enumerate(recent, 1):
            q = r["question"][:50] + ("..." if len(r["question"]) > 50 else "")
            print(
                f"  {i}. [{r['timestamp']}] \"{q}\" "
                f"| in={r['input_tokens']} out={r['output_tokens']} "
                f"| {r['latency_ms']}ms"
            )
    print()


def cmd_config():
    """Print current configuration."""
    settings.print_config()
    errors = settings.validate()
    if errors:
        print("\nConfiguration Errors:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("\nConfiguration is valid.")


COMMANDS = {
    "serve": cmd_serve,
    "ingest": cmd_ingest,
    "query": cmd_query,
    "stats": cmd_stats,
    "usage": cmd_usage,
    "config": cmd_config,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("Available commands:")
        for cmd, fn in COMMANDS.items():
            print(f"  {cmd:10s} - {fn.__doc__.strip()}")
        sys.exit(1)

    command = sys.argv[1]
    COMMANDS[command]()


if __name__ == "__main__":
    main()
