"""
Batch Chat Script
=================
Sends N queries to the /api/v1/chat endpoint (one per simulated user)
and writes all results to response.json.

Usage:
    python scripts/batch_chat.py                          # uses built-in sample queries
    python scripts/batch_chat.py --file queries.txt       # one query per line
    python scripts/batch_chat.py --url http://host:8000   # custom base URL
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict

import httpx

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:8001"
CHAT_ENDPOINT = "/api/v1/chat"

SAMPLE_QUERIES = [
    "How many movies are in the database?",
    "List top 5 actors by number of movies they acted in.",
    "Which director has directed the most movies?",
    "What are the most popular genres?",
    "Tell me about Tom Hanks movies.",
]


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    user: int
    session_id: str
    query: str
    response: str | None
    tokens_used: int | None
    status: str  # "success" or "error"
    error: str | None
    elapsed_seconds: float


# ── Core ──────────────────────────────────────────────────────────────────────

async def send_query(
    client: httpx.AsyncClient,
    url: str,
    user_index: int,
    query: str,
    request_timeout: float,
) -> QueryResult:
    """Send a single chat query and return a QueryResult."""
    session_id = str(uuid.uuid4())
    payload = {"session_id": session_id, "message": query}
    start = time.perf_counter()

    try:
        resp = await client.post(url, json=payload, timeout=request_timeout)
        elapsed = time.perf_counter() - start
        resp.raise_for_status()
        data = resp.json()
        return QueryResult(
            user=user_index,
            session_id=session_id,
            query=query,
            response=data.get("message"),
            tokens_used=data.get("tokens_used"),
            status="success",
            error=None,
            elapsed_seconds=round(elapsed, 3),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return QueryResult(
            user=user_index,
            session_id=session_id,
            query=query,
            response=None,
            tokens_used=None,
            status="error",
            error=str(exc),
            elapsed_seconds=round(elapsed, 3),
        )


async def run_batch(
    base_url: str,
    queries: list[str],
    concurrency: int,
    request_timeout: float,
) -> list[QueryResult]:
    """Send all queries (with bounded concurrency) and return results."""
    url = f"{base_url.rstrip('/')}{CHAT_ENDPOINT}"
    semaphore = asyncio.Semaphore(concurrency)
    results: list[QueryResult] = []

    async def _limited(idx: int, q: str) -> QueryResult:
        async with semaphore:
            return await send_query(client, url, idx, q, request_timeout)

    async with httpx.AsyncClient() as client:
        tasks = [_limited(i + 1, q) for i, q in enumerate(queries)]
        results = await asyncio.gather(*tasks)

    return sorted(results, key=lambda r: r.user)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-query the chat API and save responses.")
    p.add_argument(
        "--file", "-f",
        help="Path to a text file with one query per line. If omitted, built-in samples are used.",
    )
    p.add_argument(
        "--url", "-u",
        default=DEFAULT_BASE_URL,
        help=f"Base URL of the API (default: {DEFAULT_BASE_URL}).",
    )
    p.add_argument(
        "--concurrency", "-c",
        type=int,
        default=5,
        help="Max concurrent requests (default: 5).",
    )
    p.add_argument(
        "--timeout", "-t",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds (default: 120).",
    )
    p.add_argument(
        "--output", "-o",
        default="response.json",
        help="Output JSON file path (default: response.json).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load queries
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            queries = [line.strip() for line in fh if line.strip()]
        if not queries:
            print(f"ERROR: No queries found in {args.file}")
            return
    else:
        queries = SAMPLE_QUERIES

    print(f"Sending {len(queries)} queries to {args.url}{CHAT_ENDPOINT}")
    print(f"Concurrency: {args.concurrency} | Timeout: {args.timeout}s\n")

    total_start = time.perf_counter()
    results = asyncio.run(run_batch(args.url, queries, args.concurrency, args.timeout))  # type: ignore[arg-type]
    total_elapsed = round(time.perf_counter() - total_start, 3)

    # Print summary
    success = sum(1 for r in results if r.status == "success")
    failed = len(results) - success
    print(f"\n{'='*60}")
    print(f"  Total queries : {len(results)}")
    print(f"  Success       : {success}")
    print(f"  Failed        : {failed}")
    print(f"  Total time    : {total_elapsed}s")
    print(f"{'='*60}\n")

    for r in results:
        tag = "OK" if r.status == "success" else "FAIL"
        preview = (r.response or r.error or "")[:80]
        print(f"  [User {r.user}] [{tag}] {r.query[:50]}")
        print(f"           -> {preview}...")
        print()

    # Write JSON output
    output = {
        "total_queries": len(results),
        "success_count": success,
        "failure_count": failed,
        "total_elapsed_seconds": total_elapsed,
        "results": [asdict(r) for r in results],
    }

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
