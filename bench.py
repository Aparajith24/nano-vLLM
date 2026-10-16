import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8080/v1/completions"
PROMPT = "The future of AI inference is"
RESULTS = Path(__file__).parent / "results"


def percentile(values, p):
    """The value below which p percent of the sorted values fall."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


async def one_user(client, max_tokens, start_line):
    """Wait for the starting signal, stream one answer, and record when each token arrived."""
    await start_line.wait()
    sent = time.perf_counter()
    arrivals, completion_tokens = [], 0
    body = {"prompt": PROMPT, "max_tokens": max_tokens, "stream": True}
    async with client.stream("POST", URL, json=body) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk["usage"] is None:
                arrivals.append(time.perf_counter())
            else:
                completion_tokens = chunk["usage"]["completion_tokens"]
    return {"sent": sent, "arrivals": arrivals, "done": time.perf_counter(), "completion_tokens": completion_tokens}


async def run_level(users, max_tokens):
    """Start this many users at the same moment and wait for all of them to finish."""
    start_line = asyncio.Event()
    limits = httpx.Limits(max_connections=users, max_keepalive_connections=users)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        tasks = [asyncio.create_task(one_user(client, max_tokens, start_line)) for _ in range(users)]
        await asyncio.sleep(0.1)
        start_line.set()
        return await asyncio.gather(*tasks)


def summarize(users, requests):
    """Turn raw timestamps into total tok/s, TTFT, gaps between tokens, and per-user tok/s."""
    start = min(r["sent"] for r in requests)
    end = max(r["done"] for r in requests)
    total_tokens = sum(r["completion_tokens"] for r in requests)
    ttfts = [(r["arrivals"][0] - r["sent"]) * 1000 for r in requests if r["arrivals"]]
    gaps = [(b - a) * 1000 for r in requests for a, b in zip(r["arrivals"], r["arrivals"][1:])]
    per_user = [r["completion_tokens"] / (r["done"] - r["sent"]) for r in requests]
    return {
        "users": users,
        "wall_s": end - start,
        "total_tokens": total_tokens,
        "total_tok_s": total_tokens / (end - start),
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p99_ms": percentile(ttfts, 99),
        "itl_p50_ms": percentile(gaps, 50),
        "itl_p99_ms": percentile(gaps, 99),
        "per_user_tok_s": statistics.mean(per_user),
    }


def print_table(rows):
    """Print one line per user count."""
    print(f"{'users':>5} {'tok/s':>8} {'TTFT p50':>9} {'TTFT p99':>9} {'ITL p50':>8} {'ITL p99':>8} {'per user':>9}")
    for r in rows:
        print(f"{r['users']:>5} {r['total_tok_s']:>8.0f} {r['ttft_p50_ms']:>7.0f}ms {r['ttft_p99_ms']:>7.0f}ms "
              f"{r['itl_p50_ms']:>6.1f}ms {r['itl_p99_ms']:>6.1f}ms {r['per_user_tok_s']:>9.1f}")


async def main(args):
    """Warm the server up, run every user count in turn, then print and save the results."""
    await run_level(1, 5)
    rows, raw = [], {}
    for users in args.users:
        requests = await run_level(users, args.max_tokens)
        rows.append(summarize(users, requests))
        raw[users] = requests
        print(f"done: {users} users")
        await asyncio.sleep(args.pause)
    print_table(rows)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{args.label}.json"
    path.write_text(json.dumps({"label": args.label, "prompt": PROMPT, "max_tokens": args.max_tokens,
                                "summary": rows, "raw": raw}, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the completions server with many streaming users.")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 10, 30, 40])
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--pause", type=float, default=2.0)
    asyncio.run(main(parser.parse_args()))
