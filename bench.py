import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8080/v1/completions"
PROMPT = "The future of AI inference is"
SYSTEM_SENTENCE = "You are a helpful assistant who answers questions about the history of printing."
LATE_SENTENCE = "The archive holds letters, maps and ledgers from many old trading houses."
RESULTS = Path(__file__).parent / "results"


def percentile(values, p):
    """The value below which p percent of the sorted values fall."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def prompt_for(user, system_repeats):
    """The plain benchmark prompt, or a long shared system prompt followed by this user's own question."""
    if not system_repeats:
        return PROMPT
    return " ".join([SYSTEM_SENTENCE] * system_repeats) + f" Question {user}: what happened in the year {1400 + user}?"


def late_prompt(user, repeats):
    """A long prompt that starts differently for every late user, so no blocks are shared through the prefix cache."""
    return f"Report {user}: " + " ".join([LATE_SENTENCE] * repeats)


async def one_user(client, max_tokens, start_line, prompt, delay=0.0, group="early"):
    """Wait for the starting signal (plus a delay), stream one answer, and record when each token arrived."""
    await start_line.wait()
    await asyncio.sleep(delay)
    sent = time.perf_counter()
    arrivals, completion_tokens = [], 0
    body = {"prompt": prompt, "max_tokens": max_tokens, "stream": True}
    async with client.stream("POST", URL, json=body) as response:
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk["usage"] is None:
                arrivals.append(time.perf_counter())
            else:
                completion_tokens = chunk["usage"]["completion_tokens"]
    return {"group": group, "sent": sent, "arrivals": arrivals, "done": time.perf_counter(),
            "completion_tokens": completion_tokens}


async def run_level(users, max_tokens, system_repeats=0, late=None):
    """Start this many users at the same moment, plus any late users after their delay, and wait for all of them."""
    start_line = asyncio.Event()
    late_users, late_repeats, late_delay, late_max_tokens = late or (0, 0, 0.0, 0)
    limits = httpx.Limits(max_connections=users + late_users, max_keepalive_connections=users + late_users)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        tasks = [asyncio.create_task(one_user(client, max_tokens, start_line, prompt_for(user, system_repeats)))
                 for user in range(users)]
        tasks += [asyncio.create_task(one_user(client, late_max_tokens, start_line, late_prompt(user, late_repeats),
                                               late_delay, "late"))
                  for user in range(late_users)]
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
        "itl_max_ms": max(gaps) if gaps else None,
        "per_user_tok_s": statistics.mean(per_user),
    }


def print_table(rows):
    """Print one line per user count (or per group of users)."""
    print(f"{'users':>7} {'tok/s':>8} {'TTFT p50':>9} {'TTFT p99':>9} {'ITL p50':>8} {'ITL p99':>8} {'ITL max':>8} "
          f"{'per user':>9}")
    for r in rows:
        print(f"{r['users']!s:>7} {r['total_tok_s']:>8.0f} {r['ttft_p50_ms']:>7.0f}ms {r['ttft_p99_ms']:>7.0f}ms "
              f"{r['itl_p50_ms']:>6.1f}ms {r['itl_p99_ms']:>6.1f}ms {r['itl_max_ms']:>6.0f}ms {r['per_user_tok_s']:>9.1f}")


async def main(args):
    """Warm the server up, run every user count in turn, then print and save the results."""
    await run_level(1, 5)
    rows, raw = [], {}
    late = (args.late_users, args.late_repeats, args.late_delay, args.late_max_tokens)
    for users in args.users:
        requests = await run_level(users, args.max_tokens, args.system_repeats, late)
        rows.append(summarize(users, requests))
        if args.late_users:
            for group in ("early", "late"):
                rows[-1][group] = summarize(f"{group}", [r for r in requests if r["group"] == group])
        raw[users] = requests
        print(f"done: {users} users")
        await asyncio.sleep(args.pause)
    print_table(rows)
    if args.late_users:
        print_table([row[group] for row in rows for group in ("early", "late")])
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{args.label}.json"
    path.write_text(json.dumps({"label": args.label, "prompt": prompt_for(0, args.system_repeats),
                                "max_tokens": args.max_tokens,
                                "summary": rows, "raw": raw}, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the completions server with many streaming users.")
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 10, 30, 40])
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--pause", type=float, default=2.0)
    parser.add_argument("--system-repeats", type=int, default=0,
                        help="start every prompt with the same system sentence repeated this many times")
    parser.add_argument("--late-users", type=int, default=0, help="extra users who arrive later with long prompts")
    parser.add_argument("--late-repeats", type=int, default=60, help="sentences in each late user's prompt")
    parser.add_argument("--late-delay", type=float, default=2.0, help="seconds after the start that late users arrive")
    parser.add_argument("--late-max-tokens", type=int, default=50)
    asyncio.run(main(parser.parse_args()))
