import asyncio
import json
import os
import queue
import time
import traceback
import uuid
from collections import deque
from contextlib import asynccontextmanager
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from blocks import DEFAULT_NUM_BLOCKS, BlockAllocator, KVPool, PoolFull
from model import SmolLM2

HOST, PORT = "127.0.0.1", 8080
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_BATCH = 30
NUM_BLOCKS = DEFAULT_NUM_BLOCKS
PREFIX_CACHING = os.environ.get("NANO_PREFIX_CACHE", "1") == "1"
PREFILL_CHUNK = int(os.environ.get("NANO_PREFILL_CHUNK", "512"))
CONTEXT_LENGTH = 1024
DEFAULT_MAX_TOKENS = 16
TEMPERATURE, TOP_K = 0.7, 10
WELCOME_TEXT = "hi! welcome"
PARTS = ("forward", "sampling", "sending")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SmolLM2.from_pretrained(MODEL_NAME, device=DEVICE)
EOS = tokenizer.eos_token_id


class CompletionRequest(BaseModel):
    """What the client sends: a non-empty prompt, optionally max_tokens, and whether to stream (yes by default)."""

    prompt: str = Field(min_length=1)
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1)
    stream: bool = True


class Seat:
    """One user inside the engine: their block table, what they still have to read, their answer so far, and where to send their tokens."""

    def __init__(self, prompt_ids, max_tokens, loop):
        self.prompt_ids, self.max_tokens, self.loop = prompt_ids, max_tokens, loop
        self.tokens = asyncio.Queue()
        self.ids, self.output, self.preempted = prompt_ids, [], 0
        self.table, self.cached, self.prefilled, self.length, self.last_token = [], 0, 0, 0, None
        self.done = self.cancelled = False

    def reading(self):
        """True while part of ids has not been prefilled yet."""
        return self.prefilled < len(self.ids)

    def send(self, token_id):
        """Hand a token id, or None for the end, to this user's request handler on the event loop."""
        self.loop.call_soon_threadsafe(self.tokens.put_nowait, token_id)


def sample(logits):
    """Pick one token per row: keep the TOP_K likeliest, sharpen with TEMPERATURE, then draw at random."""
    top_logits, top_ids = torch.topk(logits / TEMPERATURE, TOP_K)
    choice = torch.multinomial(torch.softmax(top_logits, dim=-1), 1)
    return top_ids.gather(-1, choice).view(-1).tolist()


def gpu_wait():
    """Block until the GPU has finished its queued work, so a stopwatch times the work and not just the queueing."""
    if DEVICE == "mps":
        torch.mps.synchronize()


class Stopwatches:
    """Total time spent in each part of the decode rounds since the last report."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Set every total back to zero."""
        self.totals, self.rounds, self.users, self.last = dict.fromkeys(PARTS, 0.0), 0, 0, 0.0

    def start(self, users):
        """Start timing a new round of this many users."""
        gpu_wait()
        self.rounds, self.users, self.last = self.rounds + 1, self.users + users, time.perf_counter()

    def lap(self, part):
        """Add the time since the previous lap to this part."""
        gpu_wait()
        now = time.perf_counter()
        self.totals[part] += now - self.last
        self.last = now

    def report(self):
        """Print each part's total, its average per round and its share of the time, then start over."""
        total = sum(self.totals.values())
        print(f"decode rounds: {self.rounds}, average batch {self.users / self.rounds:.1f} users")
        print(f"  {'part':<11} {'total ms':>9} {'ms/round':>9} {'share':>6}")
        for part, seconds in list(self.totals.items()) + [("all", total)]:
            print(f"  {part:<11} {seconds * 1000:>9.0f} {seconds * 1000 / self.rounds:>9.2f} {seconds / total:>6.0%}")
        self.reset()


class BatchEngine:
    """Continuous batching over a paged KV cache: each round finds every decoder a block for its next token, seats people whose prompts fit, then gives every seated user one token."""

    def __init__(self):
        self.waiting, self.line, self.seated, self.stopwatches = queue.Queue(), deque(), [], Stopwatches()
        self.allocator = BlockAllocator(NUM_BLOCKS)
        c = model.config
        self.pool = KVPool(c.layers, c.kv_heads, c.head_dim, num_blocks=NUM_BLOCKS, device=DEVICE)

    def run(self):
        """Run rounds forever."""
        while True:
            self.run_round()

    def run_round(self):
        """Free finished users, sleep if nobody is seated or waiting, spend the prefill budget, then decode one token for everyone ready."""
        self.leave([s for s in self.seated if s.done or s.cancelled])
        if not self.seated and self.stopwatches.rounds:
            self.stopwatches.report()
        if not self.seated and not self.line:
            self.line.append(self.waiting.get())
        while not self.waiting.empty():
            self.line.append(self.waiting.get_nowait())
        try:
            with torch.no_grad():
                self.make_room()
                self.prefill_step()
                active = [s for s in self.seated if not s.done and not s.reading()]
                if active:
                    self.decode_round(active)
        except Exception:
            traceback.print_exc()
            for seat in self.seated:
                seat.send(None)
            self.leave(self.seated)

    def admit(self):
        """Yield people from the front of the line, one at a time, while there is a seat and enough blocks for what they have to read plus one token; each starts prefilling before the next looks up the cache."""
        while self.line and len(self.seated) < MAX_BATCH:
            seat = self.line[0]
            if seat.cancelled:
                self.line.popleft()
                continue
            try:
                seat.table, seat.cached = self.allocator.allocate_for(
                    seat.ids, len(seat.ids) + 1, use_cache=PREFIX_CACHING)
            except PoolFull:
                return
            seat.prefilled = seat.cached
            yield self.line.popleft()

    def make_room(self):
        """Give every seated user who is decoding a block for their next token; when the pool runs out, preempt the newest seated user."""
        for seat in list(self.seated):
            while seat in self.seated and not seat.reading():
                try:
                    self.allocator.grow(seat.table, seat.length + 1)
                    break
                except PoolFull:
                    self.preempt(self.seated[-1])

    def preempt(self, seat):
        """Take this user's seat and blocks away and put them at the front of the line, to read their prompt and answer so far again when seated."""
        self.allocator.free(seat.table)
        seat.table, seat.cached, seat.prefilled = [], 0, 0
        seat.ids, seat.preempted = seat.prompt_ids + seat.output[:-1], seat.preempted + 1
        self.seated.remove(seat)
        self.line.appendleft(seat)

    def prefill_step(self):
        """Spend at most PREFILL_CHUNK prompt tokens this round (0 means no limit): half-read prompts first, then new people while budget is left."""
        budget = PREFILL_CHUNK or float("inf")
        reading = [s for s in self.seated if s.reading()]
        newcomers = self.admit()
        while budget > 0:
            seat = reading.pop(0) if reading else next(newcomers, None)
            if seat is None:
                return
            if seat not in self.seated:
                self.seated.append(seat)
            count = int(min(budget, len(seat.ids) - seat.prefilled))
            budget -= count
            self.prefill(seat, count)

    def leave(self, seats):
        """Remove these users from their seats and give their blocks back to the pool."""
        for seat in seats:
            self.allocator.free(seat.table)
            seat.table = []
        self.seated = [s for s in self.seated if s not in seats]

    def accept(self, seat, token_id):
        """Send a new token to its user; finish them on the end token or when they reach max_tokens."""
        seat.last_token = token_id
        seat.output.append(token_id)
        seat.send(token_id)
        if token_id == EOS or len(seat.output) == seat.max_tokens:
            seat.done = True
            seat.send(None)

    def prefill(self, seat, count):
        """Read the next count tokens of ids, store their K and V, remember the full blocks so far, and pick the first token once a new user's prompt is done."""
        end = seat.prefilled + count
        hidden = model.prefill(seat.ids, seat.table, self.pool, start=seat.prefilled, end=end)
        seat.prefilled = end
        if PREFIX_CACHING:
            self.allocator.register(seat.table, seat.ids[:end])
        if not seat.reading():
            seat.length = len(seat.ids)
            if not seat.output:
                self.accept(seat, sample(model.logits(hidden[None]))[0])

    def decode_round(self, seats):
        """One token for every seated user: the model writes and gathers K and V through the block tables."""
        self.stopwatches.start(len(seats))
        hidden = model.decode([s.last_token for s in seats], [s.table for s in seats],
                              [s.length for s in seats], self.pool)
        logits = model.logits(hidden)
        self.stopwatches.lap("forward")
        next_tokens = sample(logits)
        self.stopwatches.lap("sampling")
        for seat, token_id in zip(seats, next_tokens):
            seat.length += 1
            self.accept(seat, token_id)
        self.stopwatches.lap("sending")


engine = BatchEngine()


@asynccontextmanager
async def lifespan(app):
    """Start the engine's thread when the server starts."""
    Thread(target=engine.run, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


async def model_tokens(prompt_ids, max_tokens):
    """Join the engine's waiting line and yield this user's tokens as they come; leaving early frees the seat."""
    seat = Seat(prompt_ids, max_tokens, asyncio.get_running_loop())
    engine.waiting.put(seat)
    try:
        while (token_id := await seat.tokens.get()) is not None:
            yield token_id
    finally:
        seat.cancelled = True


async def welcome_tokens(max_tokens):
    """Yield the welcome message's token ids and then the end token, as if the model had said it; cut at max_tokens."""
    for token_id in (tokenizer.encode(WELCOME_TEXT) + [EOS])[:max_tokens]:
        yield token_id


async def text_pieces(token_source):
    """For each token, yield the new text it adds and all ids so far; decoding the whole answer keeps split characters whole."""
    ids, sent = [], ""
    async for token_id in token_source:
        ids.append(token_id)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if text.endswith("�"):
            yield "", ids
            continue
        yield text[len(sent):], ids
        sent = text


def finish_reason(ids):
    """'stop' if the answer ended with the end token, otherwise 'length' (it hit max_tokens)."""
    return "stop" if ids and ids[-1] == EOS else "length"


def completion_object(request_id, created, text, finish, prompt_tokens, completion_tokens):
    """One OpenAI text_completion object; used for a whole answer and for each streamed piece."""
    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": MODEL_NAME,
        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


async def sse_events(prompt_ids, token_source):
    """Send one SSE message per token, then one with finish_reason and the token usage, then [DONE]."""
    request_id, created, ids = f"cmpl-{uuid.uuid4().hex}", int(time.time()), []
    async for piece, ids in text_pieces(token_source):
        chunk = completion_object(request_id, created, piece, None, len(prompt_ids), len(ids))
        chunk["usage"] = None
        yield f"data: {json.dumps(chunk)}\n\n"
    chunk = completion_object(request_id, created, "", finish_reason(ids), len(prompt_ids), len(ids))
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


async def full_answer(prompt_ids, token_source):
    """Collect every token, then return the whole answer as one object."""
    text, ids = "", []
    async for piece, ids in text_pieces(token_source):
        text += piece
    return completion_object(f"cmpl-{uuid.uuid4().hex}", int(time.time()), text, finish_reason(ids),
                             len(prompt_ids), len(ids))


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Check the request, pick the model or the welcome message, then stream the answer or send it whole."""
    prompt_ids = tokenizer.encode(request.prompt)
    if len(prompt_ids) + request.max_tokens > CONTEXT_LENGTH:
        raise HTTPException(status_code=400, detail=(
            f"prompt ({len(prompt_ids)} tokens) + max_tokens ({request.max_tokens}) is over "
            f"this server's context limit of {CONTEXT_LENGTH} tokens"))
    if request.prompt.strip():
        source = model_tokens(prompt_ids, request.max_tokens)
    else:
        source = welcome_tokens(request.max_tokens)
    if request.stream:
        return StreamingResponse(sse_events(prompt_ids, source), media_type="text/event-stream")
    return await full_answer(prompt_ids, source)


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
