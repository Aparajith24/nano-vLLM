import asyncio
import json
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

from blocks import DEFAULT_NUM_BLOCKS, BlockAllocator, KVPool
from model import SmolLM2

HOST, PORT = "127.0.0.1", 8080
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_BATCH = 30
NUM_BLOCKS = DEFAULT_NUM_BLOCKS
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
    """One user inside the engine: their block table, how far they have got, and where to send their tokens."""

    def __init__(self, prompt_ids, max_tokens, loop):
        self.prompt_ids, self.max_tokens, self.loop = prompt_ids, max_tokens, loop
        self.tokens = asyncio.Queue()
        self.table, self.length, self.last_token, self.generated = [], 0, None, 0
        self.done = self.cancelled = False

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
    """Continuous batching over a paged KV cache: each round seats people whose blocks fit, then gives every seated user one token."""

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
        """Free finished users, sleep if nobody is seated or waiting, seat whoever fits, then decode one token for everyone."""
        self.leave([s for s in self.seated if s.done or s.cancelled])
        if not self.seated and self.stopwatches.rounds:
            self.stopwatches.report()
        if not self.seated and not self.line:
            self.line.append(self.waiting.get())
        while not self.waiting.empty():
            self.line.append(self.waiting.get_nowait())
        try:
            with torch.no_grad():
                for seat in self.admit():
                    self.seated.append(seat)
                    self.prefill(seat)
                active = [s for s in self.seated if not s.done]
                if active:
                    self.decode_round(active)
        except Exception:
            traceback.print_exc()
            for seat in self.seated:
                seat.send(None)
            self.leave(self.seated)

    def admit(self):
        """Take people from the front of the line while there is a seat and enough free blocks for prompt + max_tokens."""
        admitted = []
        while self.line and len(self.seated) + len(admitted) < MAX_BATCH:
            seat = self.line[0]
            if seat.cancelled:
                self.line.popleft()
                continue
            needed = self.allocator.blocks_for(len(seat.prompt_ids) + seat.max_tokens)
            if needed > self.allocator.num_free():
                break
            seat.table = self.allocator.allocate(needed)
            admitted.append(self.line.popleft())
        return admitted

    def leave(self, seats):
        """Remove these users from their seats and give their blocks back to the pool."""
        for seat in seats:
            self.allocator.free(seat.table)
            seat.table = []
        self.seated = [s for s in self.seated if s not in seats]

    def accept(self, seat, token_id):
        """Send a new token to its user; finish them on the end token or when they reach max_tokens."""
        seat.last_token, seat.generated = token_id, seat.generated + 1
        seat.send(token_id)
        if token_id == EOS or seat.generated == seat.max_tokens:
            seat.done = True
            seat.send(None)

    def prefill(self, seat):
        """Read the whole prompt once, store its K and V in the user's blocks, and pick the first token."""
        hidden = model.prefill(seat.prompt_ids, seat.table, self.pool)
        seat.length = len(seat.prompt_ids)
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
