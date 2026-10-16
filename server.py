import asyncio
import json
import queue
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from threading import Thread

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

HOST, PORT = "127.0.0.1", 8080
MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_BATCH = 30
CONTEXT_LENGTH = 1024
DEFAULT_MAX_TOKENS = 16
TEMPERATURE, TOP_K = 0.7, 10
WELCOME_TEXT = "hi! welcome"
PARTS = ("pad/concat", "forward", "sampling", "slice-back", "sending")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(DEVICE).eval()
N_LAYERS = model.config.num_hidden_layers
EOS = tokenizer.eos_token_id


class CompletionRequest(BaseModel):
    """What the client sends: a non-empty prompt, optionally max_tokens, and whether to stream (yes by default)."""

    prompt: str = Field(min_length=1)
    max_tokens: int = Field(DEFAULT_MAX_TOKENS, ge=1)
    stream: bool = True


class Seat:
    """One user inside the engine: their KV cache, how far they have got, and where to send their tokens."""

    def __init__(self, prompt_ids, max_tokens, loop):
        self.prompt_ids, self.max_tokens, self.loop = prompt_ids, max_tokens, loop
        self.tokens = asyncio.Queue()
        self.cache, self.length, self.last_token, self.generated = None, 0, None, 0
        self.done = self.cancelled = False

    def send(self, token_id):
        """Hand a token id, or None for the end, to this user's request handler on the event loop."""
        self.loop.call_soon_threadsafe(self.tokens.put_nowait, token_id)


def sample(logits):
    """Pick one token per row: keep the TOP_K likeliest, sharpen with TEMPERATURE, then draw at random."""
    top_logits, top_ids = torch.topk(logits / TEMPERATURE, TOP_K)
    choice = torch.multinomial(torch.softmax(top_logits, dim=-1), 1)
    return top_ids.gather(-1, choice).view(-1).tolist()


def layer_kv(cache):
    """The (keys, values) pair of every layer in a transformers cache."""
    return [(layer.keys, layer.values) for layer in cache.layers]


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
    """Continuous batching: one thread runs rounds; each round seats people from the waiting line, then gives every seated user one token."""

    def __init__(self):
        self.waiting, self.seated, self.stopwatches = queue.Queue(), [], Stopwatches()

    def run(self):
        """Run rounds forever; when nobody is seated, sleep until someone joins the waiting line."""
        while True:
            self.seated = [s for s in self.seated if not s.done and not s.cancelled]
            if not self.seated and self.stopwatches.rounds:
                self.stopwatches.report()
            new = [] if self.seated else [self.waiting.get()]
            while len(self.seated) + len(new) < MAX_BATCH and not self.waiting.empty():
                new.append(self.waiting.get_nowait())
            try:
                with torch.no_grad():
                    for seat in new:
                        if not seat.cancelled:
                            self.prefill(seat)
                            self.seated.append(seat)
                    active = [s for s in self.seated if not s.done]
                    if active:
                        self.decode_round(active)
            except Exception:
                traceback.print_exc()
                for seat in self.seated + new:
                    seat.send(None)
                self.seated = []

    def accept(self, seat, token_id):
        """Send a new token to its user; finish them on the end token or when they reach max_tokens."""
        seat.last_token, seat.generated = token_id, seat.generated + 1
        seat.send(token_id)
        if token_id == EOS or seat.generated == seat.max_tokens:
            seat.done = True
            seat.send(None)

    def prefill(self, seat):
        """Read the whole prompt once, keep its KV cache, and pick the first token."""
        out = model.model(torch.tensor([seat.prompt_ids], device=DEVICE), use_cache=True)
        seat.cache, seat.length = layer_kv(out.past_key_values), len(seat.prompt_ids)
        self.accept(seat, sample(model.lm_head(out.last_hidden_state[:, -1]))[0])

    def decode_round(self, seats):
        """Left-pad every user's KV cache to the same length, run one step for all of them, then cut the padding off again."""
        self.stopwatches.start(len(seats))
        longest = max(s.length for s in seats)
        cache = DynamicCache()
        for layer in range(N_LAYERS):
            keys = torch.cat([F.pad(s.cache[layer][0], (0, 0, longest - s.length, 0)) for s in seats])
            values = torch.cat([F.pad(s.cache[layer][1], (0, 0, longest - s.length, 0)) for s in seats])
            cache.update(keys, values, layer)
        mask = torch.tensor([[0] * (longest - s.length) + [1] * (s.length + 1) for s in seats], device=DEVICE)
        positions = torch.tensor([[s.length] for s in seats], device=DEVICE)
        inputs = torch.tensor([[s.last_token] for s in seats], device=DEVICE)
        self.stopwatches.lap("pad/concat")
        out = model.model(input_ids=inputs, past_key_values=cache, attention_mask=mask,
                          position_ids=positions, use_cache=True)
        logits = model.lm_head(out.last_hidden_state[:, -1])
        self.stopwatches.lap("forward")
        next_tokens = sample(logits)
        self.stopwatches.lap("sampling")
        batch_kv = layer_kv(out.past_key_values)
        for i, seat in enumerate(seats):
            start = longest - seat.length
            seat.cache = [(k[i:i + 1, :, start:], v[i:i + 1, :, start:]) for k, v in batch_kv]
            seat.length += 1
        self.stopwatches.lap("slice-back")
        for seat, token_id in zip(seats, next_tokens):
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
