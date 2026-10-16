# nano-vLLM

A small LLM inference engine, built to understand how engines like vLLM work:
an OpenAI-compatible streaming API, continuous batching, and (in progress) our
own forward pass with a paged KV cache. Every change is measured.

It starts from a **baseline server** (`baseline_server.py`, kept unchanged so it
can be re-run for comparisons): a FastAPI server that streams tokens over SSE
from `POST /v1/completions` and batches users continuously on the Mac GPU
(MPS). Each user keeps their own KV cache, and every decode round pads all
caches to the same length, runs the model, and cuts the padding off again.
Everything after that is measured against it.

## How to run it

```
python3 server.py
```

In a second terminal:

```
curl -N -X POST localhost:8080/v1/completions -H "Content-Type: application/json" \
     -d '{"prompt": "The cat sat on the", "max_tokens": 20}'
```

`python3 baseline_server.py` runs the baseline instead, on the same port, so
run only one at a time.

Load test it (the server must be running):

```
nice -n 10 python3 bench.py --label my-run --users 1 10 30 40
```

Results are saved in `results/<label>.json`, with every token's arrival time.

## Decision log

| # | Question | My decision | Why |
|---|---|---|---|
| 1 | Where to start | The baseline server, unchanged | It already has the API, SSE and continuous batching; the project is about the engine inside |
| 2 | What to do first | **Measure a baseline** before changing anything | Without a "before" number, no change can be proven |
| 3 | Largest load test | **40 users** | 60 users made the Mac hang; 40 still tests the waiting line (30 seats) |
| 4 | Model | **SmolLM2-135M** at fp32 instead of distilgpt2 | distilgpt2 (2019) is an old design. SmolLM2 has the modern Llama design (RoPE, GQA, RMSNorm, SwiGLU), is barely bigger, and writes far better English |
| 5 | Context limit | **1,024 tokens**, our own setting | SmolLM2 allows 8,192, but 1,024 keeps the comparison with distilgpt2 fair and memory small |
| 6 | How to find the slow part | **Stopwatches** around each part of a decode round, waiting for the GPU before each reading | The GPU runs in the background; without waiting, the forward time would hide inside sampling |
| 7 | How to remove the pad/concat cost | **Write our own SmolLM2 forward pass** (option B) instead of refilling Hugging Face's cache every step (option A) | We control every copy, it is what vLLM does, and it is the only way to attack the 1-user forward overhead later. Option A stays as a fallback |
| 8 | How attention reads a paged cache | One shared pool of 16-token blocks, one block table per user, tables filled out with a reserved **dummy block 0**, **one gather per layer** for K and one for V, and a mask that hides dummy and unused slots | 60 gathers per round whatever the number of users, instead of 1,800 pads and 60 concatenates. Honest limit: the gather is still a copy; real vLLM reads blocks directly with a GPU kernel we don't have on MPS |
| 9 | How attention handles GQA (9 query heads, 3 K/V heads) | **SDPA with `enable_gqa=True`** instead of copying K/V 3× with `repeat_kv` | One call, no copy written by us, and it takes the mask paged attention will need. Measured on MPS: about the same speed as `repeat_kv`, so the choice is about simpler code, not speed |

Open: pool size, the admission rule when the pool is full, and whether to keep
the 30-seat cap.

## Results (Apple M4, 16 GB, MPS)

### Baseline: SmolLM2-135M, each user asks for 100 tokens

| Users | tok/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | per user tok/s |
|---|---|---|---|---|---|---|
| 1 | 28 | 49 ms | 49 ms | 34 ms | 52 ms | 28.2 |
| 10 | 102 | 265 ms | 401 ms | 84 ms | 223 ms | 10.3 |
| 30 | 189 | 563 ms | 1,071 ms | 164 ms | 255 ms | 6.3 |
| 40 | 168 | 821 ms | 15,502 ms | 117 ms | 272 ms | 6.1 |

distilgpt2 on the same server: 62 tok/s at 1 user, 358 at 30, and **79 at 40**.
At 40 users distilgpt2 collapsed while SmolLM2 did not; not explained yet.

### Where one decode round goes (ms per round)

| Part | 1 user | 10 users | 28 users |
|---|---|---|---|
| pad/concat | 7.9 (23%) | 23.0 (30%) | 65.8 (50%) |
| forward | 23.4 (69%) | 48.3 (63%) | 58.1 (44%) |
| sampling | 2.3 | 4.0 | 6.0 |
| slice-back | 0.1 | 0.8 | 2.4 |
| sending | 0.01 | 0.01 | 0.02 |
| whole round | 33.7 | 76.2 | 132.3 |

- **Pad/concat is the part that grows with users:** at 30 users it is 1,800
  pads and 60 concatenates per round, because each user's K and V are padded
  separately in every layer.
- **Slice-back is almost free**, because slicing makes a view, not a copy.
- **At 1 user the forward takes 23.4 ms**, while reading 540 MB of weights at
  about 120 GB/s takes about 4.5 ms. The rest is per-operation overhead across
  30 layers; paging will not fix that.
- **Best case if pad/concat disappears:** 66.5 ms per round, about 421 tok/s at
  28 users instead of 212.

### Our own forward pass (`model.py`) against Hugging Face

`python3 test_forward.py` compares each piece of layer 0 and the last-position
logits of 5 prompts (1 to 260 tokens) with Hugging Face's `LlamaForCausalLM`.

| | CPU max diff | MPS max diff |
|---|---|---|
| RMSNorm, MLP | 0 | 0 |
| RoPE, attention | 0 | up to 3.2e-6 |
| Logits | 0 | up to 2.1e-5 |

The next token is the same for every prompt. To check that the test can fail,
RoPE was broken on purpose (theta 10,000 instead of 100,000): the logits moved
by 1.49 and the test failed, **even though the top token stayed the same**. So
comparing only the chosen token would have missed a real bug.

## Limits

- A tiny model on a Mac GPU, fp32, one machine, no tensor parallelism.
- The load test client runs on the same Mac as the server.
