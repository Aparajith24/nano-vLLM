import torch
from transformers import AutoModelForCausalLM

import server

SYSTEM = " ".join(["You are a helpful assistant who answers questions about the history of printing."] * 3)
PROMPTS = ["The cat sat on the", "Hello", "In the year 2050, cities around the world will have", "Water boils at",
           SYSTEM + " Who invented the press?", SYSTEM + " When did it reach England?",
           SYSTEM + " What did it change?"]
MAX_TOKENS = 20


class NowLoop:
    """Stands in for the event loop: delivers each token straight away, on the engine's thread."""

    def call_soon_threadsafe(self, fn, *args):
        """Call fn right now."""
        fn(*args)


def answer(seat):
    """Every token id the engine sent to this user, without the end-of-stream None."""
    ids = []
    while not seat.tokens.empty():
        token_id = seat.tokens.get_nowait()
        if token_id is not None:
            ids.append(token_id)
    return ids


def run_engine(num_blocks, chunk):
    """Send every prompt to a fresh engine with this many blocks and prefill budget, run rounds until all are done, and return the answers and preemption count."""
    server.NUM_BLOCKS, server.PREFILL_CHUNK = num_blocks, chunk
    engine = server.BatchEngine()
    seats = [server.Seat(server.tokenizer.encode(p), MAX_TOKENS, NowLoop()) for p in PROMPTS]
    for seat in seats:
        engine.waiting.put(seat)
    most_seated, rounds = 0, 0
    while any(not s.done for s in seats):
        engine.run_round()
        most_seated, rounds = max(most_seated, len(engine.seated)), rounds + 1
    engine.leave(list(engine.seated))
    engine.allocator.check()
    print(f"  pool of {num_blocks} blocks, prefill chunk {chunk or 'unlimited'}: {rounds} rounds, at most {most_seated} seated at once, "
          f"{engine.allocator.num_free()} of {num_blocks - 1} blocks free at the end, "
          f"cached prompt tokens reused per user: {[s.cached for s in seats]}, preemptions per user: {[s.preempted for s in seats]}")
    return [answer(s) for s in seats], sum(s.preempted for s in seats)


def main():
    """Greedy answers through the paged engine must equal Hugging Face's generate, with big and tiny pools and prefill budgets; tiny pools must preempt."""
    server.sample = lambda logits: logits.argmax(-1).view(-1).tolist()
    hf = AutoModelForCausalLM.from_pretrained(server.MODEL_NAME, dtype=torch.float32).to(server.DEVICE).eval()
    expected = []
    for prompt in PROMPTS:
        ids = torch.tensor([server.tokenizer.encode(prompt)], device=server.DEVICE)
        out = hf.generate(ids, max_new_tokens=MAX_TOKENS, do_sample=False, pad_token_id=server.EOS)[0, ids.shape[1]:].tolist()
        expected.append(out[:out.index(server.EOS) + 1] if server.EOS in out else out)
    ok = True
    for num_blocks, chunk in ((64, 0), (9, 0), (64, 8), (64, 512), (6, 0), (9, 8)):
        answers, preemptions = run_engine(num_blocks, chunk)
        if num_blocks < 10 and not preemptions:
            ok = False
            print("    FAIL a tiny pool never preempted anyone")
        for prompt, got, want in zip(PROMPTS, answers, expected):
            same = got == want
            ok &= same
            print(f"    {'MATCH' if same else 'DIFF '} {prompt!r} -> {server.tokenizer.decode(got)!r}")
    print("all engine checks passed" if ok else "SOME ENGINE CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
