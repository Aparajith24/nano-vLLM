import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from blocks import BlockAllocator, KVPool
from model import MODEL_NAME, SmolLM2

TOLERANCE = {"cpu": 5e-4, "mps": 1e-3}
STEPS = 40


def reference(hf, prompt_ids, steps, device):
    """Hugging Face's greedy continuation of a prompt, and its logits at every position of prompt + continuation."""
    ids = torch.tensor([prompt_ids], device=device)
    full = hf.generate(ids, max_new_tokens=steps, min_new_tokens=steps, do_sample=False, pad_token_id=0)[0].tolist()
    logits = hf(torch.tensor([full], device=device)).logits[0]
    return full[len(prompt_ids):], logits


def garbage_pool(ours, num_blocks, device):
    """A KV pool and allocator whose memory is filled with large finite garbage, so a wrong read cannot look right."""
    c = ours.config
    pool = KVPool(c.layers, c.kv_heads, c.head_dim, num_blocks=num_blocks, device=device)
    for layer in range(c.layers):
        pool.k[layer].copy_(torch.randn_like(pool.k[layer]) * 1e4)
        pool.v[layer].copy_(torch.randn_like(pool.v[layer]) * 1e4)
    return pool, BlockAllocator(num_blocks=num_blocks)


class User:
    """One test user: prompt, forced continuation, reference logits, block table, and the worst difference seen."""

    def __init__(self, name, prompt_ids, continuation, logits):
        self.name, self.prompt_ids, self.continuation, self.logits = name, prompt_ids, continuation, logits
        self.table, self.length, self.step, self.worst, self.same = [], 0, 0, 0.0, True

    def compare(self, ours_logits, position):
        """Record how far our logits at this position are from Hugging Face's."""
        theirs = self.logits[position]
        self.worst = max(self.worst, (ours_logits - theirs).abs().max().item())
        self.same &= ours_logits.argmax().item() == theirs.argmax().item()


def prefill(ours, user, pool, allocator):
    """Seat a user: blocks for the prompt, paged prefill, compare the last prompt position."""
    allocator.grow(user.table, len(user.prompt_ids))
    hidden = ours.prefill(user.prompt_ids, user.table, pool)
    user.compare(ours.logits(hidden), len(user.prompt_ids) - 1)
    user.length = len(user.prompt_ids)


def decode_round(ours, users, pool, allocator):
    """One batched decode step; every user is fed their forced next token (teacher forcing)."""
    for user in users:
        allocator.grow(user.table, user.length + 1)
    tokens = [user.continuation[user.step] for user in users]
    hidden = ours.decode(tokens, [u.table for u in users], [u.length for u in users], pool)
    logits = ours.logits(hidden)
    for row, user in enumerate(users):
        user.compare(logits[row], user.length)
        user.length, user.step = user.length + 1, user.step + 1


def report(title, users, device):
    """Print each user's worst difference and whether every chosen token matched; return True if all passed."""
    print(f"  {title}")
    ok = True
    for user in users:
        passed = user.worst <= TOLERANCE[device] and user.same
        ok &= passed
        print(f"    {'ok ' if passed else 'BAD'} {user.name:<34} final length {user.length:>3}  "
              f"blocks {user.table}  max diff {user.worst:.2e}  same tokens {user.same}")
    return ok


def run(device, tokenizer):
    """The one-user and several-user checks on one device."""
    hf = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device).eval()
    ours = SmolLM2.from_pretrained(device=device)
    ok = True
    with torch.no_grad():
        prompt = tokenizer.encode("The history of the printing press begins in")
        alone = User(f"alone, {len(prompt)}-token prompt", prompt, *reference(hf, prompt, STEPS, device))
        pool, allocator = garbage_pool(ours, 40, device)
        prefill(ours, alone, pool, allocator)
        for _ in range(STEPS):
            decode_round(ours, [alone], pool, allocator)
        ok &= report("one user, teacher forced across block boundaries", [alone], device)
        allocator.free(alone.table)
        allocator.check()

        prompts = ["Hi", "The cat sat on the", "Water boils at a temperature of one hundred degrees",
                   " ".join(["Paged attention keeps the cache in small blocks."] * 4)]
        users = []
        for prompt in prompts:
            ids = tokenizer.encode(prompt)
            users.append(User(f"{len(ids)}-token prompt", ids, *reference(hf, ids, STEPS, device)))
        late = users.pop(1)
        late.name += ", joins after 10 steps"
        pool, allocator = garbage_pool(ours, 40, device)
        for user in users:
            prefill(ours, user, pool, allocator)
        for step in range(STEPS):
            if step == 10:
                prefill(ours, late, pool, allocator)
                users.append(late)
            decode_round(ours, [u for u in users if u.step < STEPS], pool, allocator)
        ok &= report("four users of different lengths decoded together", users, device)
        for user in users:
            allocator.free(user.table)
        allocator.check()
        print(f"  all blocks back in the pool: {allocator.num_free()} of {allocator.num_blocks - 1}")
        ok &= prefix_case(ours, hf, tokenizer, device)
        ok &= chunked_case(ours, hf, tokenizer, device)
    return ok


def chunked_case(ours, hf, tokenizer, device, chunk=7):
    """Prefill one prompt a few tokens at a time; every chunk's last position, and the decode after it, must match Hugging Face."""
    ids = tokenizer.encode(" ".join(["Chunked prefill reads a long prompt a few tokens per round."] * 6))
    pool, allocator = garbage_pool(ours, 40, device)
    user = User(f"{len(ids)}-token prompt in chunks of {chunk}", ids, *reference(hf, ids, 20, device))
    allocator.grow(user.table, len(ids))
    for start in range(0, len(ids), chunk):
        end = min(start + chunk, len(ids))
        user.compare(ours.logits(ours.prefill(ids, user.table, pool, start=start, end=end)), end - 1)
    user.length = len(ids)
    for _ in range(20):
        decode_round(ours, [user], pool, allocator)
    ok = report(f"chunked prefill ({-(-len(ids) // chunk)} chunks, crossing every block boundary)", [user], device)
    allocator.free(user.table)
    allocator.check()
    return ok


def prefix_case(ours, hf, tokenizer, device):
    """User B starts with the same long text as user A, reuses A's cached blocks, and must still match Hugging Face."""
    system = " ".join(["You are a helpful assistant who answers questions about printing presses."] * 3)
    a_ids = tokenizer.encode(system + " Who invented it?")
    b_ids = tokenizer.encode(system + " When did it reach England?")
    pool, allocator = garbage_pool(ours, 40, device)
    first = User("A, nothing cached yet", a_ids, *reference(hf, a_ids, 20, device))
    first.table, cached_a = allocator.allocate_for(a_ids, len(a_ids) + 20)
    first.compare(ours.logits(ours.prefill(a_ids, first.table, pool)), len(a_ids) - 1)
    first.length = len(a_ids)
    allocator.register(first.table, a_ids)
    second = User("B, shares A's opening", b_ids, *reference(hf, b_ids, 20, device))
    second.table, cached_b = allocator.allocate_for(b_ids, len(b_ids) + 20)
    second.compare(ours.logits(ours.prefill(b_ids, second.table, pool, start=cached_b)), len(b_ids) - 1)
    second.length = len(b_ids)
    for _ in range(20):
        decode_round(ours, [first, second], pool, allocator)
    shared = [block for block in second.table if block in first.table]
    print(f"    prompts of {len(a_ids)} and {len(b_ids)} tokens; B reused {cached_b} cached tokens in blocks {shared}, "
          f"prefilled only {len(b_ids) - cached_b}")
    ok = report("prefix caching", [first, second], device) and cached_a == 0 and cached_b > 0
    allocator.free(first.table)
    allocator.free(second.table)
    allocator.check()
    return ok


def main(devices):
    """Run every paged check on every device and exit with an error if any fails."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ok = True
    for device in devices:
        print(f"device: {device}")
        ok &= run(device, tokenizer)
    print("all paged checks passed" if ok else "SOME PAGED CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["cpu", "mps"])
