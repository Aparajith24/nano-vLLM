import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from model import MODEL_NAME, SmolLM2, rms_norm, rope_tables, rotate

PROMPTS = [
    "The cat sat on the",
    "Hello",
    "In the year 2050, cities around the world will have",
    "Water boils at",
    " ".join(["The history of the printing press is long and full of surprises."] * 20),
]
TOLERANCE = {"cpu": 1e-4, "mps": 1e-3}


def max_diff(a, b):
    """The largest absolute difference between two tensors."""
    return (a - b).abs().max().item()


def check(name, ours, theirs, device):
    """Print how far our result is from Hugging Face's, and whether it is within tolerance."""
    diff = max_diff(ours, theirs)
    ok = diff <= TOLERANCE[device]
    print(f"  {'ok ' if ok else 'BAD'} {name:<28} max diff {diff:.2e}")
    return ok


def check_pieces(ours, hf, device):
    """Compare RMSNorm, RoPE, attention and the MLP of layer 0 with Hugging Face's modules on the same random input."""
    torch.manual_seed(0)
    c, layer, hf_layer = ours.config, ours.layers[0], hf.model.layers[0]
    x = torch.randn(2, 7, c.hidden, device=device)
    positions = torch.arange(7, device=device).expand(2, -1)
    hf_cos, hf_sin = hf.model.rotary_emb(x, positions)
    cos, sin = rope_tables(positions, c.head_dim, c.rope_theta)
    q = torch.randn(2, c.heads, 7, c.head_dim, device=device)
    k = torch.randn(2, c.kv_heads, 7, c.head_dim, device=device)
    hf_q, hf_k = apply_rotary_pos_emb(q, k, hf_cos, hf_sin)
    results = [
        check("RMSNorm", rms_norm(x, layer["attn_norm"], c.eps), hf_layer.input_layernorm(x), device),
        check("RoPE cos/sin", torch.cat([cos, sin]), torch.cat([hf_cos, hf_sin]), device),
        check("RoPE on Q", rotate(q, cos[:, None], sin[:, None]), hf_q, device),
        check("RoPE on K", rotate(k, cos[:, None], sin[:, None]), hf_k, device),
        check("attention (GQA, causal)", ours.attention(x, layer, cos[:, None], sin[:, None]),
              hf_layer.self_attn(x, position_embeddings=(hf_cos, hf_sin), attention_mask=None)[0], device),
        check("MLP (SwiGLU)", ours.mlp(x, layer), hf_layer.mlp(x), device),
    ]
    return all(results)


def check_logits(ours, hf, tokenizer, device):
    """Compare last-position logits for every prompt; on a mismatch, find the first layer that differs."""
    results = []
    for prompt in PROMPTS:
        ids = torch.tensor([tokenizer.encode(prompt)], device=device)
        theirs = hf(ids, logits_to_keep=1).logits[:, -1]
        mine = ours.logits(ours.forward(ids)[:, -1])
        ok = check(f"logits, {ids.shape[1]} tokens", mine, theirs, device)
        same_token = mine.argmax(-1).item() == theirs.argmax(-1).item()
        print(f"      same next token: {same_token}")
        results.append(ok and same_token)
        if not ok:
            find_bad_layer(ours, hf, ids)
    return all(results)


def find_bad_layer(ours, hf, ids):
    """Compare hidden states after every layer and print the first one that drifts."""
    hf_states = hf(ids, output_hidden_states=True, logits_to_keep=1).hidden_states
    positions = torch.arange(ids.shape[1], device=ids.device)[None]
    cos, sin = rope_tables(positions, ours.config.head_dim, ours.config.rope_theta)
    x = torch.nn.functional.embedding(ids, ours.embed)
    for i, layer in enumerate(ours.layers):
        x = ours.block(x, layer, cos[:, None], sin[:, None])
        diff = max_diff(x, hf_states[i + 1]) if i + 1 < len(hf_states) - 1 else None
        if diff is not None and diff > 1e-3:
            print(f"      first bad layer: {i}, max diff {diff:.2e}")
            return


def main(devices):
    """Run every check on every device and exit with an error if any fails."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    passed = True
    for device in devices:
        print(f"device: {device}")
        hf = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device).eval()
        ours = SmolLM2.from_pretrained(device=device)
        with torch.no_grad():
            passed &= check_pieces(ours, hf, device)
            passed &= check_logits(ours, hf, tokenizer, device)
        del hf, ours
    print("all checks passed" if passed else "SOME CHECKS FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main(sys.argv[1:] or ["cpu", "mps"])
