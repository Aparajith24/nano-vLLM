import json
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from blocks import DUMMY_BLOCK

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


class Config:
    """The numbers that shape the model, read from its config.json."""

    def __init__(self, raw):
        self.layers = raw["num_hidden_layers"]
        self.hidden = raw["hidden_size"]
        self.heads = raw["num_attention_heads"]
        self.kv_heads = raw["num_key_value_heads"]
        self.head_dim = raw.get("head_dim", self.hidden // self.heads)
        self.eps = raw["rms_norm_eps"]
        self.rope_theta = raw["rope_theta"]
        self.vocab = raw["vocab_size"]


def rms_norm(x, weight, eps):
    """Scale each vector so its root-mean-square is 1, then multiply by a learned weight per number."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def rope_tables(positions, head_dim, theta):
    """The cos and sin of every position's rotation angles, with dim i paired with dim i + head_dim / 2."""
    speeds = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    angles = positions[..., None].float() * speeds
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def rotate(x, cos, sin):
    """Turn every pair of numbers (i, i + half) in x by its position's angle."""
    half = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin


def decode_attention(q, k, v, mask):
    """Attention for one new token per user; each group of query heads is reshaped onto its shared K/V head, so nothing is copied."""
    users, heads, _, head_dim = q.shape
    groups = k.shape[1]
    q = q.view(users, groups, heads // groups, head_dim)
    scores = (q @ k.transpose(-1, -2)) / head_dim ** 0.5
    weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    return (weights @ v).view(users, heads, 1, head_dim)


class SmolLM2:
    """Our own SmolLM2 forward pass: plain tensors and functions, no cache yet."""

    def __init__(self, config, weights):
        self.config = config
        self.embed = weights["model.embed_tokens.weight"]
        self.norm = weights["model.norm.weight"]
        self.layers = []
        for i in range(config.layers):
            name = f"model.layers.{i}."
            self.layers.append({
                "attn_norm": weights[name + "input_layernorm.weight"],
                "q": weights[name + "self_attn.q_proj.weight"],
                "k": weights[name + "self_attn.k_proj.weight"],
                "v": weights[name + "self_attn.v_proj.weight"],
                "o": weights[name + "self_attn.o_proj.weight"],
                "mlp_norm": weights[name + "post_attention_layernorm.weight"],
                "gate": weights[name + "mlp.gate_proj.weight"],
                "up": weights[name + "mlp.up_proj.weight"],
                "down": weights[name + "mlp.down_proj.weight"],
            })

    @classmethod
    def from_pretrained(cls, name=MODEL_NAME, device="cpu"):
        """Load the config and weights from the Hugging Face checkpoint, as fp32 on the given device."""
        folder = Path(snapshot_download(name, allow_patterns=["*.json", "*.safetensors"]))
        config = Config(json.loads((folder / "config.json").read_text()))
        weights = {key: value.to(device, torch.float32) for key, value in load_file(folder / "model.safetensors").items()}
        return cls(config, weights)

    def qkv(self, x, layer, cos, sin):
        """Project x to Q, K and V shaped [batch, heads, length, head_dim], with RoPE applied to Q and K."""
        c = self.config
        batch, length, _ = x.shape
        q = F.linear(x, layer["q"]).view(batch, length, c.heads, c.head_dim).transpose(1, 2)
        k = F.linear(x, layer["k"]).view(batch, length, c.kv_heads, c.head_dim).transpose(1, 2)
        v = F.linear(x, layer["v"]).view(batch, length, c.kv_heads, c.head_dim).transpose(1, 2)
        return rotate(q, cos, sin), rotate(k, cos, sin), v

    def attention(self, x, layer, cos, sin):
        """Causal self-attention where every group of query heads shares one K/V head (GQA)."""
        q, k, v = self.qkv(x, layer, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return F.linear(out.transpose(1, 2).flatten(2), layer["o"])

    def mlp(self, x, layer):
        """SwiGLU: a SiLU-gated path times a plain path, projected back down."""
        return F.linear(F.silu(F.linear(x, layer["gate"])) * F.linear(x, layer["up"]), layer["down"])

    def block(self, x, layer, cos, sin):
        """One transformer layer: norm, attention, add back; norm, MLP, add back."""
        x = x + self.attention(rms_norm(x, layer["attn_norm"], self.config.eps), layer, cos, sin)
        return x + self.mlp(rms_norm(x, layer["mlp_norm"], self.config.eps), layer)

    def forward(self, ids):
        """Run token ids of shape (batch, length) through every layer and the final norm."""
        positions = torch.arange(ids.shape[1], device=ids.device).expand(ids.shape[0], -1)
        cos, sin = rope_tables(positions, self.config.head_dim, self.config.rope_theta)
        cos, sin = cos[:, None], sin[:, None]
        x = F.embedding(ids, self.embed)
        for layer in self.layers:
            x = self.block(x, layer, cos, sin)
        return rms_norm(x, self.norm, self.config.eps)

    def prefill(self, ids, table, pool):
        """Run one user's prompt, store its K and V in their blocks, and return the last position's hidden state."""
        length = len(ids)
        ids = torch.tensor([ids], device=pool.device)
        positions = torch.arange(length, device=pool.device)[None]
        cos, sin = rope_tables(positions, self.config.head_dim, self.config.rope_theta)
        cos, sin = cos[:, None], sin[:, None]
        blocks, slots = pool.locations(table, 0, length)
        x = F.embedding(ids, self.embed)
        for i, layer in enumerate(self.layers):
            h = rms_norm(x, layer["attn_norm"], self.config.eps)
            q, k, v = self.qkv(h, layer, cos, sin)
            pool.write(i, blocks, slots, k[0].transpose(0, 1), v[0].transpose(0, 1))
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            x = x + F.linear(out.transpose(1, 2).flatten(2), layer["o"])
            x = x + self.mlp(rms_norm(x, layer["mlp_norm"], self.config.eps), layer)
        return rms_norm(x[:, -1], self.norm, self.config.eps)[0]

    def decode(self, tokens, tables, lengths, pool):
        """One step for many users: write each new token's K and V into the pool, gather everyone's blocks, attend."""
        users, bs, device = len(tokens), pool.block_size, pool.device
        used = [n // bs + 1 for n in lengths]
        width = max(used)
        index = torch.tensor([table[:u] + [DUMMY_BLOCK] * (width - u) for table, u in zip(tables, used)], device=device)
        blocks = torch.tensor([table[n // bs] for table, n in zip(tables, lengths)], device=device)
        slots = torch.tensor([n % bs for n in lengths], device=device)
        positions = torch.tensor(lengths, device=device)
        mask = (torch.arange(width * bs, device=device)[None] <= positions[:, None])[:, None, None]
        cos, sin = rope_tables(positions[:, None], self.config.head_dim, self.config.rope_theta)
        cos, sin = cos[:, None], sin[:, None]
        x = F.embedding(torch.tensor(tokens, device=device)[:, None], self.embed)
        for i, layer in enumerate(self.layers):
            h = rms_norm(x, layer["attn_norm"], self.config.eps)
            q, k, v = self.qkv(h, layer, cos, sin)
            pool.write(i, blocks, slots, k[:, :, 0], v[:, :, 0])
            k_all = pool.k[i][index].flatten(1, 2).transpose(1, 2)
            v_all = pool.v[i][index].flatten(1, 2).transpose(1, 2)
            out = decode_attention(q, k_all, v_all, mask)
            x = x + F.linear(out.transpose(1, 2).flatten(2), layer["o"])
            x = x + self.mlp(rms_norm(x, layer["mlp_norm"], self.config.eps), layer)
        return rms_norm(x[:, 0], self.norm, self.config.eps)

    def logits(self, hidden):
        """Scores for every vocabulary token, using the embedding table as the LM head (tied weights)."""
        return F.linear(hidden, self.embed)
