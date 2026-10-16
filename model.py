import json
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

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

    def attention(self, x, layer, cos, sin):
        """Causal self-attention where every group of query heads shares one K/V head (GQA)."""
        c = self.config
        batch, length, _ = x.shape
        q = F.linear(x, layer["q"]).view(batch, length, c.heads, c.head_dim).transpose(1, 2)
        k = F.linear(x, layer["k"]).view(batch, length, c.kv_heads, c.head_dim).transpose(1, 2)
        v = F.linear(x, layer["v"]).view(batch, length, c.kv_heads, c.head_dim).transpose(1, 2)
        q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return F.linear(out.transpose(1, 2).reshape(batch, length, -1), layer["o"])

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

    def logits(self, hidden):
        """Scores for every vocabulary token, using the embedding table as the LM head (tied weights)."""
        return F.linear(hidden, self.embed)
