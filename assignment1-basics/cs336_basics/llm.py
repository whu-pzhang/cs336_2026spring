import torch
from einops import einsum, rearrange
from torch import nn
from typing import Optional


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.randn(out_features, in_features, device=self.device, dtype=self.dtype))

        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=2 / (self.in_features + self.out_features) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=self.device, dtype=self.dtype))

        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = (x / rms * self.weight).to(in_dtype)

        return result


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = (x1 * torch.sigmoid(x1)) * self.w3(x)
        result = self.w2(x2)
        return result


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) -> None:
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        pos = torch.arange(max_seq_len, device=self.device)
        inv_freq = 1 / theta ** (torch.arange(0, d_k, 2) / d_k)
        inv_freq = inv_freq.to(self.device)

        angles = einsum(pos, inv_freq, "i, j -> i j")  # max_seq_len, d_k//2

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... (pair two) -> ... pair two", two=2)
        x_real, x_imag = x[..., 0], x[..., 1]

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x_real_rot = x_real * cos - x_imag * sin
        x_imag_rot = x_real * sin + x_imag * cos

        x_rot = torch.stack([x_real_rot, x_imag_rot], dim=-1)
        x_rot = rearrange(x_rot, "... pairs two -> ... (pairs two)")

        return x_rot


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    x_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_sum


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    d_k = k.size(-1)
    qk = einsum(q, k, "... q d_k, ... k d_k -> ... q k")
    qk = qk / d_k**0.5

    if mask is not None:
        qk = qk.masked_fill(~mask, float("-inf"))
    qk = softmax(qk, dim=-1)
    result = einsum(qk, v, "... q k, ... k d_v -> ... q d_v")
    return result


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = None, max_seq_len: int = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        if theta and max_seq_len:
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len)
        else:
            self.rope = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = rearrange(q, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = x.size(-2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).to(torch.bool)

        attn = scaled_dot_product_attention(q, k, v, causal_mask)
        attn = rearrange(attn, "... num_heads seq_len d_head -> ... seq_len (num_heads d_head)")
        result = self.output_proj(attn)
        return result


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = None, max_seq_len: int = None) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, theta, max_seq_len)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(x.size(-2))
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            block = TransformerBlock(d_model, num_heads, d_ff, theta, max_seq_len=context_length)
            self.layers.append(block)

        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
