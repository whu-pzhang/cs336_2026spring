import warnings
import math

import torch
from einops import einsum, rearrange
from torch import nn


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        std = math.sqrt(2 / (d_in + d_out) ** 0.5)
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model), requires_grad=True)
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = (x / rms * self.weight).to(in_dtype)

        return result


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SiluFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        context_length: int,
        dim: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        assert dim % 2 == 0
        d = torch.arange(0, dim, 2) / dim
        freqs = theta**-d
        t = torch.arange(context_length)
        freqs = einsum(t, freqs, "i, j -> i j")  # context_length, d_k//2

        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... (pair two) -> ... pair two", two=2)
        x_real, x_imag = x[..., 0], x[..., 1]

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # 2D rotation matrix applied to pairs
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
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
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
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryPositionalEmbedding | None = None,
    ) -> None:
        super().__init__()
        if positional_encoder is None:
            warnings.warn("No positional encoder provided", stacklevel=2)
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        self.positional_encoder = positional_encoder  # RoPE

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = rearrange(q, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)
        k = rearrange(k, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)
        v = rearrange(v, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)

        if self.positional_encoder is not None:
            q = self.positional_encoder(q, token_positions)
            k = self.positional_encoder(k, token_positions)

        seq_len = x.size(-2)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).to(torch.bool)

        attn = scaled_dot_product_attention(q, k, v, causal_mask)
        attn = rearrange(attn, "... num_heads seq_len d_head -> ... seq_len (num_heads d_head)")
        result = self.output_proj(attn)
        return result


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryPositionalEmbedding | None = None,
        use_rmsnorm: bool = True,
        norm_type: str = "pre",
        ffn_type: str = "swiglu",
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        assert ffn_type in ["swiglu", "silu"]
        assert norm_type in ["pre", "post"]
        self.norm_type = norm_type

        if use_rmsnorm:
            self.ln1 = RMSNorm(d_model, eps=norm_eps)
            self.ln2 = RMSNorm(d_model, eps=norm_eps)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

        self.attn = MultiHeadAttention(d_model, num_heads, positional_encoder)

        if ffn_type == "swiglu":
            self.ffn = SwiGLUFFN(d_model, d_ff)
        else:  # ffn_type == "silu":
            self.ffn = SiluFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_positions = torch.arange(x.size(-2))
        if self.norm_type == "pre":
            x = x + self.attn(self.ln1(x), token_positions)
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x, token_positions))
            x = self.ln2(x + self.ffn(x))
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
        rope_theta: float = None,
        #
        use_rmsnorm: bool = True,
        norm_type: str = "pre",  # or "post"
        use_rope: bool = True,
        ffn_type: str = "swiglu",  # or "silu"
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size, d_model)

        d_head = d_model // num_heads
        self.positional_encoder = RotaryPositionalEmbedding(context_length, d_head, rope_theta) if use_rope else None

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    positional_encoder=self.positional_encoder,
                    use_rmsnorm=use_rmsnorm,
                    norm_type=norm_type,
                    norm_eps=norm_eps,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, eps=norm_eps) if use_rmsnorm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(x)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logsumexp subtracts the max internally and its backward only saves `logits`
    # plus the reduced result, so no vocab-sized intermediates are kept alive.
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).float().mean()
