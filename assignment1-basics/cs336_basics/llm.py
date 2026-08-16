import torch
from einops import einsum, rearrange
from torch import nn


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

        self.weights = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        result = (x / rms * self.weights).to(in_dtype)

        return result


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        self.w1_weight = nn.Parameter(torch.randn(d_ff, d_model, device=self.device, dtype=self.dtype))
        self.w2_weight = nn.Parameter(torch.randn(d_model, d_ff, device=self.device, dtype=self.dtype))
        self.w3_weight = nn.Parameter(torch.randn(d_ff, d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = einsum(x, self.w1_weight, "... d_model, d_ff d_model -> ... d_ff")
        x2 = (x1 * torch.sigmoid(x1)) * einsum(x, self.w3_weight, "... d_model, d_ff d_model -> ... d_ff")
        result = einsum(x2, self.w2_weight, "... d_ff, d_model d_ff -> ... d_model")
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
