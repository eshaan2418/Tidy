"""Low-dimensional ACT (action chunking transformer with CVAE) and Diffusion Policy (DDPM/DDIM over action chunks).

Both follow their papers' structure at small scale (D-005: object-centric low-dim observations, CPU/MPS budget):
  * ACT  (Zhao et al., 2023): CVAE encoder over (obs, action chunk) -> z; transformer decoder with K learned
    queries attending to [obs token, z token]; L1 reconstruction + beta*KL; z = 0 at test time; temporal
    ensembling at execution (policies/agent.py).
  * MLP: plain BC baseline (obs -> chunk, L1).
  * DP   (Chi et al., 2023): epsilon-prediction denoiser over the flattened K-step chunk conditioned on obs and
    the diffusion step (sinusoidal embedding + FiLM-style MLP); DDIM sampling at test time; receding horizon.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from policies.spec import ACT_DIM, OBS_DIM


class _Attn(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        S = m.shape[1]
        q = self.q(x).view(B, T, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(m).view(B, S, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        return self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D))


class _Block(nn.Module):
    """Pre-norm transformer block: self-attention, optional cross-attention, MLP (SDPA; ~3x faster on CPU
    than nn.TransformerDecoderLayer at this size)."""

    def __init__(self, d: int, heads: int, cross: bool, dropout: float):
        super().__init__()
        self.n1, self.sa = nn.LayerNorm(d), _Attn(d, heads)
        self.cross = cross
        if cross:
            self.n2, self.ca = nn.LayerNorm(d), _Attn(d, heads)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mem: torch.Tensor | None = None) -> torch.Tensor:
        y = self.n1(x)
        x = x + self.drop(self.sa(y, y))
        if self.cross and mem is not None:
            x = x + self.drop(self.ca(self.n2(x), mem))
        return x + self.drop(self.ff(self.n3(x)))


class ACT(nn.Module):
    def __init__(self, K: int = 10, d: int = 128, heads: int = 4, layers: int = 3, z_dim: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.K, self.z_dim = K, z_dim
        self.obs_in = nn.Linear(OBS_DIM, d)
        self.z_in = nn.Linear(z_dim, d)
        self.mem_pos = nn.Parameter(torch.randn(2, d) * 0.02)
        self.queries = nn.Parameter(torch.randn(K, d) * 0.02)
        self.mem_enc = _Block(d, heads, False, dropout)
        self.decoder = nn.ModuleList([_Block(d, heads, True, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, ACT_DIM)
        # CVAE encoder: [cls, obs, a_1..a_K] -> (mu, logvar)
        self.enc_act = nn.Linear(ACT_DIM, d)
        self.enc_obs = nn.Linear(OBS_DIM, d)
        self.enc_cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.enc_pos = nn.Parameter(torch.randn(K + 2, d) * 0.02)
        self.encoder = nn.ModuleList([_Block(d, heads, False, dropout) for _ in range(2)])
        self.to_latent = nn.Linear(d, 2 * z_dim)

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        mem = self.mem_enc(torch.stack([self.obs_in(obs), self.z_in(z)], dim=1) + self.mem_pos)
        x = self.queries.unsqueeze(0).expand(obs.shape[0], -1, -1)
        for b in self.decoder:
            x = b(x, mem)
        return self.head(self.norm(x))

    def forward(self, obs: torch.Tensor, actions: torch.Tensor | None = None):
        B = obs.shape[0]
        if actions is None:
            return self.decode(obs, torch.zeros(B, self.z_dim, device=obs.device)), None, None
        h = torch.cat([self.enc_cls.expand(B, -1, -1), self.enc_obs(obs).unsqueeze(1), self.enc_act(actions)], 1)
        h = h + self.enc_pos
        for b in self.encoder:
            h = b(h)
        mu, logvar = self.to_latent(h[:, 0]).chunk(2, dim=-1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(obs, z), mu, logvar

    def loss(self, obs, actions, mask, beta: float = 10.0) -> torch.Tensor:
        pred, mu, logvar = self(obs, actions)
        l1 = ((pred - actions).abs().mean(-1) * mask).sum() / mask.sum().clamp(min=1)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return l1 + beta * kl * 1e-3   # beta scaled by 1e-3: KL is summed over z_dim, L1 is a mean

    @torch.no_grad()
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        return self(obs)[0]


class MLPPolicy(nn.Module):
    """Plain behaviour-cloning baseline: MLP from obs to an action chunk, L1 loss (no latent, no diffusion)."""

    def __init__(self, K: int = 10, d: int = 512, layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.K = K
        mods: list[nn.Module] = []
        width = OBS_DIM
        for _ in range(layers):
            mods += [nn.Linear(width, d), nn.GELU(), nn.Dropout(dropout)]
            width = d
        self.net = nn.Sequential(*mods, nn.Linear(width, K * ACT_DIM))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).view(-1, self.K, ACT_DIM)

    def loss(self, obs, actions, mask, **_) -> torch.Tensor:
        l1 = (self(obs) - actions).abs().mean(-1)
        return (l1 * mask).sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        return self(obs)


def _time_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class _FiLMBlock(nn.Module):
    def __init__(self, d: int, c: int):
        super().__init__()
        self.lin = nn.Linear(d, d)
        self.film = nn.Linear(c, 2 * d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, c):
        g, b = self.film(c).chunk(2, -1)
        return x + self.lin(torch.nn.functional.mish(self.norm(x) * (1 + g) + b))


class DiffusionPolicy(nn.Module):
    def __init__(self, K: int = 8, d: int = 256, blocks: int = 4, T_train: int = 50):
        super().__init__()
        self.K, self.T = K, T_train
        c = 128
        self.cond = nn.Sequential(nn.Linear(OBS_DIM + 64, c), nn.Mish(), nn.Linear(c, c))
        self.inp = nn.Linear(K * ACT_DIM, d)
        self.blocks = nn.ModuleList([_FiLMBlock(d, c) for _ in range(blocks)])
        self.out = nn.Linear(d, K * ACT_DIM)
        beta = self._cosine_betas(T_train)
        alpha_bar = torch.cumprod(1 - beta, 0)
        self.alpha_bar: torch.Tensor
        self.register_buffer("alpha_bar", alpha_bar)

    @staticmethod
    def _cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
        x = torch.linspace(0, T, T + 1)
        ac = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
        ac = ac / ac[0]
        return torch.clip(1 - ac[1:] / ac[:-1], 1e-4, 0.999)

    def eps(self, x: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        c = self.cond(torch.cat([obs, _time_emb(t, 64)], -1))
        h = self.inp(x.flatten(1))
        for b in self.blocks:
            h = b(h, c)
        return self.out(h).view(-1, self.K, ACT_DIM)

    def loss(self, obs, actions, mask, **_) -> torch.Tensor:
        B = obs.shape[0]
        t = torch.randint(0, self.T, (B,), device=obs.device)
        ab = self.alpha_bar[t].view(B, 1, 1)
        noise = torch.randn_like(actions)
        x = ab.sqrt() * actions + (1 - ab).sqrt() * noise
        err = (self.eps(x, t, obs) - noise).pow(2).mean(-1)
        return (err * mask).sum() / mask.sum().clamp(min=1)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, steps: int = 10, generator: torch.Generator | None = None) -> torch.Tensor:
        """DDIM (eta = 0) sampling."""
        B = obs.shape[0]
        x = torch.randn(B, self.K, ACT_DIM, generator=generator, device=obs.device)
        ts = torch.linspace(self.T - 1, 0, steps).round().long()
        for i, t in enumerate(ts):
            ab = self.alpha_bar[t]
            e = self.eps(x, t.repeat(B).to(obs.device), obs)
            x0 = (x - (1 - ab).sqrt() * e) / ab.sqrt()
            x0 = x0.clamp(-6, 6)
            if i + 1 < len(ts):
                ab_prev = self.alpha_bar[ts[i + 1]]
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * e
            else:
                x = x0
        return x

    @torch.no_grad()
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        return self.sample(obs)
