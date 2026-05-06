from __future__ import annotations

import torch


class DiffusionScheduler:
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str | None = None,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.device = device
        betas = torch.linspace(beta_start, beta_end, self.num_train_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def to(self, device: torch.device | str) -> "DiffusionScheduler":
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self

    def sample_timesteps(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return torch.randint(0, self.num_train_timesteps, (batch_size,), device=device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        coeff_x0 = self.sqrt_alpha_bars[timesteps].view(-1, 1, 1)
        coeff_noise = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1, 1)
        xt = coeff_x0 * x0 + coeff_noise * noise
        return xt, noise

