import torch

def beta_schedule(
    T: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    schedule_type: str = "linear",
) -> torch.Tensor:
    """
    Generate a beta schedule for a diffusion process.

    The beta schedule defines the variance of the forward diffusion
    process at each timestep. Different schedules influence training
    stability and sample quality.

    Args:
        T (int):
            Total number of diffusion timesteps. Must be strictly positive.

        beta_start (float, optional):
            Initial beta value (used in schedules that require a range).
            Default is 1e-4.

        beta_end (float, optional):
            Final beta value (used in schedules that require a range).
            Default is 2e-2.

        schedule_type (str, optional):
            Type of schedule to generate. Supported values may include:

            - "linear": Linearly spaced betas from beta_start to beta_end.
            - "cosine": Cosine schedule based on a cumulative alpha_bar
              formulation (e.g., Nichol & Dhariwal style).

            Default is "linear".

    Returns:
        torch.Tensor:
            A 1D tensor of shape (T,) containing the beta values
            for each diffusion timestep. The tensor is typically
            of dtype torch.float32.
    """
    if T <= 0:
        raise ValueError(f"`T` must be > 0, got {T}")

    if schedule_type == "linear":
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)

    if schedule_type == "cosine":
        # Nichol & Dhariwal-style cosine alpha_bar schedule.
        s = 0.008
        steps = T + 1
        x = torch.linspace(0, T, steps, dtype=torch.float32)
        alpha_bar = torch.cos(((x / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(1e-5, 0.999)

    raise ValueError(f"Unknown schedule_type={schedule_type}")

class DDPM:
    """
    Denoising Diffusion Probabilistic Model (DDPM) with epsilon-prediction.

    Implements:
        - Forward diffusion q(x_t | x_0)
        - Reverse sampling p_theta(x_{t-1} | x_t)
        - Full sampling loop

    Uses the standard noise-prediction parameterization.
    """

    def __init__(
        self,
        betas: torch.Tensor | None = None,
        T: int = 1000,
        device: str | torch.device = "cuda",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        schedule_type: str = "linear",
    ) -> None:
        """
        Initialize diffusion process.

        Args:
            betas: Optional precomputed beta schedule of shape (T,).
            T: Number of diffusion timesteps.
            device: Device for tensors.
            beta_start: Starting beta value (if schedule generated internally).
            beta_end: Final beta value (if schedule generated internally).
            schedule_type: Type of beta schedule ("linear", "cosine", ...).
        """
        self.device = torch.device(device)
        self.T = int(T)
        if self.T <= 0:
            raise ValueError(f"`T` must be > 0, got {self.T}")
        # Apple MPS does not support float64 tensors.
        self.math_dtype = torch.float32 if self.device.type == "mps" else torch.float64

        if betas is None:
            betas = beta_schedule(
                T=self.T,
                beta_start=beta_start,
                beta_end=beta_end,
                schedule_type=schedule_type,
            )
        # Keep diffusion coefficients in higher precision when available.
        betas = betas.to(self.device).to(self.math_dtype)
        if betas.ndim != 1 or betas.shape[0] != self.T:
            raise ValueError(f"`betas` must have shape ({self.T},), got {tuple(betas.shape)}")

        self.betas = torch.clamp(betas, min=1e-12, max=0.999)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        # Prevent exact zeros on float32 backends (notably MPS), which cause 0/0 in sampling updates.
        self.alpha_bars = torch.clamp(self.alpha_bars, min=1e-20, max=1.0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def _extract(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        x_shape: torch.Size,
    ) -> torch.Tensor:
        """
        Extract coefficients indexed by batch timesteps and reshape
        for broadcasting over input tensor.

        Args:
            a: Tensor of shape (T,).
            t: Tensor of shape (B,) containing timestep indices.
            x_shape: Shape of target tensor.

        Returns:
            Tensor reshaped to broadcast over x.
        """
        out = a.gather(0, t.long())
        return out.reshape((t.shape[0],) + (1,) * (len(x_shape) - 1))

    @staticmethod
    def _finite(x: torch.Tensor, clamp: float | None = None) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if clamp is not None:
            x = x.clamp(-clamp, clamp)
        return x

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion step.

        Args:
            x0: Clean input sample (B, ...).
            t: Timestep indices (B,).
            noise: Optional noise tensor (B, ...).

        Returns:
            x_t: Noised sample.
            noise: Noise used.
        """
        if noise is None:
            noise = torch.randn_like(x0)
        coef1 = self._extract(self.sqrt_alpha_bars, t, x0.shape).to(dtype=x0.dtype, device=x0.device)
        coef2 = self._extract(self.sqrt_one_minus_alpha_bars, t, x0.shape).to(dtype=x0.dtype, device=x0.device)
        x_t = coef1 * x0 + coef2 * noise
        return x_t, noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        coef1 = self._extract(self.sqrt_alpha_bars, t, x_t.shape).to(dtype=x_t.dtype, device=x_t.device)
        coef2 = self._extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape).to(dtype=x_t.dtype, device=x_t.device)
        # Small clamp to avoid exact division by zero while preserving scale.
        coef1 = torch.clamp(coef1, min=1e-20)
        x0_hat = (x_t - coef2 * eps_pred) / coef1
        return self._finite(x0_hat)

    def _respaced_timesteps(self, sampling_steps: int) -> list[int]:
        if sampling_steps <= 0:
            raise ValueError(f"`sampling_steps` must be > 0, got {sampling_steps}")
        sampling_steps = min(int(sampling_steps), self.T)
        ts = torch.linspace(self.T - 1, 0, sampling_steps, device=self.device)
        ts = torch.round(ts).long().tolist()
        # remove accidental duplicates while preserving order (descending)
        dedup = []
        seen = set()
        for t in ts:
            if t not in seen:
                dedup.append(t)
                seen.add(t)
        return dedup

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: int,
        eta: float = 1.0,
        clip_denoised: bool = True,
        clip_range: float = 6.0,
    ) -> torch.Tensor:
        """
        One reverse diffusion step.

        Args:
            model: Noise prediction model.
            x_t: Current noisy sample (B, ...).
            t: Current timestep indices (B,).

        Returns:
            x_{t-1}: Denoised sample.
        """
        eps_pred = model(x_t.to(torch.float32), t).to(x_t.dtype)
        x0_hat = self.predict_x0_from_eps(x_t=x_t, t=t, eps_pred=eps_pred)
        if clip_denoised:
            x0_hat = x0_hat.clamp(-clip_range, clip_range)

        if t_prev < 0:
            return x0_hat

        a_t = self._extract(self.alpha_bars, t, x_t.shape)
        a_prev = torch.full_like(a_t, self.alpha_bars[t_prev].item())

        # DDIM-style update (supports respacing and optional stochasticity via eta).
        ratio1 = (1 - a_prev) / torch.clamp(1 - a_t, min=1e-20)
        ratio2 = 1 - (a_t / torch.clamp(a_prev, min=1e-20))
        sigma_term = self._finite(ratio1 * ratio2)
        sigma = eta * torch.sqrt(torch.clamp(sigma_term, min=0.0))
        noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)

        mean_pred = (
            torch.sqrt(a_prev) * x0_hat
            + torch.sqrt(torch.clamp(1 - a_prev - sigma**2, min=0.0)) * eps_pred
        )
        x_prev = mean_pred + sigma * noise
        return self._finite(x_prev, clamp=max(10.0, clip_range * 4))

    @torch.no_grad()
    def sample(
        self,
        model,
        n: int,
        shape: int | tuple,
        device: str | torch.device | None = None,
        sampling_steps: int | None = None,
        eta: float = 1.0,
        clip_denoised: bool = True,
        clip_range: float = 6.0,
        return_x0_trajectory: bool = False,
        trajectory_steps: int = 20,
    ) -> torch.Tensor:
        """
        Full reverse sampling loop.

        Args:
            model: Noise prediction model.
            n: Number of samples.
            shape: Sample shape excluding batch.
            device: Optional override device.

        Returns:
            Generated samples.
        """
        device = self.device if device is None else torch.device(device)
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        # Keep latent trajectory in higher precision when supported by device.
        x_t = torch.randn((n, *shape), device=device, dtype=self.math_dtype)
        model = model.to(device)
        model.eval()

        num_steps = self.T if sampling_steps is None else int(sampling_steps)
        timesteps = self._respaced_timesteps(num_steps)

        traj = []
        keep_from = max(0, len(timesteps) - int(trajectory_steps))

        for idx, t_int in enumerate(timesteps):
            t = torch.full((n,), t_int, device=device, dtype=torch.long)
            eps_pred = model(x_t.to(torch.float32), t).to(self.math_dtype)
            x0_hat = self.predict_x0_from_eps(x_t=x_t, t=t, eps_pred=eps_pred)
            if clip_denoised:
                x0_hat = x0_hat.clamp(-clip_range, clip_range)

            if return_x0_trajectory and idx >= keep_from:
                traj.append(x0_hat.detach().cpu().to(torch.float32))

            t_prev = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
            if t_prev < 0:
                x_t = x0_hat
            else:
                a_t = self._extract(self.alpha_bars, t, x_t.shape)
                a_prev = torch.full_like(a_t, self.alpha_bars[t_prev].item())
                ratio1 = (1 - a_prev) / torch.clamp(1 - a_t, min=1e-20)
                ratio2 = 1 - (a_t / torch.clamp(a_prev, min=1e-20))
                sigma_term = self._finite(ratio1 * ratio2)
                sigma = eta * torch.sqrt(torch.clamp(sigma_term, min=0.0))
                noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
                x_t = (
                    torch.sqrt(a_prev) * x0_hat
                    + torch.sqrt(torch.clamp(1 - a_prev - sigma**2, min=0.0)) * eps_pred
                    + sigma * noise
                )
                x_t = self._finite(x_t, clamp=max(10.0, clip_range * 4))

        if return_x0_trajectory:
            if len(traj) == 0:
                traj = torch.empty((0, n, *shape), dtype=torch.float32)
            else:
                traj = torch.stack(traj, dim=0)
            return x_t.to(torch.float32), traj
        return x_t.to(torch.float32)