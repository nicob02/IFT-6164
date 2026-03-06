import argparse
import json
import math
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Allow "python experiments/ddpm_gaussian_dataset.py" from repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ddpm import DDPM  # noqa: E402
from models.mlp import NN  # noqa: E402
from utils import set_seed  # noqa: E402


@dataclass
class Gaussian1DConfig:
    means: tuple[float, ...] = (1.0, 2.0, 3.0)
    std: float = 0.05


@dataclass
class Gaussian2DConfig:
    grid_size: int = 5
    grid_min: float = -2.0
    grid_max: float = 2.0
    std: float = 0.05


def pick_device(user_device: str) -> torch.device:
    if user_device != "auto":
        return torch.device(user_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_1d_mixture(n: int, cfg: Gaussian1DConfig) -> np.ndarray:
    means = np.asarray(cfg.means, dtype=np.float32)
    labels = np.random.randint(0, len(means), size=n)
    x = np.random.randn(n).astype(np.float32) * cfg.std + means[labels]
    return x[:, None]


def sample_2d_grid(n: int, cfg: Gaussian2DConfig) -> np.ndarray:
    axis = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_size, dtype=np.float32)
    modes = np.array([(x, y) for x in axis for y in axis], dtype=np.float32)
    labels = np.random.randint(0, len(modes), size=n)
    x = np.random.randn(n, 2).astype(np.float32) * cfg.std + modes[labels]
    return x


def get_2d_modes(cfg: Gaussian2DConfig) -> np.ndarray:
    axis = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_size, dtype=np.float32)
    return np.array([(x, y) for x in axis for y in axis], dtype=np.float32)


def train_ddpm(
    data_np: np.ndarray,
    dim: int,
    ddpm: DDPM,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_features: int,
    t_features: int,
) -> NN:
    ds = TensorDataset(torch.from_numpy(data_np), torch.zeros(len(data_np)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    model = NN(
        in_features=dim,
        hidden_features=hidden_features,
        t_features=t_features,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for ep in range(epochs):
        model.train()
        running = 0.0
        for x0, _ in loader:
            x0 = x0.to(device)
            t = torch.randint(0, ddpm.T, (x0.size(0),), device=device).long()
            x_t, noise = ddpm.q_sample(x0=x0, t=t)
            noise_pred = model(x_t, t)
            loss = loss_fn(noise_pred, noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * x0.size(0)
        print(f"epoch {ep + 1}/{epochs} | loss={running / len(ds):.6f}")
    return model


def normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def mixture_pdf_1d(x: np.ndarray, cfg: Gaussian1DConfig) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    for m in cfg.means:
        y += normal_pdf(x, m, cfg.std) / len(cfg.means)
    return y


def classify_1d_support(x: np.ndarray, cfg: Gaussian1DConfig, sigma_thresh: float = 6.0) -> np.ndarray:
    means = np.asarray(cfg.means, dtype=np.float32)[None, :]
    d = np.abs(x[:, None] - means)
    return (d <= sigma_thresh * cfg.std).any(axis=1)


def bridge_mass_1d(x: np.ndarray, cfg: Gaussian1DConfig, sigma_thresh: float = 6.0) -> float:
    means = np.array(sorted(cfg.means), dtype=np.float32)
    in_support = classify_1d_support(x, cfg, sigma_thresh=sigma_thresh)
    between_global = (x >= (means[0] - sigma_thresh * cfg.std)) & (x <= (means[-1] + sigma_thresh * cfg.std))
    bridge = (~in_support) & between_global
    return float(bridge.mean())


def classify_2d_support(x: np.ndarray, modes: np.ndarray, std: float, radius_sigma: float = 3.0) -> np.ndarray:
    diff = x[:, None, :] - modes[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    return (dist <= radius_sigma * std).any(axis=1)


def bridge_mass_2d(x: np.ndarray, modes: np.ndarray, std: float, radius_sigma: float = 3.0) -> float:
    in_support = classify_2d_support(x, modes, std, radius_sigma=radius_sigma)
    mins = modes.min(axis=0) - 1.0
    maxs = modes.max(axis=0) + 1.0
    in_bbox = (x[:, 0] >= mins[0]) & (x[:, 0] <= maxs[0]) & (x[:, 1] >= mins[1]) & (x[:, 1] <= maxs[1])
    bridge = (~in_support) & in_bbox
    return float(bridge.mean())


def hal_metric_from_traj(x0_traj: torch.Tensor) -> np.ndarray:
    # x0_traj: (K, N, D)
    # Hal(x) = mean_t ||x0_t - mean_t(x0)||^2
    if x0_traj.ndim != 3:
        raise ValueError(f"Expected trajectory shape (K,N,D), got {tuple(x0_traj.shape)}")
    arr = x0_traj.numpy()
    mu = arr.mean(axis=0, keepdims=True)
    sq = ((arr - mu) ** 2).mean(axis=(0, 2))
    return sq


@torch.no_grad()
def sample_model_in_batches(
    ddpm: DDPM,
    model: NN,
    total_n: int,
    shape: int | tuple[int, ...],
    device: torch.device,
    sampling_steps: int,
    eta: float,
    clip_denoised: bool,
    clip_range: float,
    max_batch: int,
    return_x0_trajectory: bool = False,
    trajectory_steps: int = 20,
) -> tuple[np.ndarray, torch.Tensor | None]:
    chunks = []
    traj_chunks = []
    produced = 0
    while produced < total_n:
        n = min(max_batch, total_n - produced)
        out = ddpm.sample(
            model=model,
            n=n,
            shape=shape,
            device=device,
            sampling_steps=sampling_steps,
            eta=eta,
            clip_denoised=clip_denoised,
            clip_range=clip_range,
            return_x0_trajectory=return_x0_trajectory,
            trajectory_steps=trajectory_steps,
        )
        if return_x0_trajectory:
            x, traj = out
            traj_chunks.append(traj)
        else:
            x = out
        chunks.append(x.detach().cpu())
        produced += n
    x_cat = torch.cat(chunks, dim=0).numpy()
    if return_x0_trajectory:
        # concat over sample axis: (K, n1, d) + (K, n2, d) -> (K, n1+n2, d)
        traj_cat = torch.cat(traj_chunks, dim=1)
        return x_cat, traj_cat
    return x_cat, None


def choose_threshold(scores: np.ndarray, labels_hallucinated: np.ndarray) -> tuple[float, float, float]:
    # Maximize balanced accuracy over candidate thresholds.
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    best = (candidates[0], -1.0, 0.0, 0.0)  # thr, bal_acc, tpr, tnr
    y = labels_hallucinated.astype(bool)
    for thr in candidates:
        pred = scores >= thr
        tp = np.logical_and(pred, y).sum()
        tn = np.logical_and(~pred, ~y).sum()
        fp = np.logical_and(pred, ~y).sum()
        fn = np.logical_and(~pred, y).sum()
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        bal = 0.5 * (tpr + tnr)
        if bal > best[1]:
            best = (float(thr), float(bal), float(tpr), float(tnr))
    return best[0], best[2], best[3]


def oracle_eps_from_xt_1d(x_t: torch.Tensor, t: torch.Tensor, ddpm: DDPM, cfg: Gaussian1DConfig) -> torch.Tensor:
    # For q_t(x): mixture over i of N(mu_i_t, var_t)
    # mu_i_t = sqrt(alpha_bar_t) * mu_i
    # var_t = alpha_bar_t * sigma_data^2 + (1 - alpha_bar_t)
    # score = d/dx log q_t(x) = sum_i w_i N_i * (-(x-mu_i_t)/var_t) / sum_i w_i N_i
    x = x_t[:, 0]
    ab = ddpm.alpha_bars.gather(0, t).float()
    means0 = torch.tensor(cfg.means, device=x.device, dtype=x.dtype)[None, :]
    mu_t = torch.sqrt(ab)[:, None] * means0
    var_t = ab[:, None] * (cfg.std**2) + (1.0 - ab)[:, None]

    x_col = x[:, None]
    comp_logp = -0.5 * ((x_col - mu_t) ** 2) / var_t - 0.5 * torch.log(2 * torch.pi * var_t)
    comp = torch.softmax(comp_logp, dim=1)
    score = (comp * (-(x_col - mu_t) / var_t)).sum(dim=1)
    eps = -torch.sqrt(1.0 - ab) * score
    return eps[:, None]


@torch.no_grad()
def sample_oracle_1d(
    ddpm: DDPM,
    n: int,
    cfg: Gaussian1DConfig,
    device: torch.device,
    sampling_steps: int = 1000,
    eta: float = 1.0,
) -> np.ndarray:
    x_t = torch.randn(n, 1, device=device)
    timesteps = ddpm._respaced_timesteps(sampling_steps)
    for idx, t_int in enumerate(timesteps):
        t = torch.full((n,), t_int, device=device, dtype=torch.long)
        eps_pred = oracle_eps_from_xt_1d(x_t=x_t, t=t, ddpm=ddpm, cfg=cfg)
        x0_hat = ddpm.predict_x0_from_eps(x_t=x_t, t=t, eps_pred=eps_pred)
        t_prev = timesteps[idx + 1] if idx + 1 < len(timesteps) else -1
        if t_prev < 0:
            x_t = x0_hat
        else:
            a_t = ddpm._extract(ddpm.alpha_bars, t, x_t.shape)
            a_prev = torch.full_like(a_t, ddpm.alpha_bars[t_prev].item())
            sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
            noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
            x_t = (
                torch.sqrt(a_prev) * x0_hat
                + torch.sqrt(torch.clamp(1.0 - a_prev - sigma**2, min=0.0)) * eps_pred
                + sigma * noise
            )
    return x_t.cpu().numpy().reshape(-1)


def parse_ablation_values(raw: str) -> list[int]:
    vals = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("`--ablation_values` must contain at least one integer.")
    return vals


def run_phase1(args) -> dict:
    device = pick_device(args.device)
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[Phase 1] device={device}")
    beta_end = args.beta_end
    if device.type == "mps" and beta_end > 0.05:
        # MPS uses float32 only; very large beta_end with T=1000 is numerically fragile.
        beta_end = 0.05
        print("[Phase 1] MPS detected: overriding beta_end to 0.05 for stability (from", args.beta_end, ").")

    cfg1d = Gaussian1DConfig(std=args.std_1d)
    cfg2d = Gaussian2DConfig(std=args.std_2d)
    modes2d = get_2d_modes(cfg2d)

    train_1d = sample_1d_mixture(args.train_size_1d, cfg1d).astype(np.float32)
    train_2d = sample_2d_grid(args.train_size_2d, cfg2d).astype(np.float32)
    ablation_values = parse_ablation_values(args.ablation_values)
    summary = {
        "phase1": {
            "ablation_target": args.ablation_target,
            "ablation_values": ablation_values,
            "train_size_1d": args.train_size_1d,
            "train_size_2d": args.train_size_2d,
            "sample_count_1d": args.sample_count,
            "sample_count_2d": args.sample_count_2d,
            "ablation": {},
        }
    }

    # IMPORTANT:
    # - sample_count is kept fixed across ablation settings.
    # - the ablated variable is chosen by --ablation_target:
    #   (a) sampling_steps: one trained model, different respacing at sampling
    #   (b) diffusion_timesteps: retrain per T and sample with full T steps
    models_1d = {}
    models_2d = {}
    ddpms_1d = {}
    ddpms_2d = {}
    ckpt_dir = os.path.join(args.outdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.ablation_target == "sampling_steps":
        ddpm_1d = DDPM(
            T=args.T,
            device=device,
            beta_start=args.beta_start,
            beta_end=beta_end,
            schedule_type=args.schedule,
        )
        ddpm_2d = DDPM(
            T=args.T,
            device=device,
            beta_start=args.beta_start,
            beta_end=beta_end,
            schedule_type=args.schedule,
        )
        model_1d = train_ddpm(
            data_np=train_1d,
            dim=1,
            ddpm=ddpm_1d,
            device=device,
            epochs=args.epochs_1d,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_features=args.hidden_features,
            t_features=args.t_features,
        )
        model_2d = train_ddpm(
            data_np=train_2d,
            dim=2,
            ddpm=ddpm_2d,
            device=device,
            epochs=args.epochs_2d,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_features=args.hidden_features,
            t_features=args.t_features,
        )
        torch.save(model_1d.state_dict(), os.path.join(ckpt_dir, f"model_1d_T{args.T}.pt"))
        torch.save(model_2d.state_dict(), os.path.join(ckpt_dir, f"model_2d_T{args.T}.pt"))
        for v in ablation_values:
            models_1d[v] = model_1d
            models_2d[v] = model_2d
            ddpms_1d[v] = ddpm_1d
            ddpms_2d[v] = ddpm_2d
    else:
        for T_val in ablation_values:
            ddpm_1d = DDPM(
                T=T_val,
                device=device,
                beta_start=args.beta_start,
                beta_end=beta_end,
                schedule_type=args.schedule,
            )
            ddpm_2d = DDPM(
                T=T_val,
                device=device,
                beta_start=args.beta_start,
                beta_end=beta_end,
                schedule_type=args.schedule,
            )
            model_1d = train_ddpm(
                data_np=train_1d,
                dim=1,
                ddpm=ddpm_1d,
                device=device,
                epochs=args.epochs_1d,
                batch_size=args.batch_size,
                lr=args.lr,
                hidden_features=args.hidden_features,
                t_features=args.t_features,
            )
            model_2d = train_ddpm(
                data_np=train_2d,
                dim=2,
                ddpm=ddpm_2d,
                device=device,
                epochs=args.epochs_2d,
                batch_size=args.batch_size,
                lr=args.lr,
                hidden_features=args.hidden_features,
                t_features=args.t_features,
            )
            torch.save(model_1d.state_dict(), os.path.join(ckpt_dir, f"model_1d_T{T_val}.pt"))
            torch.save(model_2d.state_dict(), os.path.join(ckpt_dir, f"model_2d_T{T_val}.pt"))
            models_1d[T_val] = model_1d
            models_2d[T_val] = model_2d
            ddpms_1d[T_val] = ddpm_1d
            ddpms_2d[T_val] = ddpm_2d

    # ---------- 1D ablation + Hal metric ----------
    ncols = len(ablation_values)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]
    xgrid = np.linspace(0.5, 3.5, 1000)
    true_pdf = mixture_pdf_1d(xgrid, cfg1d)
    hal_for_hist = None
    label_for_hist = None

    for ax, value in zip(axes, ablation_values):
        ddpm_1d = ddpms_1d[value]
        model_1d = models_1d[value]
        sampling_steps = value if args.ablation_target == "sampling_steps" else ddpm_1d.T
        gen, _ = sample_model_in_batches(
            ddpm=ddpm_1d,
            model=model_1d,
            total_n=args.sample_count,
            shape=1,
            device=device,
            sampling_steps=sampling_steps,
            eta=args.eta,
            clip_denoised=args.clip_denoised,
            clip_range=args.clip_range,
            max_batch=args.eval_batch_size_1d,
            return_x0_trajectory=False,
        )
        gen = gen.reshape(-1)
        finite_mask_1d = np.isfinite(gen)
        finite_fraction_1d = float(finite_mask_1d.mean())
        gen = gen[finite_mask_1d]
        if gen.size == 0:
            raise RuntimeError(
                "All generated 1D samples are non-finite. "
                "Try reducing eval batch sizes or sampling_steps."
            )
        labels_in = classify_1d_support(gen, cfg1d, sigma_thresh=args.support_sigma_1d)
        labels_hall = ~labels_in
        bridge = bridge_mass_1d(gen, cfg1d, sigma_thresh=args.support_sigma_1d)

        hal_eval_n = min(args.hal_eval_count, args.sample_count)
        hal_gen, hal_traj = sample_model_in_batches(
            ddpm=ddpm_1d,
            model=model_1d,
            total_n=hal_eval_n,
            shape=1,
            device=device,
            sampling_steps=sampling_steps,
            eta=args.eta,
            clip_denoised=args.clip_denoised,
            clip_range=args.clip_range,
            max_batch=args.hal_batch_size_1d,
            return_x0_trajectory=True,
            trajectory_steps=args.hal_window_steps,
        )
        hal_gen = hal_gen.reshape(-1)
        finite_hal = np.isfinite(hal_gen)
        hal = hal_metric_from_traj(hal_traj[:, finite_hal, :])
        labels_in_hal = classify_1d_support(hal_gen[finite_hal], cfg1d, sigma_thresh=args.support_sigma_1d)
        labels_hall_hal = ~labels_in_hal

        thr, tpr, tnr = choose_threshold(hal, labels_hall_hal)
        summary["phase1"]["ablation"][f"1d_{args.ablation_target}_{value}"] = {
            "diffusion_T": int(ddpm_1d.T),
            "sampling_steps": int(sampling_steps),
            "bridge_mass": bridge,
            "hallucination_rate": float(labels_hall.mean()),
            "finite_fraction": finite_fraction_1d,
            "generated_mean": float(np.mean(gen)),
            "generated_std": float(np.std(gen)),
            "generated_min": float(np.min(gen)),
            "generated_max": float(np.max(gen)),
            "hal_eval_count": int(hal_eval_n),
            "hal_threshold": thr,
            "hal_tpr": tpr,
            "hal_tnr": tnr,
        }

        ax.hist(gen, bins=250, density=True, alpha=0.6, color="#f4a259", label="Generated")
        ax.plot(xgrid, true_pdf, color="#1f77b4", lw=2.0, label="True")
        ax.set_yscale("log")
        ax.set_title(
            f"1D {args.ablation_target}={value}\n"
            f"T={ddpm_1d.T}, sampling={sampling_steps}, bridge={bridge:.4f}"
        )
        ax.set_xlim(0.6, 3.4)
        ax.grid(alpha=0.25)
        if value == ablation_values[min(1, len(ablation_values) - 1)]:
            hal_for_hist = hal
            label_for_hist = labels_hall_hal

    axes[0].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "phase1_1d_ablation_density.png"), dpi=170)
    plt.close(fig)

    if hal_for_hist is not None:
        plt.figure(figsize=(7, 4))
        plt.hist(
            hal_for_hist[~label_for_hist],
            bins=80,
            alpha=0.65,
            density=True,
            label="In-support",
            color="#4c78a8",
        )
        plt.hist(
            hal_for_hist[label_for_hist],
            bins=80,
            alpha=0.65,
            density=True,
            label="Hallucinated",
            color="#f58518",
        )
        plt.title(f"Hal(x) histogram (1D, {args.ablation_target} ablation)")
        plt.xlabel("Hal(x)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "phase1_1d_hal_hist.png"), dpi=170)
        plt.close()

    # ---------- 2D ablation ----------
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharex=True, sharey=True)
    if ncols == 1:
        axes = [axes]
    for ax, value in zip(axes, ablation_values):
        ddpm_2d = ddpms_2d[value]
        model_2d = models_2d[value]
        sampling_steps = value if args.ablation_target == "sampling_steps" else ddpm_2d.T
        gen, _ = sample_model_in_batches(
            ddpm=ddpm_2d,
            model=model_2d,
            total_n=args.sample_count_2d,
            shape=2,
            device=device,
            sampling_steps=sampling_steps,
            eta=args.eta,
            clip_denoised=args.clip_denoised,
            clip_range=args.clip_range,
            max_batch=args.eval_batch_size_2d,
            return_x0_trajectory=False,
        )
        finite_mask_2d = np.isfinite(gen).all(axis=1)
        finite_fraction_2d = float(finite_mask_2d.mean())
        gen = gen[finite_mask_2d]
        if gen.shape[0] == 0:
            raise RuntimeError(
                "All generated 2D samples are non-finite. "
                "Try reducing eval batch sizes or sampling_steps."
            )
        bridge = bridge_mass_2d(gen, modes2d, std=cfg2d.std, radius_sigma=args.support_sigma_2d)
        hall = ~classify_2d_support(gen, modes2d, std=cfg2d.std, radius_sigma=args.support_sigma_2d)
        summary["phase1"]["ablation"][f"2d_{args.ablation_target}_{value}"] = {
            "diffusion_T": int(ddpm_2d.T),
            "sampling_steps": int(sampling_steps),
            "bridge_mass": bridge,
            "hallucination_rate": float(hall.mean()),
            "finite_fraction": finite_fraction_2d,
            "generated_mean_norm": float(np.linalg.norm(np.mean(gen, axis=0))),
            "generated_std_norm": float(np.linalg.norm(np.std(gen, axis=0))),
        }

        show = gen[np.random.choice(len(gen), size=min(25000, len(gen)), replace=False)]
        ax.scatter(show[:, 0], show[:, 1], s=1.0, alpha=0.35, color="#f4a259", label="Generated")
        ax.scatter(modes2d[:, 0], modes2d[:, 1], s=20, color="#1f77b4", label="Mode centers")
        ax.set_title(
            f"2D {args.ablation_target}={value}\n"
            f"T={ddpm_2d.T}, sampling={sampling_steps}, bridge={bridge:.4f}"
        )
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "phase1_2d_ablation_scatter.png"), dpi=170)
    plt.close(fig)

    return summary


def run_phase2(args) -> dict:
    device = pick_device(args.device)
    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[Phase 2] device={device}")

    cfg1d = Gaussian1DConfig(std=args.std_1d)
    train_1d = sample_1d_mixture(args.train_size_1d, cfg1d).astype(np.float32)
    ddpm_1d = DDPM(
        T=args.T,
        device=device,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule_type=args.schedule,
    )

    model_1d = train_ddpm(
        data_np=train_1d,
        dim=1,
        ddpm=ddpm_1d,
        device=device,
        epochs=args.epochs_1d,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_features=args.hidden_features,
        t_features=args.t_features,
    )

    nn_samples = ddpm_1d.sample(
        model=model_1d,
        n=args.sample_count,
        shape=1,
        device=device,
        sampling_steps=ddpm_1d.T,
        eta=args.eta,
        clip_denoised=args.clip_denoised,
        clip_range=args.clip_range,
    )
    nn_samples = nn_samples.detach().cpu().numpy().reshape(-1)
    oracle_samples = sample_oracle_1d(
        ddpm=ddpm_1d,
        n=args.sample_count,
        cfg=cfg1d,
        device=device,
        sampling_steps=ddpm_1d.T,
        eta=args.eta,
    )

    bridge_nn = bridge_mass_1d(nn_samples, cfg1d, sigma_thresh=args.support_sigma_1d)
    bridge_oracle = bridge_mass_1d(oracle_samples, cfg1d, sigma_thresh=args.support_sigma_1d)

    xgrid = np.linspace(0.5, 3.5, 1000)
    true_pdf = mixture_pdf_1d(xgrid, cfg1d)
    plt.figure(figsize=(8, 4.2))
    plt.hist(nn_samples, bins=250, density=True, alpha=0.55, color="#f58518", label=f"NN score (bridge={bridge_nn:.4f})")
    plt.hist(
        oracle_samples,
        bins=250,
        density=True,
        alpha=0.45,
        color="#54a24b",
        label=f"Oracle score (bridge={bridge_oracle:.4f})",
    )
    plt.plot(xgrid, true_pdf, lw=2.0, color="#1f77b4", label="True density")
    plt.yscale("log")
    plt.xlim(0.6, 3.4)
    plt.grid(alpha=0.2)
    plt.title("Phase 2: 1D Oracle-score baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "phase2_1d_oracle_vs_nn.png"), dpi=170)
    plt.close()

    return {
        "phase2": {
            "nn_bridge_mass": bridge_nn,
            "oracle_bridge_mass": bridge_oracle,
            "bridge_reduction_percent": 100.0 * (bridge_nn - bridge_oracle) / max(bridge_nn, 1e-8),
        }
    }


def parse_args():
    p = argparse.ArgumentParser("Toy DDPM mode-interpolation reproduction (Phase 1 + Phase 2)")
    p.add_argument("--phase", type=str, default="both", choices=["phase1", "phase2", "both"])
    p.add_argument("--outdir", type=str, default=os.path.join(ROOT, "results", "toy_mode_interp"))
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--T", type=int, default=1000)
    p.add_argument(
        "--ablation_target",
        type=str,
        default="sampling_steps",
        choices=["sampling_steps", "diffusion_timesteps"],
        help=(
            "sampling_steps: fixed trained model, vary respacing steps; "
            "diffusion_timesteps: retrain for each T in --ablation_values."
        ),
    )
    p.add_argument(
        "--ablation_values",
        type=str,
        default="1000,250,50",
        help="Comma-separated integers for the selected ablation target.",
    )
    p.add_argument("--schedule", type=str, default="linear", choices=["linear", "cosine"])
    # Paper-like defaults for Gaussian toy experiments.
    p.add_argument("--beta_start", type=float, default=1e-3)
    p.add_argument("--beta_end", type=float, default=2e-1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--hidden_features", type=int, default=128)
    p.add_argument("--t_features", type=int, default=128)

    p.add_argument("--epochs_1d", type=int, default=800)
    p.add_argument("--epochs_2d", type=int, default=1000)
    p.add_argument("--train_size_1d", type=int, default=50000)
    p.add_argument("--train_size_2d", type=int, default=100000)
    p.add_argument("--std_1d", type=float, default=0.05)
    p.add_argument("--std_2d", type=float, default=0.05)

    # Evaluation sample counts for high-statistics runs.
    p.add_argument("--sample_count", type=int, default=10000000)
    p.add_argument("--sample_count_2d", type=int, default=5000000)
    p.add_argument("--hal_window_steps", type=int, default=20)
    p.add_argument("--support_sigma_1d", type=float, default=6.0)
    p.add_argument("--support_sigma_2d", type=float, default=3.0)
    p.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Sampling stochasticity. 1.0 ~= DDPM ancestral-style, 0.0 = deterministic DDIM-style.",
    )
    p.add_argument(
        "--clip_denoised",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clamp predicted x0 during sampling for numerical stability.",
    )
    p.add_argument(
        "--clip_range",
        type=float,
        default=6.0,
        help="Absolute clamp range used when --clip_denoised is enabled.",
    )
    p.add_argument("--eval_batch_size_1d", type=int, default=100000)
    p.add_argument("--eval_batch_size_2d", type=int, default=100000)
    p.add_argument("--hal_eval_count", type=int, default=200000)
    p.add_argument("--hal_batch_size_1d", type=int, default=50000)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    summary = {}
    if args.phase in ("phase1", "both"):
        summary.update(run_phase1(args))
    if args.phase in ("phase2", "both"):
        summary.update(run_phase2(args))

    summary_path = os.path.join(args.outdir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
