import argparse
import json
import os
import random
import time
import itertools

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.mlp import NN
from models.unet import UNet
from ddpm import DDPM
from utils import GaussianDataset, set_seed


SEED = 0

def parse_args():
    p = argparse.ArgumentParser(description="DDPM training script.")

    p.add_argument("--dataset", type=str, default="gaussian1d", choices=["gaussian1d", "gaussian2d", "shapes"])
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--T", type=int, default=1000)

    p.add_argument("--hidden_features", type=int, default=128)
    p.add_argument("--t_features", type=int, default=128)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--print_every", type=int, default=80)

    p.add_argument("--save_model", action="store_true")
    p.add_argument("--ckpt_path", type=str, default="checkpoints/ddpm_gaussian.pt")

    p.add_argument("--log_results", action="store_true")
    p.add_argument("--logdir", type=str, default=None)

    return p.parse_args()

@torch.no_grad()
def evaluate_ddpm(model, diffusion, dataloader, loss_fn, device, T: int):
    model.eval()
    start = time.time()
    total_loss = 0.0
    n_batches = 0

    for x0, _ in dataloader:
        x0 = x0.to(device)
        B = x0.size(0)
        t = torch.randint(0, T, (B,), device=device).long()

        x_t, noise = diffusion.q_sample(x0, t)
        noise_pred = model(x_t, t)
        loss = loss_fn(noise_pred, noise)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), time.time() - start


def train_one_epoch(args, model, diffusion, dataloader, loss_fn, optimizer, device):
    model.train()
    start = time.time()
    total_loss = 0.0
    n_batches = 0

    for idx, (x0, _) in enumerate(dataloader):
        x0 = x0.to(device)
        B = x0.size(0)

        t = torch.randint(0, args.T, (B,), device=device).long()

        x_t, noise = diffusion.q_sample(x0, t)
        noise_pred = model(x_t, t)
        loss = loss_fn(noise_pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if (idx + 1) % args.print_every == 0:
            tqdm.write(f"[TRAIN] iter {idx+1}/{len(dataloader)} | loss {loss.item():.5f}")

    return total_loss / max(n_batches, 1), time.time() - start




def main():
    set_seed(SEED)
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Gaussian dataset sizing rule:
    # train size = epochs * batch_size
    # valid size = 10% of train
    dim = 1 if args.dataset == "gaussian1d" else 2
    n_train = args.epochs * args.batch_size
    n_valid = int(0.1 * n_train)

    print(f"Dataset: {args.dataset} | dim={dim}")
    print(f"Train points: {n_train} (= epochs * batch_size)")
    print(f"Valid points: {n_valid} (= 10% of train)")

    train_ds = GaussianDataset(n=n_train, dim=dim, preset=True)
    valid_ds = GaussianDataset(n=n_valid, dim=dim, preset=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
    )

    diffusion = DDPM(T=args.T, device=device)

    in_features = dim
    model = NN(in_features=in_features, hidden_features=args.hidden_features, t_features=args.t_features).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    loss_fn = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = {
        "args": vars(args),
        "n_train": n_train,
        "n_valid": n_valid,
        "train_loss": [],
        "valid_loss": [],
        "train_time": [],
        "valid_time": [],
    }

    for epoch in range(args.epochs):
        tqdm.write(f"====== Epoch {epoch+1}/{args.epochs} ======")

        train_loss, train_wall = train_one_epoch(args, model, diffusion, train_loader, loss_fn, optimizer, device)
        valid_loss, valid_wall = evaluate_ddpm(model, diffusion, valid_loader, loss_fn, device, args.T)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_time"].append(train_wall)
        history["valid_time"].append(valid_wall)

        print(f"epoch {epoch+1}/{args.epochs} | train {train_loss:.6f} | valid {valid_loss:.6f}")

    if args.save_model:
        os.makedirs(os.path.dirname(args.ckpt_path) or ".", exist_ok=True)
        ckpt = {
            "model_state_dict": model.state_dict(),
            "cfg": {
                "dataset": args.dataset,
                "dim": dim,
                "in_features": in_features,
                "hidden_features": args.hidden_features,
                "t_features": args.t_features,
                "T": args.T,
            },
            "seed": SEED,
        }
        torch.save(ckpt, args.ckpt_path)
        print(f"Saved checkpoint to: {args.ckpt_path}")

    if args.log_results:
        if args.logdir is None:
            raise ValueError("--log_results requires --logdir")
        os.makedirs(args.logdir, exist_ok=True)
        outpath = os.path.join(args.logdir, "history.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"Saved logs to: {outpath}")


if __name__ == "__main__":
    main()