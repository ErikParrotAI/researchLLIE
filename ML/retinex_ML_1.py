# -*- coding: utf-8 -*-
"""
Retinexformer Training (PyTorch)
===============================

Навчальний скрипт для моделі покращення освітлення зображень **Retinexformer** на датасеті
LOL (папка *our485*). Скрипт містить:

* патч‑ембединг → пам’яті вистачає навіть на CPU;
* гнучкі CLI‑параметри (розмір патча, глибина, дим, тощо);
* автоматичний **train/val спліт** або окремий `--val_root`;
* збереження чекпойнтів — періодичних та **найкращого** (`retinexformer_best.pth`)
  за мінімальною val‑втратою;
* підтримка відновлення тренування (`--resume`);
* працює на CUDA, якщо доступно.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from torchvision.transforms import functional as TF
from PIL import Image

try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class LOLPairDataset(Dataset):
    """LOL датасет із підпапками *low* / *high* — будує пари за імʼям файлу."""

    def __init__(self, root: str | Path, crop: int = 256, patch: int = 8):
        self.root = Path(root)
        self.crop = crop
        self.patch = patch
        if self.crop % self.patch != 0:
            raise ValueError("crop має ділитися на patch ({} vs {})".format(self.crop, self.patch))
        low_dir, high_dir = self.root / "low", self.root / "high"
        if not low_dir.exists() or not high_dir.exists():
            raise RuntimeError("У {} мають бути папки 'low' та 'high'".format(self.root))

        self.pairs: List[Tuple[Path, Path]] = []
        for lp in low_dir.rglob("*.*"):
            if lp.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            rel = lp.relative_to(low_dir)
            hp = high_dir / rel
            if hp.exists():
                self.pairs.append((lp, hp))
        if not self.pairs:
            raise RuntimeError("У {} не знайдено пар low/high".format(self.root))
        print(f"[Dataset] {len(self.pairs)} пар у {self.root}")
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    # --- helpers ------------------------------------------------------------
    def _rand_crop(self, a: Image.Image, b: Image.Image):
        if a.width < self.crop or a.height < self.crop:
            min_side = min(a.width, a.height)
            a = TF.center_crop(a, min_side)
            b = TF.center_crop(b, min_side)
        else:
            i, j, h, w = transforms.RandomCrop.get_params(a, (self.crop, self.crop))
            a = TF.crop(a, i, j, h, w)
            b = TF.crop(b, i, j, h, w)
        return a, b

    def __getitem__(self, idx: int):
        lp, hp = self.pairs[idx]
        low, high = Image.open(lp).convert("RGB"), Image.open(hp).convert("RGB")
        low, high = self._rand_crop(low, high)
        if random.random() < 0.5:
            low, high = TF.hflip(low), TF.hflip(high)
        return self.to_tensor(low), self.to_tensor(high)

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class Retinexformer(nn.Module):
    def __init__(self, dim: int = 64, depth: int = 4, heads: int = 8, patch: int = 8):
        super().__init__()
        self.patch = patch
        self.enc_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        enc_layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, depth)
        dec_layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, depth)
        self.proj = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x: torch.Tensor):  # (B,3,H,W)
        B, _, H, W = x.shape
        f = self.enc_embed(x)            # (B,C,H',W')
        H_, W_ = f.shape[-2:]
        t = f.flatten(2).transpose(1, 2)  # (B,N,C)
        t = self.encoder(t)
        t = self.decoder(t) + t          # skip‑connection
        f = t.transpose(1, 2).view(B, -1, H_, W_)
        f = F.interpolate(f, scale_factor=self.patch, mode="bilinear", align_corners=False)
        out = torch.sigmoid(self.proj(f))
        return out

# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------

def loss_fn(pred, target, l1_w: float = 0.8):
    l1 = F.l1_loss(pred, target)
    if HAS_SSIM:
        ssim = 1 - ssim_fn(pred, target)
    else:
        ssim = 0.0
    return l1_w * l1 + (1 - l1_w) * ssim

# ----------------------------------------------------------------------------
# Train / Val helpers
# ----------------------------------------------------------------------------

def evaluate(model, loader, device):
    model.eval()
    acc_loss = 0.0
    with torch.no_grad():
        for low, high in loader:
            low, high = low.to(device), high.to(device)
            acc_loss += loss_fn(model(low), high).item()
    return acc_loss / max(1, len(loader))

# ----------------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------------

def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Пристрій: {device}")

    full_ds = LOLPairDataset(opt.dataset_root, opt.crop, opt.patch)
    if opt.val_root:
        val_ds = LOLPairDataset(opt.val_root, opt.crop, opt.patch)
        train_ds = full_ds
    else:
        val_len = int(len(full_ds) * opt.val_split)
        train_len = len(full_ds) - val_len
        train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=opt.batch_size,
            shuffle=shuffle,
            num_workers=0 if os.name == "nt" else 4,
            pin_memory=torch.cuda.is_available(),
        )

    train_ld = make_loader(train_ds, True)
    val_ld = make_loader(val_ds, False)

    model = Retinexformer(opt.dim, opt.depth, opt.heads, opt.patch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)

    start_epoch = 1
    best_val = float("inf")
    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)

    # resume support
    if opt.resume:
        ckpt = torch.load(opt.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", best_val)
        print(f"[Resume] с {start_epoch}-ї епохи | best_val={best_val:.4f}")

    for epoch in range(start_epoch, opt.epochs + 1):
        model.train()
        run = 0.0
        for i, (low, high) in enumerate(train_ld, 1):
            low, high = low.to(device), high.to(device)
            loss = loss_fn(model(low), high)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            run += loss.item()
            if i % 10 == 0:
                print(f"Ep {epoch:3d}/{opt.epochs} | {i:4d}/{len(train_ld)} | loss {run/10:.4f}")
                run = 0.0

        val_loss = evaluate(model, val_ld, device)
        print(f"[Epoch {epoch}] val_loss = {val_loss:.4f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "patch": opt.patch,
            }, Path(opt.save_dir) / "retinexformer_best.pth")
            print("   🔥 New best saved")

        # periodic checkpoint
        if epoch % opt.save_freq == 0 or epoch == opt.epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
                "patch": opt.patch,
            }, Path(opt.save_dir) / f"retinexformer_epoch_{epoch:03d}.pth")

    print(f"[Done] best val_loss = {best_val:.4f}")

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def cli():
    p = argparse.ArgumentParser("Retinexformer training script")
    p.add_argument("--dataset_root", default="../data/lol_dataset/our485")
    p.add_argument("--val_root", default=None, help="окремий валідаційний каталог")
    p.add_argument("--val_split", type=float, default=0.1, help="частка даних для валідації")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--patch", type=int, default=8)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--save_freq", type=int, default=5)
    p.add_argument("--resume", default=None, help="шлях до checkpoint для відновлення")
    return p.parse_args()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    args = cli()
    train(args)
