# -*- coding: utf-8 -*-
"""
Retinexformer Training (PyTorch)
===============================

**Проблема памʼяті вирішена ➡️ тепер використовується патч‑ембединг**

У попередній версії повний self‑attention працював на послідовності `H×W`‑токенів (для кропа
256×256 це 65 536 токенів → матриця уваги ~ 1 ТБ). Нова реалізація виконує self‑attention
на рівні **патчів** (за замовчуванням 8 × 8), тому довжина послідовності → `(H/8)×(W/8)`
(для 256 × 256 це лише 1 024 токени) — памʼяті достатньо навіть на CPU.

Головні зміни
-------------
* **Retinexformer**:
  * `patch_size` (def = 8). Патч‑ембединг — `Conv2d(stride=patch)`.
  * Після трансформера — reshape → upsample `F.interpolate(scale_factor=patch)`.
* **Dataset**: кроп розміром, кратним `patch_size` (за замовчуванням 256, що ділиться на 8).
* **CLI** – додано `--patch`.

Запуск прикладом:
```bash
python retinexformer_training.py \
  --dataset_root "D:/Python/researchLLIE/data/lol_dataset/our485" \
  --epochs 100 --batch_size 8 --patch 8
```

---
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image

try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
except ImportError:
    ssim_fn = None  # якщо torchmetrics не встановлено – буде використано лише L1

# ---------------------------
# Параметри за замовчуванням
# ---------------------------
DEF_DS_ROOT = "../data/lol_dataset/our485"
DEF_EPOCHS = 100
DEF_BS = 8
DEF_CROP = 256
DEF_LR = 1e-4
DEF_PATCH = 8  # новий параметр

# ---------------------------
# Датасет LOL
# ---------------------------
class LOLPairDataset(Dataset):
    """Dataset LOL із класичною структурою low/ та high/."""

    def __init__(self, root: str | Path, crop_size: int = 256, patch_size: int = 8):
        self.root = Path(root)
        self.crop_size = crop_size
        self.patch_size = patch_size
        if self.crop_size % self.patch_size != 0:
            raise ValueError(
                f"crop_size ({self.crop_size}) має ділитися на patch_size ({self.patch_size})"
            )

        if not self.root.exists():
            raise FileNotFoundError(
                f"Каталог {self.root} не існує – перевірте шлях до датасету."
            )

        low_dir = self.root / "low"
        high_dir = self.root / "high"
        if not low_dir.exists() or not high_dir.exists():
            raise RuntimeError(
                "Очікується, що всередині root будуть підпапки 'low' та 'high'. "
                "Перевірте структуру датасету."
            )

        # Пошук спільних файлів за іменем та розширенням
        self.pairs: List[Tuple[Path, Path]] = []
        for low_path in low_dir.rglob("*.*"):
            if low_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            rel_name = low_path.relative_to(low_dir)
            high_path = high_dir / rel_name
            if high_path.exists():
                self.pairs.append((low_path, high_path))

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"У каталозі {self.root} не знайдено збігів між 'low' та 'high'. "
                "Переконайтеся, що імена файлів у обох папках збігаються."
            )
        print(f"[Dataset] Знайдено {len(self.pairs)} пар low/high у {self.root}")

        # Трансформації
        self.to_tensor = transforms.ToTensor()

    # ----------------------------------------
    def __len__(self):  # type: ignore[override]
        return len(self.pairs)

    def random_crop(self, low: Image.Image, high: Image.Image):
        """RandomCrop, безпечний, гарантовано кратний patch_size."""
        if low.height < self.crop_size or low.width < self.crop_size:
            # fallback – центр‑кроп мін(Н,W)
            min_side = min(low.height, low.width)
            low = TF.center_crop(low, min_side)
            high = TF.center_crop(high, min_side)
        else:
            i, j, h, w = transforms.RandomCrop.get_params(low, (self.crop_size, self.crop_size))
            low = TF.crop(low, i, j, h, w)
            high = TF.crop(high, i, j, h, w)
        return low, high

    def __getitem__(self, idx: int):  # type: ignore[override]
        low_path, high_path = self.pairs[idx]
        img_low = Image.open(low_path).convert("RGB")
        img_high = Image.open(high_path).convert("RGB")

        # Аугментації
        img_low, img_high = self.random_crop(img_low, img_high)
        if torch.rand(1) < 0.5:
            img_low = TF.hflip(img_low)
            img_high = TF.hflip(img_high)

        # -> Tensor, [0,1]
        img_low = self.to_tensor(img_low)
        img_high = self.to_tensor(img_high)
        return img_low, img_high

# ---------------------------
# Модель Retinexformer (патч‑версія)
# ---------------------------
class Retinexformer(nn.Module):
    def __init__(self, dim: int = 64, depth: int = 4, heads: int = 8, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        # Патч‑ембединг (stride = patch)
        self.enc_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        encoder_layer = nn.TransformerEncoderLayer(
            dim, nhead=heads, dim_feedforward=dim * 4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        decoder_layer = nn.TransformerEncoderLayer(
            dim, nhead=heads, dim_feedforward=dim * 4, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)

        self.dec_proj = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def forward(self, x):  # x: (B,3,H,W)
        B, _, H, W = x.shape
        # --- Енкодер ---
        feat = self.enc_embed(x)  # (B, C, H', W')
        H_, W_ = feat.shape[-2:]
        tokens = feat.flatten(2).permute(0, 2, 1)  # (B, N, C) , N = H'×W'
        tokens = self.encoder(tokens)

        # --- Декодер + skip ---
        tokens = self.decoder(tokens) + tokens
        feat_dec = tokens.permute(0, 2, 1).view(B, -1, H_, W_)  # (B, C, H', W')

        # --- Upsample back to full res ---
        feat_up = F.interpolate(
            feat_dec, scale_factor=self.patch_size, mode="bilinear", align_corners=False
        )
        out = torch.sigmoid(self.dec_proj(feat_up))  # [0,1]
        return out

# ---------------------------
# Втрата
# ---------------------------

def loss_fn(pred, target, l1_weight: float = 0.8):
    l1 = F.l1_loss(pred, target)
    if ssim_fn is not None:
        ssim = 1 - ssim_fn(pred, target)
    else:
        ssim = 0.0
    return l1_weight * l1 + (1 - l1_weight) * ssim

# ---------------------------
# Цикл навчання
# ---------------------------

def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Пристрій: {device}")

    dataset = LOLPairDataset(opt.dataset_root, crop_size=opt.crop, patch_size=opt.patch)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4 if os.name != "nt" else 0,
        pin_memory=torch.cuda.is_available(),
    )

    model = Retinexformer(patch_size=opt.patch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)

    for epoch in range(1, opt.epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (low, high) in enumerate(dataloader, 1):
            low, high = low.to(device, non_blocking=True), high.to(device, non_blocking=True)

            pred = model(low)
            loss = loss_fn(pred, high)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                avg = running_loss / 10
                print(
                    f"Epoch [{epoch}/{opt.epochs}] | Step {i:4d}/{len(dataloader)} | Loss: {avg:.4f}"
                )
                running_loss = 0.0

        # 🔒 Чекпойнти (за потреби)
        # torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")

# ---------------------------
# CLI аргументи
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Навчання Retinexformer на LOL")
    p.add_argument("--dataset_root", type=str, default=DEF_DS_ROOT, help="Шлях до папки our485")
    p.add_argument("--epochs", type=int, default=DEF_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEF_BS)
    p.add_argument(
        "--crop", type=int, default=DEF_CROP, help="Розмір випадкового кропа (кратний patch)"
    )
    p.add_argument("--patch", type=int, default=DEF_PATCH, help="Розмір патча для трансформера")
    p.add_argument("--lr", type=float, default=DEF_LR)
    return p.parse_args()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")  # пришвидшує на нових GPU/CPU
    opt = parse_args()
    train(opt)
