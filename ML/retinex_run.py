# -*- coding: utf-8 -*-
"""
Retinexformer Inference
======================

Простий скрипт, який завантажує натреновані ваги й висвітлює низько‑світлинне зображення.
Усі шляхи та параметри задаються у **константах** нижче, тож можна запускати без аргументів.

Покроково:
1. Завантажує чекпойнт `MODEL_PATH` (файл *.pth* із тренінґ‑скрипта).
2. Відновлює архітектуру Retinexformer з тим самим `patch_size`.
3. Зчитує `INPUT_IMAGE`, обрізає, щоб висота/ширина кратні патчу.
4. Проганяє через мережу (GPU, якщо доступний) і зберігає результат у `OUTPUT_IMAGE`.

Запуск у терміналі: `python retinexformer_inference.py`
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms

# ---------------------
# Константи (змінюйте)
# ---------------------
MODEL_PATH = Path("checkpoints/retinexformer_best.pth")  # шлях до збережених ваг
INPUT_IMAGE = Path("../data/singles/Dark-Image2.png")                       # вхідне зображення
OUTPUT_IMAGE = Path("../output/lol_dataset/demo_enhanced2.png")                # куди зберегти результат

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# Модель (спрощена версія з тренінґ‑скрипта)
# -------------------------------------------------------
class Retinexformer(torch.nn.Module):
    def __init__(self, dim: int = 64, depth: int = 4, heads: int = 8, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.enc_embed = torch.nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        enc_layer = torch.nn.TransformerEncoderLayer(
            dim, nhead=heads, dim_feedforward=dim * 4, batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=depth)

        dec_layer = torch.nn.TransformerEncoderLayer(
            dim, nhead=heads, dim_feedforward=dim * 4, batch_first=True
        )
        self.decoder = torch.nn.TransformerEncoder(dec_layer, num_layers=depth)

        # Змінюємо вихідний шар на proj замість out_conv
        self.proj = torch.nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def forward(self, x):  # (B,3,H,W)
        B, _, H, W = x.shape
        feat = self.enc_embed(x)                     # (B,C,H',W')
        H_, W_ = feat.shape[-2:]
        tokens = feat.flatten(2).permute(0, 2, 1)    # (B,N,C)
        tokens = self.encoder(tokens)
        tokens = self.decoder(tokens) + tokens       # skip
        feat_dec = tokens.permute(0, 2, 1).view(B, -1, H_, W_)
        feat_up = torch.nn.functional.interpolate(
            feat_dec, scale_factor=self.patch_size, mode="bilinear", align_corners=False
        )
        # Змінюємо використання out_conv на proj
        return torch.sigmoid(self.proj(feat_up))

# ---------------------
# Завантаження моделі
# ---------------------

def load_model() -> Retinexformer:
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    patch = ckpt.get("patch", 8)
    model = Retinexformer(patch_size=patch)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()
    return model, patch

# ---------------------
# Обробка одного файлу
# ---------------------

def enhance_image(model: Retinexformer, patch: int):
    # Зчитуємо зображення
    img = Image.open(INPUT_IMAGE).convert("RGB")

    # Робимо розмір кратним розміру патча
    w, h = img.size
    new_w = math.floor(w / patch) * patch
    new_h = math.floor(h / patch) * patch
    if (new_w, new_h) != (w, h):
        img = img.crop((0, 0, new_w, new_h))

    to_tensor = transforms.ToTensor()
    inp = to_tensor(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(inp).clamp(0, 1)

    vutils.save_image(out.cpu(), OUTPUT_IMAGE)
    print(f"[Done] Збережено висвітлене зображення → {OUTPUT_IMAGE.resolve()}")

# ---------------------
# main
# ---------------------

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не знайдено чекпойнт: {MODEL_PATH}")
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(f"Не знайдено вхідне зображення: {INPUT_IMAGE}")

    model, patch = load_model()
    enhance_image(model, patch)
