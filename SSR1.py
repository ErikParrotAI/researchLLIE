#!/usr/bin/env python3
"""
Single-Scale Retinex (SSR) з необов’язковими метриками SSIM та PSNR.
Усе налаштовується константами нижче.
"""

from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


# ──────────────────────────── К О Н С Т А Н Т И ───────────────────────────────
# INPUT_IMAGE  = Path("data/singles/Dark-Image1.png")         # вихідне зображення
# OUTPUT_IMAGE = Path("output/singles/output1_ssr100.jpg")    # куди зберегти результат

INPUT_IMAGE  = Path("data/lol_dataset/eval15/low/778.png")
OUTPUT_IMAGE = Path("output/lol_dataset/output778_ssr100.jpg")

SIGMA        = 100
REF_IMAGE    = Path("data/lol_dataset/eval15/high/778.png")                      # напр. Path("gt.jpg") або None
# ──────────────────────────────────────────────────────────────────────────────


def single_scale_retinex(img_bgr: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Повертає SSR-зображення (uint8, 0-255) для вхідного BGR-зображення."""
    img_float = img_bgr.astype(np.float32) + 1e-8           # запобігаємо log(0)
    blur = cv2.GaussianBlur(img_float, (0, 0), sigmaX=sigma)

    # логарифмічне відношення (можна замінити на img_float / blur)
    retinex = np.log(img_float) - np.log(blur)

    # нормалізація до [0, 255] і повернення uint8
    retinex_norm = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex_norm.astype(np.uint8)


def calc_metrics(target: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    """Повертає (SSIM, PSNR) між двома BGR-зображеннями однакового розміру."""
    if target.shape != reference.shape:
        raise ValueError("Розміри зображень не збігаються.")

    # у float64 [0,1] для skimage
    t = target.astype(np.float64) / 255.0
    r = reference.astype(np.float64) / 255.0

    ssim_vals = [ssim(r[..., ch], t[..., ch], data_range=1.0) for ch in range(3)]
    psnr_vals = [psnr(r[..., ch], t[..., ch], data_range=1.0) for ch in range(3)]
    return float(np.mean(ssim_vals)), float(np.mean(psnr_vals))


if __name__ == "__main__":
    # 1. зчитуємо вхід
    img = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не вдалося відкрити {INPUT_IMAGE}")

    # 2. застосовуємо SSR
    result = single_scale_retinex(img, sigma=SIGMA)
    cv2.imwrite(str(OUTPUT_IMAGE), result)
    print(f"SSR-зображення збережено → {OUTPUT_IMAGE}")

    # 3. метрики, якщо задано еталон
    if REF_IMAGE:
        ref_img = cv2.imread(str(REF_IMAGE), cv2.IMREAD_COLOR)
        if ref_img is None:
            raise FileNotFoundError(f"Не вдалося відкрити {REF_IMAGE}")

        ssim_val, psnr_val = calc_metrics(result, ref_img)
        print(f"SSIM: {ssim_val:.4f}")
        print(f"PSNR: {psnr_val:.2f} дБ")
