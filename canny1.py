#!/usr/bin/env python3
"""
Виділення контурів методом Canny + оцінка якості (Precision, Recall, F-measure, IoU).

• Зчитує вхідне зображення, переводить у відтінки сірого.
• Застосовує cv2.Canny із заданими порогами.
• Зберігає бінарну карту контурів.
• За наявності еталонної маски рахує TP/FP/FN → Precision, Recall, F1, IoU.
"""

from pathlib import Path
import cv2
import numpy as np


# ──────────────────────────── К О Н С Т А Н Т И ───────────────────────────────
INPUT_IMAGE   = Path("data/singles/Dark-Image3.png")   # вихідне зображення
# INPUT_IMAGE   = Path("output/singles/output1_ssr100.jpg")
OUTPUT_EDGES  = Path("output/singles/edgesPlus.png")       # файл для збереження контурів
THRESH_LOW    = 20                         # нижній поріг Canny
THRESH_HIGH   = 150                       # верхній поріг Canny
REF_MASK      = None                       # напр. Path("gt_edges.png") або None
EPS           = 1e-8                       # запобігання діленню на 0
# ──────────────────────────────────────────────────────────────────────────────


def canny_edges(gray: np.ndarray,
                low: int = 50,
                high: int = 150) -> np.ndarray:
    """Повернути бінарну (0/255) карту контурів."""
    edges = cv2.Canny(gray, low, high)
    return edges  # уже uint8: 0 або 255


def compute_metrics(pred: np.ndarray,
                    ref: np.ndarray) -> tuple[float, float, float, float]:
    """
    Precision, Recall, F1, IoU для двох бінарних карт (uint8 0/255).
    Пікселі ≠0 трактуються як «контур».
    """
    if pred.shape != ref.shape:
        raise ValueError("Розміри результату та еталону не збігаються.")

    pred_bin = pred > 0
    ref_bin  = ref > 0

    tp = np.logical_and(pred_bin, ref_bin).sum(dtype=np.float64)
    fp = np.logical_and(pred_bin, np.logical_not(ref_bin)).sum(dtype=np.float64)
    fn = np.logical_and(np.logical_not(pred_bin), ref_bin).sum(dtype=np.float64)

    precision = tp / (tp + fp + EPS)
    recall    = tp / (tp + fn + EPS)
    f1        = 2 * precision * recall / (precision + recall + EPS)
    iou       = tp / (tp + fp + fn + EPS)

    return precision, recall, f1, iou


if __name__ == "__main__":
    # 1. Зчитування та перетворення у відтінки сірого
    img = cv2.imread(str(INPUT_IMAGE), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не вдалося відкрити {INPUT_IMAGE}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Canny
    edge_map = canny_edges(gray, THRESH_LOW, THRESH_HIGH)
    cv2.imwrite(str(OUTPUT_EDGES), edge_map)
    print(f"Контури збережено → {OUTPUT_EDGES}")

    # 3. Оцінка, якщо задано еталон
    if REF_MASK:
        ref = cv2.imread(str(REF_MASK), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            raise FileNotFoundError(f"Не вдалося відкрити {REF_MASK}")

        prec, rec, f1, iou = compute_metrics(edge_map, ref)
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F-мірa   : {f1:.4f}")
        print(f"IoU      : {iou:.4f}")
