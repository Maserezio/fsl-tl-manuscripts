"""Shared helpers for the HTR-based segmentation evaluation (CPU TrOCR)."""
import _paths  # noqa: F401,E402  (center_line + analysis folders on sys.path)
import time
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def load(name):
    proc = TrOCRProcessor.from_pretrained(name)
    model = VisionEncoderDecoderModel.from_pretrained(name).eval()
    # transformers 5.x leaves the sinusoidal position table of some TrOCR checkpoints on the meta device
    for m in model.modules():
        if type(m).__name__ == "TrOCRSinusoidalPositionalEmbedding" and m.weights.is_meta:
            m.weights = m.get_embedding(m.weights.shape[0], m.embedding_dim, m.padding_idx)
    dev = __import__("os").environ.get("HTR_DEVICE", "cpu")
    model = model.to(dev)
    for m in model.modules():   # the re-created position table is a plain attribute, not a buffer
        if type(m).__name__ == "TrOCRSinusoidalPositionalEmbedding":
            m.weights = m.weights.to(dev)
    return proc, model


def line_image(img, poly, pad=4):
    """Bounding-box crop of a line polygon; pixels outside the polygon are set to the page background (white)."""
    x0, y0 = np.maximum(poly.min(0) - pad, 0)
    x1, y1 = np.minimum(poly.max(0) + pad, [img.shape[1] - 1, img.shape[0] - 1])
    crop = img[y0:y1 + 1, x0:x1 + 1].copy()
    m = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(m, [poly - [x0, y0]], 1)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    crop[m == 0] = 255
    return Image.fromarray(crop)


@torch.no_grad()
def recognise(proc, model, images, bs=8, max_new_tokens=96):
    out = []
    for i in range(0, len(images), bs):
        pv = proc(images=images[i:i + bs], return_tensors="pt").pixel_values.to(model.device)
        ids = model.generate(pv, max_new_tokens=max_new_tokens, num_beams=1)
        out += proc.batch_decode(ids, skip_special_tokens=True)
    return out


def cer(ref, hyp):
    """Levenshtein distance / len(ref)."""
    if not ref:
        return float(len(hyp) > 0)
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / len(ref)
