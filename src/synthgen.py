"""
Synthetic anomaly generation utilities.

This file supports:
- method="fallback": draws scratch/crack-like lines + mild blur/noise on the image.
- method="anomalyanything": placeholder hook. If you have AnomalyAnything installed,
  call it here and return generated PIL image.

This is YOUR extension point: keep your modifications here to show contribution.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import random
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PROMPT_TO_SEVERITY = {
    "small scratch": "mild",
    "visible scratch": "moderate",
    "large crack": "severe",
}

def _overlay_scratch(img: Image.Image, severity: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    w, h = img.size
    out = img.copy()
    draw = ImageDraw.Draw(out)

    if severity == "mild":
        n_lines = rng.randint(1, 2)
        width = rng.randint(1, 2)
        alpha = 80
    elif severity == "moderate":
        n_lines = rng.randint(2, 4)
        width = rng.randint(2, 4)
        alpha = 120
    else:
        n_lines = rng.randint(4, 7)
        width = rng.randint(3, 6)
        alpha = 180

    for _ in range(n_lines):
        x0, y0 = rng.randint(0, w-1), rng.randint(0, h-1)
        angle = rng.random() * 2 * math.pi
        length = rng.randint(int(0.2*min(w,h)), int(0.7*min(w,h)))
        x1 = int(x0 + length * math.cos(angle))
        y1 = int(y0 + length * math.sin(angle))
        x1 = max(0, min(w-1, x1))
        y1 = max(0, min(h-1, y1))
        color = (alpha, alpha, alpha)  # gray-ish scratch
        draw.line((x0, y0, x1, y1), fill=color, width=width)

    # Add slight blur/noise based on severity
    if severity == "mild":
        out = out.filter(ImageFilter.GaussianBlur(radius=0.3))
    elif severity == "moderate":
        out = out.filter(ImageFilter.GaussianBlur(radius=0.6))
    else:
        out = out.filter(ImageFilter.GaussianBlur(radius=1.0))

    # Add subtle noise
    arr = np.array(out).astype(np.float32)
    noise_std = {"mild": 2.0, "moderate": 5.0, "severe": 9.0}[severity]
    noise = np.random.default_rng(seed).normal(0, noise_std, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def generate_anomaly(
    img: Image.Image,
    prompt: str,
    method: str = "fallback",
    seed: int = 0,
    extra: Optional[Dict] = None,
) -> Image.Image:
    method = method.lower()
    extra = extra or {}

    if method == "fallback":
        # map prompt -> severity (or default moderate)
        pl = prompt.lower()
        if "small" in pl:
            sev = "mild"
        elif "large" in pl or "crack" in pl:
            sev = "severe"
        else:
            sev = "moderate"
        return _overlay_scratch(img, sev, seed)

    if method == "anomalyanything":
        # ---- HOOK: integrate AnomalyAnything here ----
        # Example (pseudo):
        # from anomalyanything import AnomalyAnythingPipeline
        # pipe = AnomalyAnythingPipeline.from_pretrained(...)
        # out = pipe(image=img, prompt=prompt, seed=seed, **extra)
        # return out
        raise NotImplementedError(
            "AnomalyAnything integration not provided in this template. "
            "Please implement inside src/synthgen.py (method='anomalyanything')."
        )

    raise ValueError(f"Unknown generator method: {method}")
