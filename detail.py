# detail.py
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw

# Import from your existing graphics module
from graphics import BaseBlock, REGISTRY, _ensure_image, _parse_color, _norm01, help, params


# ---------------------------------------------------------------------------
# Sub-pipeline runner (used by ApplyMaskedEffect and a few utility blocks)
# ---------------------------------------------------------------------------
def _run_sub_pipeline(
    pipeline_str: str,
    width: int,
    height: int,
    all_params: Dict[str, Dict[str, Any]],
    initial_img: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Runs a sequence of graphics blocks defined by a pipeline string.
    all_params is the full extras dict; blocks pick what they need by name.
    """
    blocks = [b.strip().lower() for b in (pipeline_str or "").split("|") if b.strip()]
    if not blocks:
        return _ensure_image(initial_img, width, height)

    img = initial_img
    for name in blocks:
        try:
            params = all_params.get(name, {})
            block = REGISTRY.create(name)
            img = block.process(img, width, height, params=params)
            if img is None:
                raise RuntimeError(f"Sub-pipeline block '{name}' returned None")
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            if img.size != (width, height):
                print(f"Warning (sub-pipeline): Block '{name}' changed size, resizing.", file=sys.stderr)
                img = img.resize((width, height), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error running sub-pipeline block '{name}': {e}", file=sys.stderr)
            return _ensure_image(img, width, height)

    return _ensure_image(img, width, height)


# =============================================================================
# Core detail / stylistic blocks
# =============================================================================

@help("Applies a simple sharpening effect using PIL's SHARPEN kernel.")
@params({})
@dataclass
class SharpenBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        return img.filter(ImageFilter.SHARPEN)


@help("Unsharp mask sharpening.\nParams: radius(px), percent(0..500), threshold(0..255).")
@params({
    "radius":    {"type": "float", "default": 2.0, "min": 0.0, "max": 200.0, "step": 0.1},
    "percent":   {"type": "int",   "default": 150, "min": 0,   "max": 500,   "step": 1},
    "threshold": {"type": "int",   "default": 3,   "min": 0,   "max": 255,   "step": 1},
})
@dataclass
class UnsharpMaskBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        radius = float(params.get("radius", 2.0))
        percent = int(params.get("percent", 150))
        threshold = int(params.get("threshold", 3))
        return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


@help("High-pass sharpen: subtract blur to get detail and add back.\nParams: radius(px), strength(0..2), preserve_color(bool).")
@params({
    "radius":         {"type": "float", "default": 4.0, "min": 0.0, "max": 200.0, "step": 0.1},
    "strength":       {"type": "float", "default": 1.0, "min": 0.0, "max": 2.0,   "step": 0.01},
    "preserve_color": {"type": "bool",  "default": True},
})
@dataclass
class HighPassBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        src = _ensure_image(img, width, height)
        radius = float(params.get("radius", 4.0))
        strength = float(params.get("strength", 1.0))
        preserve_color = bool(params.get("preserve_color", True))

        arr = np.asarray(src).astype(np.float32) / 255.0
        blur = np.asarray(src.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32) / 255.0
        high = arr[..., :3] - blur[..., :3]

        if not preserve_color:
            lum = (0.299 * high[..., 0] + 0.587 * high[..., 1] + 0.114 * high[..., 2])[..., None]
            high = np.repeat(lum, 3, axis=2)

        out_rgb = np.clip(arr[..., :3] + strength * high, 0.0, 1.0)
        out_a = arr[..., 3:4] if arr.shape[2] == 4 else np.ones_like(out_rgb[..., :1])
        out = np.concatenate([out_rgb, out_a], axis=2)
        return Image.fromarray((out * 255.0).astype(np.uint8), "RGBA")


@help("Edge detect overlay using FIND_EDGES; composites black edges using alpha.")
@params({})
@dataclass
class EdgeDetectBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        img_gray = base.convert("L")
        edges = img_gray.filter(ImageFilter.FIND_EDGES)

        alpha = ImageOps.invert(edges)  # strong edge -> higher alpha
        edge_rgba = Image.merge(
            "RGBA",
            (
                Image.new("L", base.size, 0),
                Image.new("L", base.size, 0),
                Image.new("L", base.size, 0),
                alpha,
            ),
        )
        return Image.alpha_composite(base, edge_rgba)


@help("Sobel edge magnitude overlay.\nParams: strength(0..1), invert(bool).")
@params({
    "strength": {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
    "invert":   {"type": "bool",  "default": False},
})
@dataclass
class SobelEdgeBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        strength = _norm01(float(params.get("strength", 0.8)))
        invert = bool(params.get("invert", False))

        g = np.asarray(base.convert("L")).astype(np.float32) / 255.0
        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)

        def simple_conv(a, k):
            kh, kw = k.shape
            pad_h, pad_w = kh // 2, kw // 2
            ap = np.pad(a, ((pad_h, pad_h), (pad_w, pad_w)), mode="edge")
            out = np.zeros_like(a)
            for y in range(a.shape[0]):
                for x in range(a.shape[1]):
                    region = ap[y:y+kh, x:x+kw]
                    out[y, x] = np.sum(region * k)
            return out

        # Prefer scipy if present, otherwise fallback
        try:
            from scipy.signal import convolve2d as _c  # type: ignore
            gx = _c(g, Kx, mode="same", boundary="symm")
            gy = _c(g, Ky, mode="same", boundary="symm")
        except Exception:
            gx = simple_conv(g, Kx)
            gy = simple_conv(g, Ky)

        mag = np.sqrt(gx * gx + gy * gy)
        mag = mag / (mag.max() + 1e-9)
        if invert:
            mag = 1.0 - mag

        edge_alpha = (mag * 255.0 * strength).astype(np.uint8)
        edge_rgba = Image.merge(
            "RGBA",
            (
                Image.new("L", base.size, 0),
                Image.new("L", base.size, 0),
                Image.new("L", base.size, 0),
                Image.fromarray(edge_alpha),
            ),
        )
        return Image.alpha_composite(base, edge_rgba)


@help("Chromatic aberration by shifting R and B channels.\nParams: amount_px(int), angle_deg(float).")
@params({
    "amount_px": {"type": "int",   "default": 2,    "min": 0,   "max": 200, "step": 1},
    "angle_deg": {"type": "float", "default": 45.0, "min": -360.0, "max": 360.0, "step": 1.0},
})
@dataclass
class ChromaticAberrationBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        amount_px = int(params.get("amount_px", 2))
        angle_deg = float(params.get("angle_deg", 45.0))
        if amount_px == 0:
            return base

        dx = int(round(amount_px * math.cos(math.radians(angle_deg))))
        dy = int(round(amount_px * math.sin(math.radians(angle_deg))))

        r, g, b, a = base.split()

        r_shifted = Image.new("L", (width, height))
        r_shifted.paste(r, (-dx, -dy))
        b_shifted = Image.new("L", (width, height))
        b_shifted.paste(b, (dx, dy))

        return Image.merge("RGBA", (r_shifted, g, b_shifted, a))


@help("Film grain noise.\nParams: amount(0..1), colored(bool).")
@params({
    "amount":  {"type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
    "colored": {"type": "bool",  "default": False},
})
@dataclass
class FilmGrainBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        amount = _norm01(float(params.get("amount", 0.05)))
        colored = bool(params.get("colored", False))
        if amount <= 0:
            return base

        data = np.array(base).astype(np.float32) / 255.0
        if colored:
            noise = np.random.normal(0.0, amount, size=data[..., :3].shape).astype(np.float32)
        else:
            n = np.random.normal(0.0, amount, size=(height, width, 1)).astype(np.float32)
            noise = np.repeat(n, 3, axis=2)
        data[..., :3] = np.clip(data[..., :3] + noise, 0.0, 1.0)
        return Image.fromarray((data * 255.0).astype(np.uint8), "RGBA")


@help("Bloom/glow: isolate brights, blur, add back.\nParams: threshold(0..1), radius(px), strength(0..3).")
@params({
    "threshold": {"type": "float", "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01},
    "radius":    {"type": "float", "default": 8.0, "min": 0.0, "max": 200.0, "step": 0.1},
    "strength":  {"type": "float", "default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01},
})
@dataclass
class BloomBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        threshold = _norm01(float(params.get("threshold", 0.7)))
        radius = float(params.get("radius", 8.0))
        strength = float(params.get("strength", 1.0))

        arr = np.asarray(base).astype(np.float32) / 255.0
        lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        mask = (lum > threshold).astype(np.float32)[..., None]
        brights = arr.copy()
        brights[..., :3] *= mask

        bright_img = Image.fromarray((np.clip(brights, 0, 1) * 255.0).astype(np.uint8), "RGBA")
        blurred = bright_img.filter(ImageFilter.GaussianBlur(radius))
        add = np.asarray(blurred).astype(np.float32) / 255.0

        out = np.clip(arr + strength * add, 0.0, 1.0)
        return Image.fromarray((out * 255.0).astype(np.uint8), "RGBA")


@help("Directional motion blur by averaging shifted copies.\nParams: length(px), angle_deg, samples(int).")
@params({
    "length":    {"type": "int",   "default": 12,  "min": 1, "max": 2000, "step": 1},
    "angle_deg": {"type": "float", "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0},
    "samples":   {"type": "int",   "default": 8,   "min": 1, "max": 256, "step": 1},
})
@dataclass
class MotionBlurBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        length = max(1, int(params.get("length", 12)))
        angle_deg = float(params.get("angle_deg", 0.0))
        samples = max(1, int(params.get("samples", 8)))

        dx = math.cos(math.radians(angle_deg))
        dy = math.sin(math.radians(angle_deg))

        acc = np.zeros((height, width, 4), dtype=np.float32)
        for i in range(samples):
            t = (i / (samples - 1) if samples > 1 else 0.5) - 0.5
            sx = int(round(t * length * dx))
            sy = int(round(t * length * dy))
            shifted = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            shifted.paste(base, (sx, sy))
            acc += np.asarray(shifted).astype(np.float32)
        acc /= float(samples)
        return Image.fromarray(acc.clip(0, 255).astype(np.uint8), "RGBA")


@help("Hue rotate in HSV.\nParams: degrees(-360..360), sat_mult, val_mult.")
@params({
    "degrees":  {"type": "float", "default": 30.0, "min": -360.0, "max": 360.0, "step": 1.0},
    "sat_mult": {"type": "float", "default": 1.0,  "min": 0.0, "max": 5.0, "step": 0.01},
    "val_mult": {"type": "float", "default": 1.0,  "min": 0.0, "max": 5.0, "step": 0.01},
})
@dataclass
class HueRotateBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        deg = float(params.get("degrees", 30.0))
        sat_mult = float(params.get("sat_mult", 1.0))
        val_mult = float(params.get("val_mult", 1.0))

        arr = np.asarray(base).astype(np.float32) / 255.0
        r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

        maxc = np.max(arr[..., :3], axis=2)
        minc = np.min(arr[..., :3], axis=2)
        v = maxc
        s = (maxc - minc) / (maxc + 1e-9)

        rc = (maxc - r) / (maxc - minc + 1e-9)
        gc = (maxc - g) / (maxc - minc + 1e-9)
        bc = (maxc - b) / (maxc - minc + 1e-9)

        h = np.zeros_like(maxc)
        h = np.where(maxc == r, (bc - gc), h)
        h = np.where(maxc == g, 2.0 + (rc - bc), h)
        h = np.where(maxc == b, 4.0 + (gc - rc), h)
        h = (h / 6.0) % 1.0

        h = (h + deg / 360.0) % 1.0
        s = np.clip(s * sat_mult, 0.0, 1.0)
        v = np.clip(v * val_mult, 0.0, 1.0)

        i = np.floor(h * 6.0).astype(int)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i_mod = i % 6

        rgb = np.stack([
            np.where(i_mod == 0, v, np.where(i_mod == 1, q, np.where(i_mod == 2, p, np.where(i_mod == 3, p, np.where(i_mod == 4, t, v))))),
            np.where(i_mod == 0, t, np.where(i_mod == 1, v, np.where(i_mod == 2, v, np.where(i_mod == 3, q, np.where(i_mod == 4, p, p))))),
            np.where(i_mod == 0, p, np.where(i_mod == 1, p, np.where(i_mod == 2, t, np.where(i_mod == 3, v, np.where(i_mod == 4, v, q))))),
        ], axis=2)

        out = np.concatenate([rgb, a[..., None]], axis=2)
        return Image.fromarray((out * 255.0).clip(0, 255).astype(np.uint8), "RGBA")


@help("Per-channel lift/gamma/gain (color balance).\nParams: lift_* (-1..1), gamma_* (0.1..3), gain_* (0..3).")
@params({
    "lift_r":  {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
    "lift_g":  {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
    "lift_b":  {"type": "float", "default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
    "gamma_r": {"type": "float", "default": 1.0, "min": 0.1,  "max": 3.0, "step": 0.01},
    "gamma_g": {"type": "float", "default": 1.0, "min": 0.1,  "max": 3.0, "step": 0.01},
    "gamma_b": {"type": "float", "default": 1.0, "min": 0.1,  "max": 3.0, "step": 0.01},
    "gain_r":  {"type": "float", "default": 1.0, "min": 0.0,  "max": 3.0, "step": 0.01},
    "gain_g":  {"type": "float", "default": 1.0, "min": 0.0,  "max": 3.0, "step": 0.01},
    "gain_b":  {"type": "float", "default": 1.0, "min": 0.0,  "max": 3.0, "step": 0.01},
})
@dataclass
class ColorBalanceBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        arr = np.asarray(base).astype(np.float32) / 255.0
        rgb = arr[..., :3]
        a = arr[..., 3:4]

        # FIXED: correct param getter
        def getf(k: str, default: float) -> float:
            try:
                return float(params.get(k, default))
            except Exception:
                return float(default)

        lift = np.array([getf("lift_r", 0.0),
                         getf("lift_g", 0.0),
                         getf("lift_b", 0.0)], dtype=np.float32)
        gamma = np.array([max(0.1, getf("gamma_r", 1.0)),
                          max(0.1, getf("gamma_g", 1.0)),
                          max(0.1, getf("gamma_b", 1.0))], dtype=np.float32)
        gain = np.array([max(0.0, getf("gain_r", 1.0)),
                         max(0.0, getf("gain_g", 1.0)),
                         max(0.0, getf("gain_b", 1.0))], dtype=np.float32)

        rgb = rgb + lift[None, None, :]
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = np.power(rgb, 1.0 / gamma[None, None, :])
        rgb = np.clip(rgb * gain[None, None, :], 0.0, 1.0)

        out = np.concatenate([rgb, a], axis=2)
        return Image.fromarray((out * 255.0).astype(np.uint8), "RGBA")


@help("Gradient map: luminance -> mix between two colors.\nParams: low_color, high_color.")
@params({
    "low_color":  {"type": "color", "default": "#000000"},
    "high_color": {"type": "color", "default": "#FFFFFF"},
})
@dataclass
class GradientMapBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        c0 = _parse_color(params.get("low_color", "#000000"), (0, 0, 0, 255))
        c1 = _parse_color(params.get("high_color", "#FFFFFF"), (255, 255, 255, 255))

        arr = np.asarray(base).astype(np.float32) / 255.0
        a = arr[..., 3:4]
        lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])[..., None]

        c0v = np.array(c0, dtype=np.float32) / 255.0
        c1v = np.array(c1, dtype=np.float32) / 255.0
        rgb = c0v[None, None, :3] + (c1v[:3] - c0v[:3])[None, None, :] * lum
        out = np.concatenate([rgb, a], axis=2)
        return Image.fromarray((out * 255.0).astype(np.uint8), "RGBA")


@help("Posterize image color depth.\nParams: bits(1..8).")
@params({
    "bits": {"type": "int", "default": 4, "min": 1, "max": 8, "step": 1},
})
@dataclass
class PosterizeBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        bits = max(1, min(8, int(params.get("bits", 4))))
        rgb = base.convert("RGB")
        a = base.getchannel("A")
        poster = ImageOps.posterize(rgb, bits).convert("RGBA")
        return Image.merge("RGBA", (*poster.split()[:3], a))


@help("Binary threshold on luminance.\nParams: thresh(0..1), low_color, high_color, keep_alpha.")
@params({
    "thresh":     {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "low_color":  {"type": "color", "default": "#00000000"},
    "high_color": {"type": "color", "default": "#FFFFFFFF"},
    "keep_alpha": {"type": "bool",  "default": True},
})
@dataclass
class ThresholdBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        thresh = _norm01(float(params.get("thresh", 0.5)))
        low_color = _parse_color(params.get("low_color", "#00000000"), (0, 0, 0, 0))
        high_color = _parse_color(params.get("high_color", "#FFFFFFFF"), (255, 255, 255, 255))
        keep_alpha = bool(params.get("keep_alpha", True))

        arr = np.asarray(base).astype(np.float32) / 255.0
        lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        mask = (lum >= thresh).astype(np.float32)[..., None]
        low = (np.array(low_color)[None, None, :] / 255.0).astype(np.float32)
        high = (np.array(high_color)[None, None, :] / 255.0).astype(np.float32)

        out = low * (1.0 - mask) + high * mask
        if keep_alpha:
            out[..., 3] = arr[..., 3]
        return Image.fromarray((np.clip(out, 0, 1) * 255.0).astype(np.uint8), "RGBA")


@help("Pixelate by downscale/upscale.\nParams: factor(>1), method(nearest|box).")
@params({
    "factor": {"type": "float", "default": 8.0, "min": 1.0, "max": 2048.0, "step": 1.0},
    "method": {"type": "enum",  "default": "nearest", "options": ["nearest", "box"]},
})
@dataclass
class PixelateBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        factor = max(1.0, float(params.get("factor", 8.0)))
        method = str(params.get("method", "nearest")).lower()

        down_w = max(1, int(round(width / factor)))
        down_h = max(1, int(round(height / factor)))

        down = base.resize((down_w, down_h), Image.Resampling.BOX)
        up_mode = Image.Resampling.NEAREST if method == "nearest" else Image.Resampling.BOX
        return down.resize((width, height), up_mode)


@help("Halftone dot overlay.\nParams: cell(px), strength(0..1), angle_deg.")
@params({
    "cell":      {"type": "int",   "default": 8,    "min": 2, "max": 512, "step": 1},
    "strength":  {"type": "float", "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01},
    "angle_deg": {"type": "float", "default": 45.0, "min": -360.0, "max": 360.0, "step": 1.0},
})
@dataclass
class HalftoneBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        cell = max(2, int(params.get("cell", 8)))
        strength = _norm01(float(params.get("strength", 0.75)))
        angle_deg = float(params.get("angle_deg", 45.0))

        rot = base.rotate(angle_deg, expand=True, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0, 0))
        rw, rh = rot.size

        arr = np.asarray(rot.convert("RGBA")).astype(np.float32) / 255.0
        lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])

        dots = Image.new("L", (rw, rh), 0)
        draw = ImageDraw.Draw(dots)
        for y in range(0, rh, cell):
            for x in range(0, rw, cell):
                l = float(lum[min(rh - 1, y), min(rw - 1, x)])
                r = (1.0 - l) * (cell * 0.5)
                if r > 0.5:
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=int(255 * strength))

        dots_back = dots.rotate(-angle_deg, expand=False, resample=Image.Resampling.BILINEAR)
        dots_back = dots_back.resize((width, height), Image.Resampling.BILINEAR)

        overlay = Image.merge(
            "RGBA",
            (
                Image.new("L", (width, height), 0),
                Image.new("L", (width, height), 0),
                Image.new("L", (width, height), 0),
                dots_back,
            ),
        )
        return Image.alpha_composite(base, overlay)


@help("Emboss filter with optional blend strength.")
@params({
    "strength": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
})
@dataclass
class EmbossBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        strength = _norm01(float(params.get("strength", 1.0)))
        emb = base.filter(ImageFilter.EMBOSS)
        if strength >= 0.999:
            return emb
        return Image.blend(base, emb, strength)


@help("Tilt-shift: sharp focus band with blur outside.\nParams: focus_y(0..1), band(0..1), feather(0..1), radius(px).")
@params({
    "focus_y": {"type": "float", "default": 0.5,  "min": 0.0, "max": 1.0, "step": 0.01},
    "band":    {"type": "float", "default": 0.3,  "min": 0.0, "max": 1.0, "step": 0.01},
    "feather": {"type": "float", "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
    "radius":  {"type": "float", "default": 8.0,  "min": 0.0, "max": 200.0, "step": 0.1},
})
@dataclass
class TiltShiftBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        focus_y = _norm01(float(params.get("focus_y", 0.5)))
        band = _norm01(float(params.get("band", 0.3)))
        feather = _norm01(float(params.get("feather", 0.25)))
        radius = float(params.get("radius", 8.0))

        blurred = base.filter(ImageFilter.GaussianBlur(radius))
        y = np.linspace(0.0, 1.0, height)[:, None]
        center = focus_y
        half = band * 0.5

        m = np.clip((half - np.abs(y - center)) / max(half, 1e-6), 0.0, 1.0)
        m = m * m * (3 - 2 * m)  # smoothstep
        m = np.clip(m + feather * (1 - m), 0.0, 1.0)
        m = np.repeat(m, width, axis=1)[..., None]

        a_base = np.asarray(base).astype(np.float32)
        a_blur = np.asarray(blurred).astype(np.float32)
        out = a_base * m + a_blur * (1 - m)
        return Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGBA")


@help("Displacement map warp.\nParams: disp_path(optional), scale_x, scale_y, channel(r|g|b|a|luma).")
@params({
    "disp_path": {"type": "path",  "default": None, "nullable": True, "hint": "Image path"},
    "scale_x":   {"type": "float", "default": 10.0, "min": -500.0, "max": 500.0, "step": 0.1},
    "scale_y":   {"type": "float", "default": 10.0, "min": -500.0, "max": 500.0, "step": 0.1},
    "channel":   {"type": "enum",  "default": "luma", "options": ["luma", "r", "g", "b", "a"]},
})
@dataclass
class DisplacementMapBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        disp_path = params.get("disp_path", None)
        scale_x = float(params.get("scale_x", 10.0))
        scale_y = float(params.get("scale_y", 10.0))
        channel = str(params.get("channel", "luma")).lower()

        if disp_path:
            try:
                D = Image.open(str(disp_path)).convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
            except Exception as e:
                print(f"[displacement] failed to load disp_path: {e}", file=sys.stderr)
                D = base
        else:
            D = base

        darr = np.asarray(D).astype(np.float32) / 255.0
        if channel == "luma":
            disp = (0.299 * darr[..., 0] + 0.587 * darr[..., 1] + 0.114 * darr[..., 2])
        else:
            idx = {"r": 0, "g": 1, "b": 2, "a": 3}.get(channel, 0)
            disp = darr[..., idx]
        disp = (disp * 2.0 - 1.0)  # -1..1

        yy, xx = np.indices((height, width), dtype=np.float32)
        map_x = np.clip(xx + disp * scale_x, 0, width - 1)
        map_y = np.clip(yy + disp * scale_y, 0, height - 1)

        src = np.asarray(base).astype(np.float32)
        x0 = np.floor(map_x).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, width - 1)
        y0 = np.floor(map_y).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, height - 1)

        wx = map_x - x0
        wy = map_y - y0
        wa = (1 - wx) * (1 - wy)
        wb = wx * (1 - wy)
        wc = (1 - wx) * wy
        wd = wx * wy

        out = (
            src[y0, x0] * wa[..., None] +
            src[y0, x1] * wb[..., None] +
            src[y1, x0] * wc[..., None] +
            src[y1, x1] * wd[..., None]
        )
        return Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGBA")


@help("Colored vignette with curve power.\nParams: strength(0..1), color, power(>0), cx, cy.")
@params({
    "strength": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "color":    {"type": "color", "default": "#000000"},
    "power":    {"type": "float", "default": 1.5, "min": 0.1, "max": 10.0, "step": 0.01},
    "cx":       {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy":       {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
})
@dataclass
class VignettePlusBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        strength = _norm01(float(params.get("strength", 0.5)))
        color = _parse_color(params.get("color", "#000000"), (0, 0, 0, 255))
        power = max(0.1, float(params.get("power", 1.5)))
        cx = float(params.get("cx", 0.5))
        cy = float(params.get("cy", 0.5))

        arr = np.asarray(base).astype(np.float32)
        xs = ((np.arange(width) + 0.5) / width) - cx
        ys = ((np.arange(height) + 0.5) / height) - cy
        X, Y = np.meshgrid(xs, ys)
        d = np.sqrt(X * X + Y * Y) / np.sqrt(0.5**2 + 0.5**2)

        mask = np.clip(d, 0.0, 1.0) ** power
        mask = (mask * strength)[..., None]

        col = np.array(color, dtype=np.float32)[None, None, :3]
        arr[..., :3] = np.clip(arr[..., :3] * (1 - mask) + col * mask, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGBA")


@help(
    "Apply a draw sub-pipeline (builds an RGBA layer), optionally process that layer with an effect sub-pipeline, "
    "then alpha-composite onto the base.\n"
    "Params: draw_pipeline(str), effect_pipeline(str)\n"
    "Note: sub-pipelines read their params from the same extras dict keyed by block name."
)
@params({
    "draw_pipeline":   {"type": "str", "default": ""},
    "effect_pipeline": {"type": "str", "default": ""},
})
@dataclass
class ApplyMaskedEffectBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        base_img = _ensure_image(img, width, height)
        draw_pipeline = str(params.get("draw_pipeline", ""))
        effect_pipeline = str(params.get("effect_pipeline", ""))

        if not draw_pipeline:
            print("Warning [apply_masked_effect]: 'draw_pipeline' is empty. Skipping.", file=sys.stderr)
            return base_img

        drawn_layer = _run_sub_pipeline(draw_pipeline, width, height, params, initial_img=None)
        if not effect_pipeline:
            return Image.alpha_composite(base_img, drawn_layer)

        effected_layer = _run_sub_pipeline(effect_pipeline, width, height, params, initial_img=drawn_layer)
        return Image.alpha_composite(base_img, effected_layer)


# =============================================================================
# Registration
# =============================================================================

REGISTRY.register("sharpen", SharpenBlock)
REGISTRY.register("unsharpmask", UnsharpMaskBlock)
REGISTRY.register("highpass", HighPassBlock)

REGISTRY.register("edgedetect", EdgeDetectBlock)
REGISTRY.register("sobeledge", SobelEdgeBlock)

REGISTRY.register("chromaticaberration", ChromaticAberrationBlock)
REGISTRY.register("filmgrain", FilmGrainBlock)
REGISTRY.register("bloom", BloomBlock)
REGISTRY.register("motionblur", MotionBlurBlock)

REGISTRY.register("huerotate", HueRotateBlock)
REGISTRY.register("colorbalance", ColorBalanceBlock)
REGISTRY.register("gradientmap", GradientMapBlock)

REGISTRY.register("posterize", PosterizeBlock)
REGISTRY.register("threshold", ThresholdBlock)
REGISTRY.register("pixelate", PixelateBlock)
REGISTRY.register("halftone", HalftoneBlock)
REGISTRY.register("emboss", EmbossBlock)

REGISTRY.register("tiltshift", TiltShiftBlock)
REGISTRY.register("displacement", DisplacementMapBlock)
REGISTRY.register("vignetteplus", VignettePlusBlock)

REGISTRY.register("apply_masked_effect", ApplyMaskedEffectBlock)