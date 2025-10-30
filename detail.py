# detail.py
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw

# Import from your existing graphics module
from graphics import BaseBlock, REGISTRY, _ensure_image, _parse_color, _norm01


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


# ---------------------------------------------------------------------------
# Core detail / stylistic blocks
# ---------------------------------------------------------------------------

# --- Sharpen Block ---
@dataclass
class SharpenBlock(BaseBlock):
    """Applies a simple sharpening effect (PIL SHARPEN)."""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        return img.filter(ImageFilter.SHARPEN)


# --- Unsharp Mask ---
@dataclass
class UnsharpMaskBlock(BaseBlock):
    """
    Unsharp mask sharpening.
    Params: radius (px), percent (amount 0..500), threshold (0..255)
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        radius = float(params.get("radius", 2.0))
        percent = int(params.get("percent", 150))
        threshold = int(params.get("threshold", 3))
        return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


# --- High-Pass Sharpen (blend highpass over image) ---
@dataclass
class HighPassBlock(BaseBlock):
    """
    High-pass filter for crisp detail.
    Params: radius (px), strength (0..2), preserve_color (bool)
    """
    def process(self, img, width, height, *, params):
        src = _ensure_image(img, width, height)
        radius = float(params.get("radius", 4.0))
        strength = float(params.get("strength", 1.0))
        preserve_color = bool(params.get("preserve_color", True))

        # Work in linear-ish float
        arr = np.asarray(src).astype(np.float32) / 255.0
        blur = np.asarray(src.filter(ImageFilter.GaussianBlur(radius))).astype(np.float32) / 255.0
        high = arr[..., :3] - blur[..., :3]  # detail
        if not preserve_color:
            # Grayscale high-pass applied equally
            lum = (0.299 * high[..., 0] + 0.587 * high[..., 1] + 0.114 * high[..., 2])[..., None]
            high = np.repeat(lum, 3, axis=2)

        out_rgb = np.clip(arr[..., :3] + strength * high, 0.0, 1.0)
        out_a = arr[..., 3:4] if arr.shape[2] == 4 else np.ones_like(out_rgb[..., :1])
        out = np.concatenate([out_rgb, out_a], axis=2)
        return Image.fromarray((out * 255.0).astype(np.uint8), "RGBA")


# --- Edge Detect (stylized, alpha composited) ---
@dataclass
class EdgeDetectBlock(BaseBlock):
    """Detect edges and overlay black edges over the source with alpha."""
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        img_gray = base.convert("L")
        edges = img_gray.filter(ImageFilter.FIND_EDGES)
        edges_inverted = ImageOps.invert(edges).convert("L")  # brighter = weaker edge
        # Build RGBA: alpha where edge is strong (dark in 'edges')
        alpha = ImageOps.invert(edges)  # strong edge -> high alpha
        edge_rgba = Image.merge("RGBA", (Image.new("L", base.size, 0),
                                         Image.new("L", base.size, 0),
                                         Image.new("L", base.size, 0),
                                         alpha))
        return Image.alpha_composite(base, edge_rgba)


# --- Sobel Edge (cleaner than FIND_EDGES) ---
@dataclass
class SobelEdgeBlock(BaseBlock):
    """
    Sobel edge magnitude overlay.
    Params: strength (0..1), invert (bool)
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        strength = _norm01(float(params.get("strength", 0.8)))
        invert = bool(params.get("invert", False))

        g = np.asarray(base.convert("L")).astype(np.float32) / 255.0
        # Sobel kernels
        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float32)

        def conv2(a, k):
            from scipy.signal import convolve2d as _c  # optional dependency; fallback below if missing
            return _c(a, k, mode="same", boundary="symm")

        try:
            gx = conv2(g, Kx)
            gy = conv2(g, Ky)
        except Exception:
            # Simple fallback convolution
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
            gx = simple_conv(g, Kx)
            gy = simple_conv(g, Ky)

        mag = np.sqrt(gx * gx + gy * gy)
        mag = mag / (mag.max() + 1e-9)
        if invert:
            mag = 1.0 - mag
        edge_alpha = (mag * 255.0 * strength).astype(np.uint8)
        edge_rgba = Image.merge("RGBA", (Image.new("L", base.size, 0),
                                         Image.new("L", base.size, 0),
                                         Image.new("L", base.size, 0),
                                         Image.fromarray(edge_alpha)))
        return Image.alpha_composite(base, edge_rgba)


# --- Chromatic Aberration ---
@dataclass
class ChromaticAberrationBlock(BaseBlock):
    """
    Simulates lens chromatic aberration by shifting color channels.
    Params: amount_px (px), angle_deg (0 deg = right)
    """
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


# --- Film Grain ---
@dataclass
class FilmGrainBlock(BaseBlock):
    """
    Adds monochrome noise film grain.
    Params: amount (0..1), colored (bool)
    """
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


# --- Bloom / Glow ---
@dataclass
class BloomBlock(BaseBlock):
    """
    Adds a soft glow by thresholding brights, blurring, and adding back.
    Params: threshold (0..1), radius (px), strength (0..3)
    """
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
        # Convert back to image, blur, and add
        bright_img = Image.fromarray((np.clip(brights, 0, 1) * 255.0).astype(np.uint8), "RGBA")
        blurred = bright_img.filter(ImageFilter.GaussianBlur(radius))
        add = np.asarray(blurred).astype(np.float32) / 255.0
        out = np.clip(arr + strength * add, 0.0, 1.0)
        return Image.fromarray((out * 255.0).astype(np.uint8), "RGBA")


# --- Motion Blur (multi-shift average) ---
@dataclass
class MotionBlurBlock(BaseBlock):
    """
    Directional motion blur by averaging shifted copies.
    Params: length (px), angle_deg, samples (int)
    """
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


# --- Hue Rotate ---
@dataclass
class HueRotateBlock(BaseBlock):
    """
    Rotate hue in HSV space.
    Params: degrees (-360..360), sat_mult (S scale), val_mult (V scale)
    """
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
        # Hue calculation
        rc = (maxc - r) / (maxc - minc + 1e-9)
        gc = (maxc - g) / (maxc - minc + 1e-9)
        bc = (maxc - b) / (maxc - minc + 1e-9)
        h = np.zeros_like(maxc)
        h = np.where(maxc == r, (bc - gc), h)
        h = np.where(maxc == g, 2.0 + (rc - bc), h)
        h = np.where(maxc == b, 4.0 + (gc - rc), h)
        h = (h / 6.0) % 1.0

        # Rotate and scale
        h = (h + deg / 360.0) % 1.0
        s = np.clip(s * sat_mult, 0.0, 1.0)
        v = np.clip(v * val_mult, 0.0, 1.0)

        # HSV -> RGB
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


# --- Color Balance (lift/gamma/gain style) ---
@dataclass
class ColorBalanceBlock(BaseBlock):
    """
    Per-channel lift/gamma/gain.
    Params: lift_r/g/b (-1..1), gamma_r/g/b (0.1..3), gain_r/g/b (0..3)
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        arr = np.asarray(base).astype(np.float32) / 255.0
        rgb = arr[..., :3]
        a = arr[..., 3:4]

        def g(k, d, default):
            return float(params.get(k, default)) if isinstance(default, float) else default

        lift = np.array([g("lift_r", params, 0.0),
                         g("lift_g", params, 0.0),
                         g("lift_b", params, 0.0)], dtype=np.float32)
        gamma = np.array([max(0.1, g("gamma_r", params, 1.0)),
                          max(0.1, g("gamma_g", params, 1.0)),
                          max(0.1, g("gamma_b", params, 1.0))], dtype=np.float32)
        gain = np.array([max(0.0, g("gain_r", params, 1.0)),
                         max(0.0, g("gain_g", params, 1.0)),
                         max(0.0, g("gain_b", params, 1.0))], dtype=np.float32)

        rgb = (rgb + lift[None, None, :])
        rgb = np.clip(rgb, 0.0, 1.0)
        rgb = np.power(rgb, 1.0 / gamma[None, None, :])
        rgb = np.clip(rgb * gain[None, None, :], 0.0, 1.0)

        out = np.concatenate([rgb, a], axis=2)
        return Image.fromarray((out * 255.0).astype(np.uint8), "RGBA")


# --- Gradient Map (luminance -> two colors) ---
@dataclass
class GradientMapBlock(BaseBlock):
    """
    Maps luminance to a two-color gradient.
    Params: low_color, high_color
    """
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


# --- Posterize ---
@dataclass
class PosterizeBlock(BaseBlock):
    """
    Posterize image color depth.
    Params: bits (1..8)
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        bits = max(1, min(8, int(params.get("bits", 4))))
        # Separate alpha
        if base.mode == "RGBA":
            rgb = base.convert("RGB")
            a = base.getchannel("A")
            poster = ImageOps.posterize(rgb, bits).convert("RGBA")
            return Image.merge("RGBA", (*poster.split()[:3], a))
        else:
            return ImageOps.posterize(base, bits)


# --- Threshold (binary) ---
@dataclass
class ThresholdBlock(BaseBlock):
    """
    Binary threshold on luminance.
    Params: thresh (0..1), low_color, high_color, keep_alpha (bool)
    """
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


# --- Pixelate ---
@dataclass
class PixelateBlock(BaseBlock):
    """
    Pixelates the image by down/up scaling.
    Params: factor (>1), method (nearest|box)
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        factor = max(1.0, float(params.get("factor", 8.0)))
        method = str(params.get("method", "nearest")).lower()
        down_w = max(1, int(round(width / factor)))
        down_h = max(1, int(round(height / factor)))
        down = base.resize((down_w, down_h), Image.Resampling.BOX)
        up_mode = Image.Resampling.NEAREST if method == "nearest" else Image.Resampling.BOX
        return down.resize((width, height), up_mode)


# --- Halftone (simple dot pattern over luminance) ---
@dataclass
class HalftoneBlock(BaseBlock):
    """
    Dot halftone overlay.
    Params: cell (px), strength (0..1), angle_deg
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        cell = max(2, int(params.get("cell", 8)))
        strength = _norm01(float(params.get("strength", 0.75)))
        angle_deg = float(params.get("angle_deg", 45.0))

        # Rotate a copy to align grid
        rot = base.rotate(angle_deg, expand=True, resample=Image.Resampling.BILINEAR, fillcolor=(0,0,0,0))
        rw, rh = rot.size
        arr = np.asarray(rot.convert("RGBA")).astype(np.float32) / 255.0
        lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])

        # Build dot canvas
        dots = Image.new("L", (rw, rh), 0)
        draw = ImageDraw.Draw(dots)
        for y in range(0, rh, cell):
            for x in range(0, rw, cell):
                l = float(lum[min(rh - 1, y), min(rw - 1, x)])
                r = (1.0 - l) * (cell * 0.5)
                if r > 0.5:
                    draw.ellipse([x - r, y - r, x + r, y + r], fill=int(255 * strength))
        # Rotate dots back and crop
        dots_back = dots.rotate(-angle_deg, expand=False, resample=Image.Resampling.BILINEAR)
        dots_back = dots_back.resize((width, height), Image.Resampling.BILINEAR)

        overlay = Image.merge("RGBA", (Image.new("L", (width, height), 0),
                                       Image.new("L", (width, height), 0),
                                       Image.new("L", (width, height), 0),
                                       dots_back))
        return Image.alpha_composite(base, overlay)


# --- Emboss (stylized relief) ---
@dataclass
class EmbossBlock(BaseBlock):
    """Emboss the image (PIL EMBOSS), optional strength via blend."""
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        strength = _norm01(float(params.get("strength", 1.0)))
        emb = base.filter(ImageFilter.EMBOSS)
        if strength >= 0.999:
            return emb
        return Image.blend(base, emb, strength)


# --- Tilt-Shift (focus band with blur outside) ---
@dataclass
class TiltShiftBlock(BaseBlock):
    """
    Keeps a horizontal focus band sharp; blurs outside with feather.
    Params: focus_y (0..1), band (0..1 height), feather (0..1), radius (px)
    """
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
        # 1 inside band, 0 far away, feathered edges
        m = np.clip((half - np.abs(y - center)) / max(half, 1e-6), 0.0, 1.0)
        # Feather with smoothstep
        m = m * m * (3 - 2 * m)
        m = np.clip(m + feather * (1 - m), 0.0, 1.0)
        m = np.repeat(m, width, axis=1)[..., None]

        a_base = np.asarray(base).astype(np.float32)
        a_blur = np.asarray(blurred).astype(np.float32)
        out = a_base * m + a_blur * (1 - m)
        return Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGBA")


# --- Displacement Map (warp by another image/layer) ---
@dataclass
class DisplacementMapBlock(BaseBlock):
    """
    Displace pixels by a provided displacement image (D).
    Params:
      disp_path (optional), scale_x, scale_y, channel ('r','g','b','a','luma')
      If no disp_path: uses current image's luminance as displacement.
    """
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
            idx = {"r":0, "g":1, "b":2, "a":3}.get(channel, 0)
            disp = darr[..., idx]
        disp = (disp * 2.0 - 1.0)  # -1..1

        yy, xx = np.indices((height, width), dtype=np.float32)
        map_x = np.clip(xx + disp * scale_x, 0, width - 1)
        map_y = np.clip(yy + disp * scale_y, 0, height - 1)

        src = np.asarray(base).astype(np.float32)
        # Bilinear sample
        x0 = np.floor(map_x).astype(np.int32); x1 = np.clip(x0 + 1, 0, width - 1)
        y0 = np.floor(map_y).astype(np.int32); y1 = np.clip(y0 + 1, 0, height - 1)
        wx = map_x - x0; wy = map_y - y0
        wa = (1 - wx) * (1 - wy)
        wb = wx * (1 - wy)
        wc = (1 - wx) * wy
        wd = wx * wy

        out = (src[y0, x0] * wa[..., None] +
               src[y0, x1] * wb[..., None] +
               src[y1, x0] * wc[..., None] +
               src[y1, x1] * wd[..., None])
        return Image.fromarray(out.clip(0, 255).astype(np.uint8), "RGBA")


# --- Vignette+ (color + strength curve) ---
@dataclass
class VignettePlusBlock(BaseBlock):
    """
    Colored vignette with curve power.
    Params: strength (0..1), color, power (>0), cx, cy
    """
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

        col = (np.array(color, dtype=np.float32)[None, None, :3])
        arr[..., :3] = np.clip(arr[..., :3] * (1 - mask) + col * mask, 0, 255)
        return Image.fromarray(arr.astype(np.uint8), "RGBA")


# --- Apply Masked Effect (draw->effect->composite) ---
@dataclass
class ApplyMaskedEffectBlock(BaseBlock):
    """
    Draw shapes via 'draw_pipeline', optionally process with 'effect_pipeline',
    then composite onto the base image.

    Params:
      draw_pipeline:   e.g. "drawcircle|drawrect"
      effect_pipeline: e.g. "gaussianblur|bloom"
      (All parameters for sub-pipelines are passed through.)
    """
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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
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
