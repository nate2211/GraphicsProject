# image.py
from __future__ import annotations

import io
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
from urllib.parse import urlparse, unquote

import numpy as np
import requests
from PIL import Image, ImageEnhance, ImageOps

# ------------------------------------------------------------
# Import engine helpers
# ------------------------------------------------------------
try:
    from graphics import (
        REGISTRY,
        BaseBlock,
        params,
        help as helptext,
        _ensure_image,
        _parse_color,
        _norm01,
    )
except Exception as e:
    raise ImportError(
        "image.py must be able to import from graphics.py "
        "(REGISTRY, BaseBlock, params, help, _ensure_image, _parse_color, _norm01). "
        f"Original error: {e}"
    )

# ------------------------------------------------------------
# Path + URL handling (quotes, file:// URIs, Windows quirks)
# ------------------------------------------------------------
def _sanitize_path(p: Any) -> str:
    """
    Accepts:
      - r'C:\\path\\file.png'
      - '"C:\\path\\file.png"' or "'C:\\path\\file.png'"
      - file:///C:/path/file.png
      - file:///C:/path/My%20File.png
    Returns a normal filesystem path string.
    """
    if p is None:
        return ""
    s = str(p).strip()

    # Strip wrapping quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()

    # Common drag/drop yields file:// URI
    if s.lower().startswith("file:"):
        u = urlparse(s)
        s = unquote(u.path)

        # Windows: /C:/... -> C:/...
        if os.name == "nt" and len(s) >= 3 and s[0] == "/" and s[2] == ":":
            s = s[1:]

    # Normalize slashes a bit (harmless on Windows)
    s = s.replace("/", os.sep)
    return s


def _is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme.lower() in ("http", "https")
    except Exception:
        return False


# ------------------------------------------------------------
# Image load + cache (file + URL)
# ------------------------------------------------------------
_IMAGE_CACHE_LOCK = threading.Lock()
_IMAGE_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}
# key: (source_key, max_dim) -> {"img": PIL.Image RGBA, "w": int, "h": int, "src": str}

def _download_bytes(url: str, timeout: float = 15.0) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GraphicsEngine/1.0; +https://example.invalid)"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.content


def _load_image_any(src: Any, max_dim: int = 0) -> Dict[str, Any]:
    """
    Load + cache an image from:
      - local path (quoted ok)
      - file:// URI
      - http(s) URL
    Returns dict {"img": RGBA PIL.Image, "w": int, "h": int, "src": str}
    """
    if src is None:
        raise ValueError("image source is empty.")

    s = str(src).strip()
    if not s:
        raise ValueError("image source is empty.")

    # Strip wrapping quotes early
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()

    max_dim = int(max_dim or 0)
    max_dim = max(0, min(max_dim, 16384))

    # Build a stable cache key
    cache_key = (s, max_dim)
    with _IMAGE_CACHE_LOCK:
        cached = _IMAGE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    # Resolve source
    if s.lower().startswith("file:") or (not _is_url(s) and os.path.exists(_sanitize_path(s))):
        p = _sanitize_path(s)
        ap = os.path.abspath(os.path.expanduser(p))
        if not os.path.exists(ap):
            raise FileNotFoundError(f"Image file not found: {ap}")
        img = Image.open(ap)
        src_used = ap
    elif _is_url(s):
        data = _download_bytes(s)
        img = Image.open(io.BytesIO(data))
        src_used = s
    else:
        # last attempt: treat as filesystem path
        p = _sanitize_path(s)
        ap = os.path.abspath(os.path.expanduser(p))
        if not os.path.exists(ap):
            raise FileNotFoundError(f"Image source not found (not a URL, not a file): {ap}")
        img = Image.open(ap)
        src_used = ap

    # Normalize to RGBA
    try:
        img = img.convert("RGBA")
    except Exception:
        img = ImageOps.exif_transpose(img).convert("RGBA")

    # Optional pre-resize for memory safety / speed
    if max_dim > 0:
        w, h = img.size
        m = max(w, h)
        if m > max_dim:
            scale = float(max_dim) / float(m)
            nw = max(1, int(round(w * scale)))
            nh = max(1, int(round(h * scale)))
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)

    payload = {"img": img, "w": img.size[0], "h": img.size[1], "src": src_used}
    with _IMAGE_CACHE_LOCK:
        _IMAGE_CACHE[cache_key] = payload
    return payload


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------
def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)

def _clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        iv = int(v)
    except Exception:
        iv = int(default)
    return max(lo, min(hi, iv))

def _fit_rect(src_w: int, src_h: int, dst_w: int, dst_h: int, mode: str) -> Tuple[int, int]:
    """
    Return new (w,h) for src to fit/cover dst based on mode:
      - 'contain' keeps full image visible (letterbox)
      - 'cover' fills dst (may crop)
      - 'stretch' force to dst
      - 'none' no scaling
    """
    mode = (mode or "contain").strip().lower()
    if mode == "none":
        return src_w, src_h
    if mode == "stretch":
        return dst_w, dst_h

    if src_w <= 0 or src_h <= 0:
        return dst_w, dst_h

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    s = min(sx, sy) if mode == "contain" else max(sx, sy)
    nw = max(1, int(round(src_w * s)))
    nh = max(1, int(round(src_h * s)))
    return nw, nh


# ------------------------------------------------------------
# Blocks
# ------------------------------------------------------------
@helptext(
    """
    LoadImage: load an image from a local file or an http(s) URL and composite it onto the current frame.

    Sources:
      - path: "C:\\Users\\...\\img.png"
      - file:// URIs from drag/drop
      - url: https://...

    Placement:
      - x/y are normalized (0..1) anchor position
      - anchor: 'center'|'topleft'|'topright'|'bottomleft'|'bottomright'

    Scaling:
      - fit: 'contain'|'cover'|'stretch'|'none'
      - max_dim: optional pre-resize of the loaded image for speed/memory
    """
)
@params({
    "src": {"type": "str", "default": None, "nullable": True, "hint": "file path, file:// URI, or https:// URL"},
    "max_dim": {"type": "int", "default": 0, "min": 0, "max": 16384, "step": 64},

    "fit": {"type": "str", "default": "contain"},
    "w": {"type": "float", "default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01, "hint": "relative to canvas width"},
    "h": {"type": "float", "default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01, "hint": "relative to canvas height"},

    "x": {"type": "float", "default": 0.5, "min": -2.0, "max": 3.0, "step": 0.01},
    "y": {"type": "float", "default": 0.5, "min": -2.0, "max": 3.0, "step": 0.01},
    "anchor": {"type": "str", "default": "center"},

    "opacity": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "blend": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},

    "rotate_deg": {"type": "float", "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0},
    "flip_x": {"type": "bool", "default": False},
    "flip_y": {"type": "bool", "default": False},

    "brightness": {"type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
    "contrast": {"type": "float", "default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
})
@dataclass
class LoadImage(BaseBlock):
    def process(self, img, width, height, *, params, ctx=None, **kwargs):
        base = _ensure_image(img, width, height)

        src = params.get("src", None)
        if not src:
            return base

        max_dim = _clamp_int(params.get("max_dim", 0), 0, 16384, 0)

        try:
            payload = _load_image_any(src, max_dim=max_dim)
        except Exception as e:
            print(f"[image] LoadImage load failed: {e}", file=sys.stderr)
            return base

        im = payload["img"]

        # optional transforms
        if bool(params.get("flip_x", False)):
            im = ImageOps.mirror(im)
        if bool(params.get("flip_y", False)):
            im = ImageOps.flip(im)

        rot = _safe_float(params.get("rotate_deg", 0.0), 0.0)
        if abs(rot) > 1e-6:
            im = im.rotate(rot, resample=Image.Resampling.BICUBIC, expand=True)

        br = _safe_float(params.get("brightness", 1.0), 1.0)
        if abs(br - 1.0) > 1e-6:
            im = ImageEnhance.Brightness(im).enhance(br)

        ct = _safe_float(params.get("contrast", 1.0), 1.0)
        if abs(ct - 1.0) > 1e-6:
            im = ImageEnhance.Contrast(im).enhance(ct)

        # destination box size
        dst_w = max(1, int(round(_safe_float(params.get("w", 1.0), 1.0) * width)))
        dst_h = max(1, int(round(_safe_float(params.get("h", 1.0), 1.0) * height)))

        fit = str(params.get("fit", "contain") or "contain").strip().lower()
        nw, nh = _fit_rect(im.size[0], im.size[1], dst_w, dst_h, fit)

        if (nw, nh) != im.size:
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)

        # If cover, crop to exact dst_w/dst_h
        if fit == "cover":
            # center crop
            cx = nw // 2
            cy = nh // 2
            x0 = max(0, cx - dst_w // 2)
            y0 = max(0, cy - dst_h // 2)
            x1 = min(nw, x0 + dst_w)
            y1 = min(nh, y0 + dst_h)
            im = im.crop((x0, y0, x1, y1))
            # ensure exact
            if im.size != (dst_w, dst_h):
                im = im.resize((dst_w, dst_h), Image.Resampling.LANCZOS)
        elif fit in ("stretch",):
            if im.size != (dst_w, dst_h):
                im = im.resize((dst_w, dst_h), Image.Resampling.LANCZOS)
        elif fit == "contain":
            # letterbox into dst box
            canvas = Image.new("RGBA", (dst_w, dst_h), (0, 0, 0, 0))
            ox = (dst_w - im.size[0]) // 2
            oy = (dst_h - im.size[1]) // 2
            canvas.alpha_composite(im, (ox, oy))
            im = canvas
        else:
            # none: use im as-is
            pass

        # placement
        x = _safe_float(params.get("x", 0.5), 0.5) * width
        y = _safe_float(params.get("y", 0.5), 0.5) * height
        anchor = str(params.get("anchor", "center") or "center").strip().lower()

        iw, ih = im.size
        if anchor == "topleft":
            px, py = int(round(x)), int(round(y))
        elif anchor == "topright":
            px, py = int(round(x - iw)), int(round(y))
        elif anchor == "bottomleft":
            px, py = int(round(x)), int(round(y - ih))
        elif anchor == "bottomright":
            px, py = int(round(x - iw)), int(round(y - ih))
        else:  # center
            px, py = int(round(x - iw / 2)), int(round(y - ih / 2))

        opacity = _norm01(_safe_float(params.get("opacity", 1.0), 1.0))
        blend = _norm01(_safe_float(params.get("blend", 1.0), 1.0))

        if opacity < 0.999:
            # scale alpha channel
            arr = np.asarray(im).astype(np.float32)
            arr[..., 3] = np.clip(arr[..., 3] * opacity, 0, 255)
            im = Image.fromarray(arr.astype(np.uint8), "RGBA")

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        overlay.alpha_composite(im, (px, py))

        if blend >= 0.999:
            return Image.alpha_composite(base, overlay)
        if blend <= 1e-6:
            return base

        ov = np.asarray(overlay).astype(np.float32)
        bs = np.asarray(base).astype(np.float32)

        a = (ov[..., 3:4] / 255.0) * blend
        out = bs.copy()
        out[..., :3] = bs[..., :3] * (1.0 - a) + ov[..., :3] * a
        out[..., 3:4] = np.clip(bs[..., 3:4] + ov[..., 3:4] * blend, 0, 255)
        return Image.fromarray(out.astype(np.uint8), "RGBA")


# ------------------------------------------------------------
# Registration
# ------------------------------------------------------------
REGISTRY.register("loadimage", LoadImage)
