# warp.py — warp/distortion blocks for your GraphicsEngine
# Requires: numpy, Pillow
#
# Uses your registry decorators:
#   - @help("...")  (single string argument)
#   - @params({ ... })  (structured schema for GUI)
#
# NOTE:
# - Any param that can be blank in the UI MUST be {"nullable": True, "default": None}
# - All params are read from the per-block `params` dict passed into process()

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image

from graphics import BaseBlock, REGISTRY, _ensure_image, _norm01, help, params


# --------------------------- common helpers ---------------------------

def _meshgrid_wh(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((height, width), dtype=np.float32)
    return xx, yy  # (x,y)

def _bilinear_sample(img_arr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """
    img_arr: HxWx4 float32 in [0,255]
    map_x, map_y: float coordinate maps in pixel units
    Clamps at borders.
    """
    h, w, _ = img_arr.shape

    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)

    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wx = map_x - x0
    wy = map_y - y0
    wa = (1.0 - wx) * (1.0 - wy)
    wb = wx * (1.0 - wy)
    wc = (1.0 - wx) * wy
    wd = wx * wy

    Ia = img_arr[y0, x0]
    Ib = img_arr[y0, x1]
    Ic = img_arr[y1, x0]
    Id = img_arr[y1, x1]

    out = Ia * wa[..., None] + Ib * wb[..., None] + Ic * wc[..., None] + Id * wd[..., None]
    return out

def _center_and_radius(p: Dict[str, Any], width: int, height: int) -> Tuple[float, float, float]:
    cx = float(p["cx"]) * width
    cy = float(p["cy"]) * height
    R = float(p["radius"]) * max(width, height)
    return cx, cy, max(1.0, R)

def _to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32)

def _from_float(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


# =============================================================================
# 1) Offset (wrap)
# =============================================================================

@help("offsetwarp — Translate image with wrap-around sampling.")
@params({
    "dx_px": {"type": "float", "default": 0.0, "step": 1.0, "min": -4096.0, "max": 4096.0},
    "dy_px": {"type": "float", "default": 0.0, "step": 1.0, "min": -4096.0, "max": 4096.0},
})
@dataclass
class OffsetWarp(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        dx = float(params["dx_px"])
        dy = float(params["dy_px"])

        xx, yy = _meshgrid_wh(width, height)
        map_x = (xx - dx) % width
        map_y = (yy - dy) % height

        arr = _to_float(base)
        out = _bilinear_sample(arr, map_x, map_y)
        return _from_float(out)


# =============================================================================
# 2) Affine
# =============================================================================

@help("affinewarp — Apply an affine transform using a 2x3 matrix (inverse-mapped sampling).")
@params({
    "a":  {"type": "float", "default": 1.0, "step": 0.01, "min": -10.0, "max": 10.0},
    "b":  {"type": "float", "default": 0.0, "step": 0.01, "min": -10.0, "max": 10.0},
    "c":  {"type": "float", "default": 0.0, "step": 0.01, "min": -10.0, "max": 10.0},
    "d":  {"type": "float", "default": 1.0, "step": 0.01, "min": -10.0, "max": 10.0},
    "tx": {"type": "float", "default": 0.0, "step": 1.0,  "min": -4096.0, "max": 4096.0},
    "ty": {"type": "float", "default": 0.0, "step": 1.0,  "min": -4096.0, "max": 4096.0},
})
@dataclass
class AffineWarp(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        a = float(params["a"])
        b = float(params["b"])
        c = float(params["c"])
        d = float(params["d"])
        tx = float(params["tx"])
        ty = float(params["ty"])

        xx, yy = _meshgrid_wh(width, height)

        det = a * d - b * c
        if abs(det) < 1e-8:
            det = 1e-8

        inv_a = d / det
        inv_b = -b / det
        inv_c = -c / det
        inv_d = a / det

        u = inv_a * (xx - tx) + inv_b * (yy - ty)
        v = inv_c * (xx - tx) + inv_d * (yy - ty)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# =============================================================================
# 3) Perspective (quad)
# =============================================================================

@help("perspectivewarp — Map the output rectangle into a source quad (x0..x3,y0..y3). "
      "Use normalized coords or *_px overrides.")
@params({
    # Normalized quad corners (defaults: identity rectangle)
    "x0": {"type": "float", "default": 0.0, "min": -2.0, "max": 3.0, "step": 0.01},
    "y0": {"type": "float", "default": 0.0, "min": -2.0, "max": 3.0, "step": 0.01},
    "x1": {"type": "float", "default": 1.0, "min": -2.0, "max": 3.0, "step": 0.01},
    "y1": {"type": "float", "default": 0.0, "min": -2.0, "max": 3.0, "step": 0.01},
    "x2": {"type": "float", "default": 1.0, "min": -2.0, "max": 3.0, "step": 0.01},
    "y2": {"type": "float", "default": 1.0, "min": -2.0, "max": 3.0, "step": 0.01},
    "x3": {"type": "float", "default": 0.0, "min": -2.0, "max": 3.0, "step": 0.01},
    "y3": {"type": "float", "default": 1.0, "min": -2.0, "max": 3.0, "step": 0.01},

    # Optional pixel overrides (blank in UI allowed)
    "x0_px": {"type": "float", "default": None, "nullable": True, "step": 1.0},
    "y0_px": {"type": "float", "default": None, "nullable": True, "step": 1.0},
    "x1_px": {"type": "float", "default": None, "nullable": True, "step": 1.0},
    "y1_px": {"type": "float", "default": None, "nullable": True, "step": 1.0},
    "x2_px": {"type": "float", "default": None, "nullable": True, "step": 1.0},
    "y2_px": {"type": "float", "default": None, "nullable": True, "step": 1.0},
    "x3_px": {"type": "float", "default": None, "nullable": True, "step": 1.0},
    "y3_px": {"type": "float", "default": None, "nullable": True, "step": 1.0},
})
@dataclass
class PerspectiveWarp(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        def gv(name: str, axis: int) -> float:
            px = params.get(f"{name}_px", None)
            if px is not None:
                return float(px)
            # normalized
            n = float(params[name])
            return n * (width if axis == 0 else height)

        xx, yy = _meshgrid_wh(width, height)

        x0 = gv("x0", 0); y0 = gv("y0", 1)
        x1 = gv("x1", 0); y1 = gv("y1", 1)
        x2 = gv("x2", 0); y2 = gv("y2", 1)
        x3 = gv("x3", 0); y3 = gv("y3", 1)

        s = xx / max(1.0, width - 1)
        t = yy / max(1.0, height - 1)

        U = (1 - s) * (1 - t) * x0 + s * (1 - t) * x1 + s * t * x2 + (1 - s) * t * x3
        V = (1 - s) * (1 - t) * y0 + s * (1 - t) * y1 + s * t * y2 + (1 - s) * t * y3

        arr = _to_float(base)
        out = _bilinear_sample(arr, U, V)
        return _from_float(out)


# =============================================================================
# 4) Swirl
# =============================================================================

@help("swirlwarp — Swirl distortion around a center point within radius.")
@params({
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "radius": {"type": "float", "default": 0.707, "min": 0.001, "max": 4.0, "step": 0.01},
    "angle_deg": {"type": "float", "default": 180.0, "min": -720.0, "max": 720.0, "step": 1.0},
})
@dataclass
class SwirlWarp(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        cx, cy, R = _center_and_radius(params, width, height)
        ang = math.radians(float(params["angle_deg"]))

        xx, yy = _meshgrid_wh(width, height)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)

        k = np.clip(r / R, 0.0, 1.0)
        theta2 = theta + ang * (1.0 - k)

        u = cx + r * np.cos(theta2)
        v = cy + r * np.sin(theta2)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# =============================================================================
# 5) Ripple (radial)
# =============================================================================

@help("ripplewarp — Radial ripple rings from center.")
@params({
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "radius": {"type": "float", "default": 0.707, "min": 0.001, "max": 4.0, "step": 0.01},
    "amp_px": {"type": "float", "default": 6.0, "min": 0.0, "max": 512.0, "step": 0.5},
    "freq": {"type": "float", "default": 8.0, "min": 0.0, "max": 128.0, "step": 0.1},
    "phase_deg": {"type": "float", "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0},
})
@dataclass
class RippleWarp(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        cx, cy, R = _center_and_radius(params, width, height)
        amp = float(params["amp_px"])
        freq = float(params["freq"])
        phase = math.radians(float(params["phase_deg"]))

        xx, yy = _meshgrid_wh(width, height)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)

        k = np.clip(r / R, 0.0, 1.0)
        offset = amp * np.sin(2 * math.pi * freq * k + phase) * (1.0 - k)

        u = cx + (r + offset) * np.cos(theta)
        v = cy + (r + offset) * np.sin(theta)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# =============================================================================
# 6) Wave (horizontal/vertical)
# =============================================================================

@help("wavewarp — Sine wave warp along X and/or Y.")
@params({
    "amp_x_px": {"type": "float", "default": 0.0, "min": 0.0, "max": 512.0, "step": 0.5},
    "freq_x": {"type": "float", "default": 3.0, "min": 0.0, "max": 64.0, "step": 0.1},
    "phase_x_deg": {"type": "float", "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0},

    "amp_y_px": {"type": "float", "default": 10.0, "min": 0.0, "max": 512.0, "step": 0.5},
    "freq_y": {"type": "float", "default": 2.0, "min": 0.0, "max": 64.0, "step": 0.1},
    "phase_y_deg": {"type": "float", "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0},
})
@dataclass
class WaveWarp(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        amp_x = float(params["amp_x_px"])
        freq_x = float(params["freq_x"])
        phx = math.radians(float(params["phase_x_deg"]))

        amp_y = float(params["amp_y_px"])
        freq_y = float(params["freq_y"])
        phy = math.radians(float(params["phase_y_deg"]))

        xx, yy = _meshgrid_wh(width, height)
        ox = amp_x * np.sin(2 * math.pi * freq_x * (yy / max(1.0, height - 1)) + phx)
        oy = amp_y * np.sin(2 * math.pi * freq_y * (xx / max(1.0, width - 1)) + phy)

        u = xx + ox
        v = yy + oy

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# =============================================================================
# 7) Fisheye / Bulge
# =============================================================================

@help("fisheyewarp — Radial bulge (barrel-like) distortion.")
@params({
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "radius": {"type": "float", "default": 0.707, "min": 0.001, "max": 4.0, "step": 0.01},
    "strength": {"type": "float", "default": 0.5, "min": -2.0, "max": 2.0, "step": 0.01},
})
@dataclass
class FisheyeWarp(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        cx, cy, R = _center_and_radius(params, width, height)
        s = float(params["strength"])

        xx, yy = _meshgrid_wh(width, height)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)

        k = np.clip(r / R, 0.0, 1.0)
        rn = (k ** (1.0 - s)) * R

        u = cx + rn * np.cos(theta)
        v = cy + rn * np.sin(theta)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# =============================================================================
# 8) Pinch
# =============================================================================

@help("pinchwarp — Radial pinch (pincushion-like) distortion.")
@params({
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "radius": {"type": "float", "default": 0.707, "min": 0.001, "max": 4.0, "step": 0.01},
    "strength": {"type": "float", "default": 0.5, "min": -2.0, "max": 2.0, "step": 0.01},
})
@dataclass
class PinchWarp(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        cx, cy, R = _center_and_radius(params, width, height)
        s = float(params["strength"])

        xx, yy = _meshgrid_wh(width, height)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)

        k = np.clip(r / R, 0.0, 1.0)
        rn = (k ** (1.0 + s)) * R

        u = cx + rn * np.cos(theta)
        v = cy + rn * np.sin(theta)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# =============================================================================
# 9) Spherize
# =============================================================================

@help("spherizewarp — Project inside radius onto a sphere (amount 0..1).")
@params({
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "radius": {"type": "float", "default": 0.707, "min": 0.001, "max": 4.0, "step": 0.01},
    "amount": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
})
@dataclass
class SpherizeWarp(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        cx, cy, R = _center_and_radius(params, width, height)
        amt = _norm01(float(params["amount"]))

        xx, yy = _meshgrid_wh(width, height)
        dx = (xx - cx) / R
        dy = (yy - cy) / R
        r2 = dx * dx + dy * dy
        mask = (r2 <= 1.0).astype(np.float32)

        z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
        u = cx + (xx - cx) * ((1.0 - amt) + amt * z)
        v = cy + (yy - cy) * ((1.0 - amt) + amt * z)

        u = u * mask + xx * (1.0 - mask)
        v = v * mask + yy * (1.0 - mask)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# =============================================================================
# 10) Rect -> Polar
# =============================================================================

@help("recttopolarwarp — Unwrap around center: θ on X, radius on Y.")
@params({
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "radius": {"type": "float", "default": 0.707, "min": 0.001, "max": 4.0, "step": 0.01},
})
@dataclass
class RectToPolarWarp(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        cx, cy, R = _center_and_radius(params, width, height)

        xx, yy = _meshgrid_wh(width, height)
        theta = (xx / max(1.0, width - 1)) * (2 * math.pi) - math.pi
        r = (yy / max(1.0, height - 1)) * R

        u = cx + r * np.cos(theta)
        v = cy + r * np.sin(theta)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# =============================================================================
# 11) Polar -> Rect
# =============================================================================

@help("polartorectwarp — Wrap a polar image (θ on X, radius on Y) back into a rectangle.")
@params({
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "radius": {"type": "float", "default": 0.707, "min": 0.001, "max": 4.0, "step": 0.01},
})
@dataclass
class PolarToRectWarp(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        cx, cy, R = _center_and_radius(params, width, height)

        xx, yy = _meshgrid_wh(width, height)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)

        theta = (np.arctan2(dy, dx) + math.pi) / (2.0 * math.pi)

        src_x = theta * (width - 1)
        src_y = (r / R) * (height - 1)

        arr = _to_float(base)
        out = _bilinear_sample(arr, src_x, src_y)
        return _from_float(out)


# =============================================================================
# 12) Kaleidoscope
# =============================================================================

@help("kaleidoscopewarp — Kaleidoscope effect by angular folding around center.")
@params({
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "radius": {"type": "float", "default": 0.707, "min": 0.001, "max": 4.0, "step": 0.01},
    "slices": {"type": "int", "default": 8, "min": 1, "max": 64, "step": 1},
    "rotation_deg": {"type": "float", "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0},
})
@dataclass
class KaleidoscopeWarp(BaseBlock):
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        cx, cy, R = _center_and_radius(params, width, height)
        slices = max(1, int(params["slices"]))
        rot = math.radians(float(params["rotation_deg"]))

        xx, yy = _meshgrid_wh(width, height)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)

        theta = np.arctan2(dy, dx) + rot

        sector = (2.0 * math.pi) / slices
        theta_mod = np.mod(theta, sector)
        theta_fold = np.where(theta_mod > sector * 0.5, sector - theta_mod, theta_mod)

        u = cx + r * np.cos(theta_fold - rot)
        v = cy + r * np.sin(theta_fold - rot)

        k = (r <= R).astype(np.float32)
        u = u * k + xx * (1.0 - k)
        v = v * k + yy * (1.0 - k)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# --------------------------- registration ---------------------------

REGISTRY.register("offsetwarp", OffsetWarp)
REGISTRY.register("affinewarp", AffineWarp)
REGISTRY.register("perspectivewarp", PerspectiveWarp)
REGISTRY.register("swirlwarp", SwirlWarp)
REGISTRY.register("ripplewarp", RippleWarp)
REGISTRY.register("wavewarp", WaveWarp)
REGISTRY.register("fisheyewarp", FisheyeWarp)
REGISTRY.register("pinchwarp", PinchWarp)
REGISTRY.register("spherizewarp", SpherizeWarp)
REGISTRY.register("recttopolarwarp", RectToPolarWarp)
REGISTRY.register("polartorectwarp", PolarToRectWarp)
REGISTRY.register("kaleidoscopewarp", KaleidoscopeWarp)
