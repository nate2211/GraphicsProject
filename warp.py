# warp.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image

from graphics import BaseBlock, REGISTRY, _ensure_image, _norm01


# --------------------------- common helpers ---------------------------

def _meshgrid_wh(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((height, width), dtype=np.float32)
    return xx, yy  # note: return in (x,y) order

def _bilinear_sample(img_arr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """
    img_arr: HxWx4 float32 in [0,255] or [0,1], sample with bilinear interpolation.
    map_x, map_y are float coordinate maps in pixel units.
    Clamps at borders.
    """
    h, w, c = img_arr.shape
    x0 = np.floor(map_x).astype(np.int32)
    y0 = np.floor(map_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    wx = map_x - x0
    wy = map_y - y0
    wa = (1 - wx) * (1 - wy)
    wb = wx * (1 - wy)
    wc = (1 - wx) * wy
    wd = wx * wy

    Ia = img_arr[y0, x0]
    Ib = img_arr[y0, x1]
    Ic = img_arr[y1, x0]
    Id = img_arr[y1, x1]
    out = Ia * wa[..., None] + Ib * wb[..., None] + Ic * wc[..., None] + Id * wd[..., None]
    return out

def _center_and_radius(params: Dict[str, Any], width: int, height: int) -> Tuple[float, float, float]:
    cx = float(params.get("cx", 0.5)) * width
    cy = float(params.get("cy", 0.5)) * height
    # default radius hits near corners from center
    default_r = float(params.get("radius", 0.707)) * max(width, height)
    return cx, cy, max(1.0, default_r)

def _to_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32)

def _from_float(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


# --------------------------- 1) Offset (wrap) ---------------------------

@dataclass
class OffsetWarp(BaseBlock):
    """
    Translate image with wrap-around.
    Params:
      dx_px, dy_px (pixels; can be negative)
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        dx = float(params.get("dx_px", 0.0))
        dy = float(params.get("dy_px", 0.0))

        xx, yy = _meshgrid_wh(width, height)
        map_x = (xx - dx) % width
        map_y = (yy - dy) % height

        arr = _to_float(base)
        out = _bilinear_sample(arr, map_x, map_y)
        return _from_float(out)


# --------------------------- 2) Affine Warp ---------------------------

@dataclass
class AffineWarp(BaseBlock):
    """
    Apply a custom affine transform (2x3 matrix):
      | a b tx |
      | c d ty |
    Params: a,b,c,d,tx,ty (floats)
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        a = float(params.get("a", 1.0))
        b = float(params.get("b", 0.0))
        c = float(params.get("c", 0.0))
        d = float(params.get("d", 1.0))
        tx = float(params.get("tx", 0.0))
        ty = float(params.get("ty", 0.0))

        # We need inverse mapping: for each output (x,y), find input (u,v)
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


# --------------------------- 3) Perspective Warp ---------------------------

@dataclass
class PerspectiveWarp(BaseBlock):
    """
    4-corner perspective warp.
    Params (normalized unless *_px used):
      x0,y0 (top-left), x1,y1 (top-right), x2,y2 (bottom-right), x3,y3 (bottom-left)
      You may pass xN_px/yN_px to use absolute pixels.
    If omitted, defaults to the original rectangle.
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        def gv(name: str, axis: int, default: float) -> float:
            if f"{name}_px" in params:
                return float(params[f"{name}_px"])
            return float(params.get(name, default)) * (width if axis == 0 else height)

        # Destination grid (rect)
        xx, yy = _meshgrid_wh(width, height)

        # Source quad (x0,y0 ... x3,y3)
        x0 = gv("x0", 0, 0.0); y0 = gv("y0", 1, 0.0)
        x1 = gv("x1", 0, 1.0); y1 = gv("y1", 1, 0.0)
        x2 = gv("x2", 0, 1.0); y2 = gv("y2", 1, 1.0)
        x3 = gv("x3", 0, 0.0); y3 = gv("y3", 1, 1.0)

        # Map rectangle coords (xx,yy) in [0..W-1],[0..H-1] -> normalized s,t in [0..1]
        s = xx / max(1.0, width - 1)
        t = yy / max(1.0, height - 1)

        # Bilinear interpolation between quad corners (inverse mapping)
        # P(s,t) = (1-s)(1-t)P0 + s(1-t)P1 + s t P2 + (1-s)t P3
        U = (1 - s) * (1 - t) * x0 + s * (1 - t) * x1 + s * t * x2 + (1 - s) * t * x3
        V = (1 - s) * (1 - t) * y0 + s * (1 - t) * y1 + s * t * y2 + (1 - s) * t * y3

        arr = _to_float(base)
        out = _bilinear_sample(arr, U, V)
        return _from_float(out)


# --------------------------- 4) Swirl Warp ---------------------------

@dataclass
class SwirlWarp(BaseBlock):
    """
    Swirl around center within radius.
    Params: cx, cy, radius, angle_deg (maximum swirl angle at radius)
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        cx, cy, R = _center_and_radius(params, width, height)
        angle_deg = float(params.get("angle_deg", 180.0))

        xx, yy = _meshgrid_wh(width, height)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)

        k = np.clip(r / R, 0.0, 1.0)  # 0 at center -> 1 at radius
        theta2 = theta + np.deg2rad(angle_deg) * (1 - k)  # stronger twist near center

        u = cx + r * np.cos(theta2)
        v = cy + r * np.sin(theta2)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# --------------------------- 5) Ripple Warp (radial) ---------------------------

@dataclass
class RippleWarp(BaseBlock):
    """
    Radial ripple rings from center.
    Params: cx, cy, radius, amp_px, freq (cycles within radius), phase_deg
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        cx, cy, R = _center_and_radius(params, width, height)
        amp = float(params.get("amp_px", 6.0))
        freq = float(params.get("freq", 8.0))
        phase = math.radians(float(params.get("phase_deg", 0.0)))

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


# --------------------------- 6) Wave Warp (horizontal/vertical) ---------------------------

@dataclass
class WaveWarp(BaseBlock):
    """
    Sine wave warp along X and/or Y.
    Params:
      amp_x_px, freq_x (cycles across width), phase_x_deg
      amp_y_px, freq_y (cycles across height), phase_y_deg
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)

        amp_x = float(params.get("amp_x_px", 0.0))
        freq_x = float(params.get("freq_x", 3.0))
        phx = math.radians(float(params.get("phase_x_deg", 0.0)))

        amp_y = float(params.get("amp_y_px", 10.0))
        freq_y = float(params.get("freq_y", 2.0))
        phy = math.radians(float(params.get("phase_y_deg", 0.0)))

        xx, yy = _meshgrid_wh(width, height)
        ox = amp_x * np.sin(2 * math.pi * freq_x * (yy / max(1.0, height - 1)) + phx)
        oy = amp_y * np.sin(2 * math.pi * freq_y * (xx / max(1.0, width - 1)) + phy)

        u = xx + ox
        v = yy + oy

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# --------------------------- 7) Fisheye / Bulge Warp ---------------------------

@dataclass
class FisheyeWarp(BaseBlock):
    """
    Radial bulge (barrel) distortion.
    Params: cx, cy, radius, strength (positive bulge, e.g. 0.5)
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        cx, cy, R = _center_and_radius(params, width, height)
        s = float(params.get("strength", 0.5))  # positive -> bulge

        xx, yy = _meshgrid_wh(width, height)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        k = np.clip(r / R, 0.0, 1.0)

        # radial mapping (simple barrel-like function)
        rn = k ** (1.0 - s) * R
        u = cx + rn * np.cos(theta)
        v = cy + rn * np.sin(theta)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# --------------------------- 8) Pinch Warp ---------------------------

@dataclass
class PinchWarp(BaseBlock):
    """
    Radial pinch (pincushion) distortion.
    Params: cx, cy, radius, strength (positive pinches center, e.g. 0.5)
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        cx, cy, R = _center_and_radius(params, width, height)
        s = float(params.get("strength", 0.5))  # positive -> pinch

        xx, yy = _meshgrid_wh(width, height)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        k = np.clip(r / R, 0.0, 1.0)

        # inverse of bulge
        rn = k ** (1.0 + s) * R
        u = cx + rn * np.cos(theta)
        v = cy + rn * np.sin(theta)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# --------------------------- 9) Spherize Warp ---------------------------

@dataclass
class SpherizeWarp(BaseBlock):
    """
    Project onto a sphere centered at cx,cy within radius.
    Params: cx, cy, radius, amount (0..1)
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        cx, cy, R = _center_and_radius(params, width, height)
        amt = _norm01(float(params.get("amount", 1.0)))

        xx, yy = _meshgrid_wh(width, height)
        dx = (xx - cx) / R
        dy = (yy - cy) / R
        r2 = dx * dx + dy * dy
        mask = (r2 <= 1.0).astype(np.float32)

        z = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))  # sphere z
        # Warp factor pulls coords toward spherical projection
        u = cx + (xx - cx) * ((1 - amt) + amt * z)
        v = cy + (yy - cy) * ((1 - amt) + amt * z)

        # For outside the radius, keep original
        u = u * mask + xx * (1 - mask)
        v = v * mask + yy * (1 - mask)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# --------------------------- 10) RectToPolar ---------------------------

@dataclass
class RectToPolarWarp(BaseBlock):
    """
    Convert rectangular coords to polar (angle around cx,cy; radius outward).
    Creates a polar unwrap (θ on X, r on Y).
    Params: cx, cy, radius
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        cx, cy, R = _center_and_radius(params, width, height)

        # Output image (x: angle 0..2π, y: radius 0..R)
        xx, yy = _meshgrid_wh(width, height)
        theta = (xx / max(1.0, width - 1)) * (2 * math.pi) - math.pi
        r = (yy / max(1.0, height - 1)) * R

        u = cx + r * np.cos(theta)
        v = cy + r * np.sin(theta)

        arr = _to_float(base)
        out = _bilinear_sample(arr, u, v)
        return _from_float(out)


# --------------------------- 11) PolarToRect ---------------------------

@dataclass
class PolarToRectWarp(BaseBlock):
    """
    Map a polar image (θ on X, r on Y) back into a rectangle centered at cx,cy.
    Params: cx, cy, radius
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        cx, cy, R = _center_and_radius(params, width, height)

        xx, yy = _meshgrid_wh(width, height)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = (np.arctan2(dy, dx) + math.pi) / (2 * math.pi)

        src_x = theta * (width - 1)
        src_y = (r / R) * (height - 1)

        arr = _to_float(base)
        out = _bilinear_sample(arr, src_x, src_y)
        return _from_float(out)


# --------------------------- 12) Kaleidoscope ---------------------------

@dataclass
class KaleidoscopeWarp(BaseBlock):
    """
    Kaleidoscope by folding angles around center.
    Params:
      cx, cy, radius, slices (int>=1), rotation_deg
    """
    def process(self, img, width, height, *, params):
        base = _ensure_image(img, width, height)
        cx, cy, R = _center_and_radius(params, width, height)
        slices = max(1, int(params.get("slices", 8)))
        rot = math.radians(float(params.get("rotation_deg", 0.0)))

        xx, yy = _meshgrid_wh(width, height)
        dx = xx - cx
        dy = yy - cy
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx) + rot

        # Fold angle into 0..(2π/slices), then mirror every other wedge
        sector = (2 * math.pi) / slices
        theta_mod = np.mod(theta, sector)
        theta_fold = np.where(theta_mod > sector * 0.5, sector - theta_mod, theta_mod)

        u = cx + r * np.cos(theta_fold - rot)
        v = cy + r * np.sin(theta_fold - rot)

        # Outside radius -> clamp back to original coords
        k = (r <= R).astype(np.float32)
        u = u * k + xx * (1 - k)
        v = v * k + yy * (1 - k)

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
