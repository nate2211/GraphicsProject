# threed.py — lightweight 3D wireframes for your GraphicsEngine
# Requires: numpy, Pillow
# Works with your graphics.py registry/engine.
#
# Blocks:
#   - drawgrid3d         : Infinite-style ground/grid plane (clipped to image) in 3D
#   - drawicosahedron    : Regular icosahedron wireframe
#   - drawtoruswire      : Torus wireframe (rings × segments)
#   - drawfrustum        : Camera frustum visualization
#   - drawspiral3d       : Helix/spiral polyline
#   - drawcylinderwire   : Cylinder wireframe (caps + verticals)
#   - drawplane          : Simple oriented plane rectangle
#
# Example:
#   python main.py run --out demo.png --pipeline "solidcolor|drawgrid3d|drawaxes3d|drawcube|drawicosahedron|drawtoruswire|drawfrustum|drawspiral3d|drawcylinderwire|drawplane"

from __future__ import annotations

import ast
import math
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from collections.abc import Sequence
import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Pull base helpers/registry from graphics.py (with safe fallbacks)
# ---------------------------------------------------------------------------
try:
    from graphics import BaseBlock, REGISTRY, _ensure_image, _parse_color, help, params
except Exception:
    # Fallback stubs (lets file import without graphics.py for linting)
    class BaseBlock:  # type: ignore
        pass

    class _DummyReg:  # type: ignore
        def register(self, *_a, **_k): ...
    REGISTRY = _DummyReg()  # type: ignore

    def _ensure_image(img, w, h):
        return Image.new("RGBA", (w, h))

    def _parse_color(val: Any, default: Optional[Tuple[int, int, int, int]]):
        return default

    def help(_s: str):  # type: ignore
        def deco(cls): return cls
        return deco

    def params(_d: Dict[str, Any]):  # type: ignore
        def deco(cls): return cls
        return deco


# ---------------- math helpers ----------------

def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float32)

def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)

def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]], dtype=np.float32)

def _compose_rot(ax: float, ay: float, az: float) -> np.ndarray:
    # Y then X then Z (tweak if you prefer a different order)
    return _rot_z(az) @ _rot_x(ax) @ _rot_y(ay)


def _project_points(
    pts: np.ndarray,
    mode: str,
    width: int,
    height: int,
    *,
    cx: float,
    cy: float,
    fov_deg: float = 60.0,
    z_offset: float = 2.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects Nx3 into Nx2.
    Returns (xy_pixels Nx2, z_values Nx1 for depth shading).
    mode: 'orthographic' | 'perspective' | 'isometric'
    """
    mode_l = (mode or "orthographic").strip().lower()
    pts = np.asarray(pts, dtype=np.float32)

    if mode_l == "isometric":
        # Classic iso: rotate ~35.264° about X and 45° about Z
        R_iso = _rot_x(math.radians(35.264)) @ _rot_z(math.radians(45))
        q = pts @ R_iso.T
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        sx = min(width, height) * 0.75
        cx_pix, cy_pix = cx * width, cy * height
        xy = np.stack([cx_pix + x * sx, cy_pix - y * sx], axis=1)
        return xy, z

    if mode_l == "perspective":
        # Simple pinhole camera on +Z axis looking toward origin
        f = 0.5 * min(width, height) / math.tan(max(1e-3, math.radians(fov_deg) / 2.0))
        z = pts[:, 2] + z_offset
        z = np.maximum(z, 1e-3)
        x = pts[:, 0] * (f / z)
        y = pts[:, 1] * (f / z)
        cx_pix, cy_pix = cx * width, cy * height
        xy = np.stack([cx_pix + x, cy_pix - y], axis=1)
        return xy, (pts[:, 2])

    # Orthographic
    sx = min(width, height) * 0.75
    cx_pix, cy_pix = cx * width, cy * height
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    xy = np.stack([cx_pix + x * sx, cy_pix - y * sx], axis=1)
    return xy, z


def _depth_tint(color_rgba: Tuple[int, int, int, int], zvals: np.ndarray, k: float = 0.5) -> List[Tuple[int, int, int, int]]:
    """
    Darken color based on depth percentile (farther -> darker).
    k controls strength (0..1).
    """
    if zvals.size == 0:
        return [color_rgba]
    z = zvals.astype(np.float32)
    z_min, z_max = float(np.min(z)), float(np.max(z))
    span = max(1e-6, z_max - z_min)
    t = (z - z_min) / span  # 0 near .. 1 far
    out: List[Tuple[int, int, int, int]] = []
    r, g, b, a = color_rgba
    for ti in t:
        d = 1.0 - k * float(ti)
        out.append((int(r * d), int(g * d), int(b * d), a))
    return out


# ---------------- primitive generators ----------------

def _unit_cube() -> np.ndarray:
    # centered at origin, side length 2
    vs = [
        (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1),
        (-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1),
    ]
    return np.array(vs, dtype=np.float32)

_CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

def _square_pyramid() -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    # base square (-1..1) on z=-1, apex at (0,0,1)
    vs = np.array([
        (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1),
        ( 0,  0,  1),
    ], dtype=np.float32)
    edges = [(0,1),(1,2),(2,3),(3,0),(0,4),(1,4),(2,4),(3,4)]
    return vs, edges

def _axis_lines(length: float = 1.0) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int,int,int,int]]]:
    # X (red), Y (green), Z (blue)
    L = float(length)
    verts = np.array([
        (0,0,0), (L,0,0),
        (0,0,0), (0,L,0),
        (0,0,0), (0,0,L)
    ], dtype=np.float32)
    edges = [(0,1),(2,3),(4,5)]
    colors = [(255,64,64,255),(64,255,64,255),(64,128,255,255)]
    return verts, edges, colors

def _icosahedron() -> Tuple[np.ndarray, List[Tuple[int,int]]]:
    # Regular icosahedron (edge graph)
    phi = (1 + math.sqrt(5)) / 2.0
    a, b = 1.0, 1.0 / phi
    verts = np.array([
        (-a,  b,  0), ( a,  b,  0), (-a, -b,  0), ( a, -b,  0),
        ( 0, -a,  b), ( 0,  a,  b), ( 0, -a, -b), ( 0,  a, -b),
        ( b,  0, -a), ( b,  0,  a), (-b,  0, -a), (-b,  0,  a)
    ], dtype=np.float32)
    edges = [
        (0,1),(0,5),(0,7),(0,10),(0,11),
        (1,5),(1,7),(1,8),(1,9),
        (2,3),(2,4),(2,6),(2,10),(2,11),
        (3,4),(3,6),(3,8),(3,9),
        (4,5),(4,9),(4,11),
        (5,9),(5,11),
        (6,7),(6,8),(6,10),
        (7,8),(7,10),
        (8,9),(10,11)
    ]
    return verts, edges


# ---------------- robust param coercion helpers ----------------

def _coerce_to_float(val: Any, default_val: float) -> float:
    if val is None:
        return float(default_val)

    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, str):
        # allow numbers or tuple strings
        try:
            parsed = ast.literal_eval(val)
            val = parsed
        except Exception:
            try:
                return float(val)
            except Exception:
                print(f"[threed] WARN: could not parse '{val}' -> float; default {default_val}", file=sys.stderr)
                return float(default_val)

    if isinstance(val, Sequence) and not isinstance(val, (bytes, bytearray, str)):
        for x in val:
            if isinstance(x, (int, float)):
                return float(x)
        print(f"[threed] WARN: sequence '{val}' not numeric-first; default {default_val}", file=sys.stderr)
        return float(default_val)

    print(f"[threed] WARN: unexpected type {type(val)} (value: {val}); default {default_val}", file=sys.stderr)
    return float(default_val)

def _coerce_to_int(val: Any, default_val: int) -> int:
    if val is None:
        return int(default_val)
    if isinstance(val, bool):
        return int(default_val)
    if isinstance(val, int):
        return int(val)
    if isinstance(val, float):
        return int(round(val))
    if isinstance(val, str):
        try:
            return int(round(float(val.strip())))
        except Exception:
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, (int, float)):
                    return int(round(float(parsed)))
            except Exception:
                pass
    return int(default_val)

def _coerce_to_bool(val: Any, default_val: bool) -> bool:
    if val is None:
        return bool(default_val)
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default_val)

def _vec3_from(
    p: Dict[str, Any],
    *,
    prefix: str,
    default: Tuple[float, float, float],
    alt_key: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Supports either separate keys like tx/ty/tz (prefix='t' -> tx/ty/tz),
    or a combined tuple/list via alt_key (e.g. 't' => "(0.1, 0.2, 0.0)").
    """
    if alt_key and alt_key in p:
        v = p.get(alt_key)
        if isinstance(v, str):
            try:
                v = ast.literal_eval(v)
            except Exception:
                v = None
        if isinstance(v, Sequence) and not isinstance(v, (bytes, bytearray, str)) and len(v) >= 3:
            return (
                _coerce_to_float(v[0], default[0]),
                _coerce_to_float(v[1], default[1]),
                _coerce_to_float(v[2], default[2]),
            )

    return (
        _coerce_to_float(p.get(f"{prefix}x"), default[0]),
        _coerce_to_float(p.get(f"{prefix}y"), default[1]),
        _coerce_to_float(p.get(f"{prefix}z"), default[2]),
    )

def _as_mode(p: Dict[str, Any], default: str) -> str:
    return str(p.get("mode", default)).strip().lower()

def _as_color(p: Dict[str, Any], key: str, default: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    c = _parse_color(p.get(key), default)
    return c if isinstance(c, tuple) and len(c) >= 3 else default


# =============================================================================
# Blocks (help/params annotated)
# =============================================================================

@help("Wireframe cube in 3D (orthographic/perspective/isometric).")
@params({
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    "size": {"type": "float", "default": 0.5, "min": 0.0, "max": 2.0, "hint": "scale in screen units; unit cube is side=2"},
    "angle_x": {"type": "float", "default": 20.0, "unit": "deg"},
    "angle_y": {"type": "float", "default": 35.0, "unit": "deg"},
    "angle_z": {"type": "float", "default": 0.0, "unit": "deg"},
    "tx": {"type": "float", "default": 0.0},
    "ty": {"type": "float", "default": 0.0},
    "tz": {"type": "float", "default": 0.0},
    "mode": {"type": "enum", "default": "orthographic", "choices": ["orthographic", "perspective", "isometric"]},
    "color": {"type": "color", "default": "#00ffff"},
    "width": {"type": "int", "default": 2, "min": 1, "max": 32},
    "depth_tint": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0},
    "fov": {"type": "float", "default": 60.0, "unit": "deg"},
    "z_offset": {"type": "float", "default": 2.5},
})
@dataclass
class DrawCube(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        p = dict(params or {})
        cx = _coerce_to_float(p.get("cx"), 0.5)
        cy = _coerce_to_float(p.get("cy"), 0.5)
        size = _coerce_to_float(p.get("size"), 0.5) * 0.5  # unit cube side=2
        ax = math.radians(_coerce_to_float(p.get("angle_x"), 20.0))
        ay = math.radians(_coerce_to_float(p.get("angle_y"), 35.0))
        az = math.radians(_coerce_to_float(p.get("angle_z"), 0.0))
        tx = _coerce_to_float(p.get("tx"), 0.0)
        ty = _coerce_to_float(p.get("ty"), 0.0)
        tz = _coerce_to_float(p.get("tz"), 0.0)
        mode = _as_mode(p, "orthographic")
        color = _as_color(p, "color", (0, 255, 255, 255))
        width_px = _coerce_to_int(p.get("width"), 2)
        depth_tint = max(0.0, min(1.0, _coerce_to_float(p.get("depth_tint"), 0.45)))
        fov = _coerce_to_float(p.get("fov"), 60.0)
        z_off = _coerce_to_float(p.get("z_offset"), 2.5)

        V = _unit_cube() * size
        R = _compose_rot(ax, ay, az)
        V = (V @ R.T) + np.array([tx, ty, tz], dtype=np.float32)

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        edge_depth = (Z[[a for a, b in _CUBE_EDGES]] + Z[[b for a, b in _CUBE_EDGES]]) * 0.5
        col_edges = _depth_tint(color, edge_depth, k=depth_tint)

        for i, (a, b) in enumerate(_CUBE_EDGES):
            draw.line([tuple(XY[a]), tuple(XY[b])], fill=col_edges[i], width=width_px)
        return img


@help("Wireframe square pyramid in 3D.")
@params({
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    "size": {"type": "float", "default": 0.6, "min": 0.0, "max": 2.0},
    "angle_x": {"type": "float", "default": 15.0, "unit": "deg"},
    "angle_y": {"type": "float", "default": 25.0, "unit": "deg"},
    "angle_z": {"type": "float", "default": 0.0, "unit": "deg"},
    "tx": {"type": "float", "default": 0.0},
    "ty": {"type": "float", "default": 0.0},
    "tz": {"type": "float", "default": 0.0},
    "mode": {"type": "enum", "default": "orthographic", "choices": ["orthographic", "perspective", "isometric"]},
    "color": {"type": "color", "default": "#ffcc66"},
    "width": {"type": "int", "default": 2, "min": 1, "max": 32},
    "depth_tint": {"type": "float", "default": 0.4, "min": 0.0, "max": 1.0},
    "fov": {"type": "float", "default": 60.0, "unit": "deg"},
    "z_offset": {"type": "float", "default": 2.5},
})
@dataclass
class DrawPyramid(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        p = dict(params or {})
        cx = _coerce_to_float(p.get("cx"), 0.5)
        cy = _coerce_to_float(p.get("cy"), 0.5)
        size = _coerce_to_float(p.get("size"), 0.6) * 0.5
        ax = math.radians(_coerce_to_float(p.get("angle_x"), 15.0))
        ay = math.radians(_coerce_to_float(p.get("angle_y"), 25.0))
        az = math.radians(_coerce_to_float(p.get("angle_z"), 0.0))
        tx = _coerce_to_float(p.get("tx"), 0.0)
        ty = _coerce_to_float(p.get("ty"), 0.0)
        tz = _coerce_to_float(p.get("tz"), 0.0)
        mode = _as_mode(p, "orthographic")
        color = _as_color(p, "color", (255, 204, 102, 255))
        width_px = _coerce_to_int(p.get("width"), 2)
        depth_tint = max(0.0, min(1.0, _coerce_to_float(p.get("depth_tint"), 0.4)))
        fov = _coerce_to_float(p.get("fov"), 60.0)
        z_off = _coerce_to_float(p.get("z_offset"), 2.5)

        V0, ED = _square_pyramid()
        V = V0 * size
        R = _compose_rot(ax, ay, az)
        V = (V @ R.T) + np.array([tx, ty, tz], dtype=np.float32)

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        edge_depth = (Z[[a for a, b in ED]] + Z[[b for a, b in ED]]) * 0.5
        cols = _depth_tint(color, edge_depth, k=depth_tint)

        for i, (a, b) in enumerate(ED):
            draw.line([tuple(XY[a]), tuple(XY[b])], fill=cols[i], width=width_px)
        return img


@help("Draw XYZ axes as 3 colored lines from the origin.")
@params({
    "cx": {"type": "float", "default": 0.15, "min": 0.0, "max": 1.0},
    "cy": {"type": "float", "default": 0.85, "min": 0.0, "max": 1.0},
    "length": {"type": "float", "default": 0.35, "min": 0.0, "max": 5.0},
    "width": {"type": "int", "default": 4, "min": 1, "max": 64},
    "mode": {"type": "enum", "default": "orthographic", "choices": ["orthographic", "perspective", "isometric"]},
    "angle_x": {"type": "float", "default": 30.0, "unit": "deg"},
    "angle_y": {"type": "float", "default": 30.0, "unit": "deg"},
    "angle_z": {"type": "float", "default": 0.0, "unit": "deg"},
    "fov": {"type": "float", "default": 60.0, "unit": "deg"},
    "z_offset": {"type": "float", "default": 2.5},
})
@dataclass
class DrawAxes3D(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        p = dict(params or {})
        cx = _coerce_to_float(p.get("cx"), 0.15)
        cy = _coerce_to_float(p.get("cy"), 0.85)
        length = _coerce_to_float(p.get("length"), 0.35)
        width_px = _coerce_to_int(p.get("width"), 4)
        mode = _as_mode(p, "orthographic")
        fov = _coerce_to_float(p.get("fov"), 60.0)
        z_off = _coerce_to_float(p.get("z_offset"), 2.5)
        ax = math.radians(_coerce_to_float(p.get("angle_x"), 30.0))
        ay = math.radians(_coerce_to_float(p.get("angle_y"), 30.0))
        az = math.radians(_coerce_to_float(p.get("angle_z"), 0.0))

        V, ED, cols = _axis_lines(length)
        R = _compose_rot(ax, ay, az)
        V = V @ R.T

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        depth = (Z[[a for a, b in ED]] + Z[[b for a, b in ED]]) * 0.5
        cols_tinted = [_depth_tint(c, np.array([d], dtype=np.float32), k=0.35)[0] for c, d in zip(cols, depth)]

        for i, (a, b) in enumerate(ED):
            draw.line([tuple(XY[a]), tuple(XY[b])], fill=cols_tinted[i], width=width_px)
        return img


@help("Dot sphere (latitude/longitude sample) projected to 2D.")
@params({
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    "radius": {"type": "float", "default": 0.45, "min": 0.0, "max": 2.0},
    "rings": {"type": "int", "default": 14, "min": 2, "max": 512},
    "segments": {"type": "int", "default": 20, "min": 3, "max": 1024},
    "dot": {"type": "int", "default": 2, "min": 1, "max": 32},
    "mode": {"type": "enum", "default": "perspective", "choices": ["orthographic", "perspective", "isometric"]},
    "fov": {"type": "float", "default": 60.0, "unit": "deg"},
    "z_offset": {"type": "float", "default": 2.5},
    "angle_x": {"type": "float", "default": 0.0, "unit": "deg"},
    "angle_y": {"type": "float", "default": 0.0, "unit": "deg"},
    "angle_z": {"type": "float", "default": 0.0, "unit": "deg"},
    "color": {"type": "color", "default": "#ffffff"},
    "depth_tint": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
})
@dataclass
class DrawSpherePoints(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        p = dict(params or {})
        cx = _coerce_to_float(p.get("cx"), 0.5)
        cy = _coerce_to_float(p.get("cy"), 0.5)
        rad = _coerce_to_float(p.get("radius"), 0.45)
        rings = max(2, _coerce_to_int(p.get("rings"), 14))
        segs = max(3, _coerce_to_int(p.get("segments"), 20))
        dot = max(1, _coerce_to_int(p.get("dot"), 2))
        mode = _as_mode(p, "perspective")
        fov = _coerce_to_float(p.get("fov"), 60.0)
        z_off = _coerce_to_float(p.get("z_offset"), 2.5)
        ax = math.radians(_coerce_to_float(p.get("angle_x"), 0.0))
        ay = math.radians(_coerce_to_float(p.get("angle_y"), 0.0))
        az = math.radians(_coerce_to_float(p.get("angle_z"), 0.0))
        color = _as_color(p, "color", (255, 255, 255, 255))
        depth_tint = max(0.0, min(1.0, _coerce_to_float(p.get("depth_tint"), 0.5)))

        pts: List[Tuple[float, float, float]] = []
        for i in range(1, rings):  # omit poles for neatness
            phi = math.pi * i / rings
            for j in range(segs):
                theta = 2 * math.pi * j / segs
                x = math.sin(phi) * math.cos(theta)
                y = math.cos(phi)
                z = math.sin(phi) * math.sin(theta)
                pts.append((x, y, z))
        V = np.array(pts, dtype=np.float32) * rad

        R = _compose_rot(ax, ay, az)
        V = V @ R.T

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        cols = _depth_tint(color, Z, k=depth_tint)

        r = dot / 2.0
        for (x, y), c in zip(XY, cols):
            draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
        return img


@help("Draw a 3D polyline from 'points' formatted as 'x:y:z, x:y:z, ...'.")
@params({
    "points": {"type": "str", "default": "", "hint": "format: 'x:y:z, x:y:z, ...'"},
    "closed": {"type": "bool", "default": False},
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    "mode": {"type": "enum", "default": "orthographic", "choices": ["orthographic", "perspective", "isometric"]},
    "fov": {"type": "float", "default": 60.0, "unit": "deg"},
    "z_offset": {"type": "float", "default": 2.5},
    "color": {"type": "color", "default": "#ff99ff"},
    "width": {"type": "int", "default": 3, "min": 1, "max": 64},
    "angle_x": {"type": "float", "default": 0.0, "unit": "deg"},
    "angle_y": {"type": "float", "default": 0.0, "unit": "deg"},
    "angle_z": {"type": "float", "default": 0.0, "unit": "deg"},
    "tx": {"type": "float", "default": 0.0},
    "ty": {"type": "float", "default": 0.0},
    "tz": {"type": "float", "default": 0.0},
    "scale": {"type": "float", "default": 1.0, "min": 0.0, "max": 1e6},
    "depth_tint": {"type": "float", "default": 0.3, "min": 0.0, "max": 1.0},
})
@dataclass
class Polyline3D(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)
        p = dict(params or {})

        def parse_points(s: str) -> np.ndarray:
            out: List[Tuple[float, float, float]] = []
            for chunk in (s or "").split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    xs, ys, zs = chunk.split(":")
                    out.append((float(xs), float(ys), float(zs)))
                except Exception:
                    pass
            if not out:
                out = [(-0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0, 0), (0, -0.5, 0)]
            return np.array(out, dtype=np.float32)

        pts = parse_points(str(p.get("points", "")))
        closed = _coerce_to_bool(p.get("closed"), False)
        cx = _coerce_to_float(p.get("cx"), 0.5)
        cy = _coerce_to_float(p.get("cy"), 0.5)
        mode = _as_mode(p, "orthographic")
        fov = _coerce_to_float(p.get("fov"), 60.0)
        z_off = _coerce_to_float(p.get("z_offset"), 2.5)
        color = _as_color(p, "color", (255, 153, 255, 255))
        width_px = _coerce_to_int(p.get("width"), 3)
        ax = math.radians(_coerce_to_float(p.get("angle_x"), 0.0))
        ay = math.radians(_coerce_to_float(p.get("angle_y"), 0.0))
        az = math.radians(_coerce_to_float(p.get("angle_z"), 0.0))
        tx = _coerce_to_float(p.get("tx"), 0.0)
        ty = _coerce_to_float(p.get("ty"), 0.0)
        tz = _coerce_to_float(p.get("tz"), 0.0)
        scale = _coerce_to_float(p.get("scale"), 1.0)
        depth_tint = max(0.0, min(1.0, _coerce_to_float(p.get("depth_tint"), 0.3)))

        V = pts * scale
        R = _compose_rot(ax, ay, az)
        V = (V @ R.T) + np.array([tx, ty, tz], dtype=np.float32)

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)

        indices = list(range(len(XY) - 1))
        if closed and len(XY) > 2:
            indices.append(len(XY) - 1)

        def seg(i: int) -> Tuple[int, int]:
            if i == len(XY) - 1 and closed:
                return i, 0
            return i, i + 1

        depths = np.array([(Z[a] + Z[b]) * 0.5 for a, b in (seg(i) for i in indices)], dtype=np.float32)
        cols = _depth_tint(color, depths, k=depth_tint)

        for i, (a, b) in enumerate(seg(j) for j in indices):
            draw.line([tuple(XY[a]), tuple(XY[b])], fill=cols[i], width=width_px)
        return img


# ---------------- New Blocks ----------------

@help("3D ground grid plane (clipped to image).")
@params({
    "cx": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    "cy": {"type": "float", "default": 0.75, "min": 0.0, "max": 1.0},
    "spacing": {"type": "float", "default": 0.1, "min": 1e-6, "max": 10.0, "hint": "world units between lines"},
    "lines": {"type": "int", "default": 20, "min": 1, "max": 1000, "hint": "count each side of origin"},
    "angle_x": {"type": "float", "default": 80.0, "unit": "deg"},
    "angle_y": {"type": "float", "default": 0.0, "unit": "deg"},
    "angle_z": {"type": "float", "default": 0.0, "unit": "deg"},
    "tx": {"type": "float", "default": 0.0},
    "ty": {"type": "float", "default": -0.3},
    "tz": {"type": "float", "default": 0.0},
    "mode": {"type": "enum", "default": "perspective", "choices": ["orthographic", "perspective", "isometric"]},
    "color": {"type": "color", "default": "#2a3355"},
    "width": {"type": "int", "default": 1, "min": 1, "max": 32},
    "fov": {"type": "float", "default": 60.0, "unit": "deg"},
    "z_offset": {"type": "float", "default": 2.5},
})
@dataclass
class DrawGrid3D(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        p = dict(params or {})
        cx = _coerce_to_float(p.get("cx"), 0.5)
        cy = _coerce_to_float(p.get("cy"), 0.75)
        spacing = max(1e-9, _coerce_to_float(p.get("spacing"), 0.1))
        lines = max(1, _coerce_to_int(p.get("lines"), 20))
        ax = math.radians(_coerce_to_float(p.get("angle_x"), 80.0))
        ay = math.radians(_coerce_to_float(p.get("angle_y"), 0.0))
        az = math.radians(_coerce_to_float(p.get("angle_z"), 0.0))
        tx = _coerce_to_float(p.get("tx"), 0.0)
        ty = _coerce_to_float(p.get("ty"), -0.3)
        tz = _coerce_to_float(p.get("tz"), 0.0)
        mode = _as_mode(p, "perspective")
        color = _as_color(p, "color", (42, 51, 85, 255))
        width_px = max(1, _coerce_to_int(p.get("width"), 1))
        fov = _coerce_to_float(p.get("fov"), 60.0)
        z_off = _coerce_to_float(p.get("z_offset"), 2.5)

        half = lines * spacing

        coords: List[np.ndarray] = []
        for i in range(-lines, lines + 1):
            z = i * spacing
            coords.append(np.array([[-half, 0.0, z], [half, 0.0, z]], dtype=np.float32))
        for i in range(-lines, lines + 1):
            x = i * spacing
            coords.append(np.array([[x, 0.0, -half], [x, 0.0, half]], dtype=np.float32))

        R = _compose_rot(ax, ay, az)
        off = np.array([tx, ty, tz], dtype=np.float32)

        for seg3 in coords:
            V = (seg3 @ R.T) + off
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
            col = _depth_tint(color, np.array([(Z[0] + Z[1]) * 0.5], dtype=np.float32), k=0.5)[0]
            draw.line([tuple(XY[0]), tuple(XY[1])], fill=col, width=width_px)

        return img


@help("Regular icosahedron wireframe.")
@params({
    "cx": {"type": "float", "default": 0.75, "min": 0.0, "max": 1.0},
    "cy": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0},
    "size": {"type": "float", "default": 0.45, "min": 0.0, "max": 2.0},
    "angle_x": {"type": "float", "default": 10.0, "unit": "deg"},
    "angle_y": {"type": "float", "default": 20.0, "unit": "deg"},
    "angle_z": {"type": "float", "default": 0.0, "unit": "deg"},
    "tx": {"type": "float", "default": 0.0},
    "ty": {"type": "float", "default": 0.0},
    "tz": {"type": "float", "default": 0.0},
    "mode": {"type": "enum", "default": "perspective", "choices": ["orthographic", "perspective", "isometric"]},
    "color": {"type": "color", "default": "#ffd166"},
    "width": {"type": "int", "default": 2, "min": 1, "max": 32},
    "depth_tint": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0},
    "fov": {"type": "float", "default": 60.0, "unit": "deg"},
    "z_offset": {"type": "float", "default": 2.5},
})
@dataclass
class DrawIcosahedron(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        p = dict(params or {})
        cx = _coerce_to_float(p.get("cx"), 0.75)
        cy = _coerce_to_float(p.get("cy"), 0.45)
        size = _coerce_to_float(p.get("size"), 0.45) * 0.5
        ax = math.radians(_coerce_to_float(p.get("angle_x"), 10.0))
        ay = math.radians(_coerce_to_float(p.get("angle_y"), 20.0))
        az = math.radians(_coerce_to_float(p.get("angle_z"), 0.0))
        tx = _coerce_to_float(p.get("tx"), 0.0)
        ty = _coerce_to_float(p.get("ty"), 0.0)
        tz = _coerce_to_float(p.get("tz"), 0.0)
        mode = _as_mode(p, "perspective")
        color = _as_color(p, "color", (255, 209, 102, 255))
        width_px = _coerce_to_int(p.get("width"), 2)
        depth_tint = max(0.0, min(1.0, _coerce_to_float(p.get("depth_tint"), 0.35)))
        fov = _coerce_to_float(p.get("fov"), 60.0)
        z_off = _coerce_to_float(p.get("z_offset"), 2.5)

        V0, ED = _icosahedron()
        V = V0 * size
        R = _compose_rot(ax, ay, az)
        V = (V @ R.T) + np.array([tx, ty, tz], dtype=np.float32)

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        depths = (Z[[a for a, b in ED]] + Z[[b for a, b in ED]]) * 0.5
        cols = _depth_tint(color, depths, k=depth_tint)

        for i, (a, b) in enumerate(ED):
            draw.line([tuple(XY[a]), tuple(XY[b])], fill=cols[i], width=width_px)
        return img


@help("Torus wireframe: draws both major rings and minor segments.")
@params({
    "cx": {"type": "float", "default": 0.6, "min": 0.0, "max": 1.0},
    "cy": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    "radius": {"type": "float", "default": 0.35, "min": 0.0, "max": 10.0, "hint": "major radius"},
    "tube": {"type": "float", "default": 0.10, "min": 0.0, "max": 10.0, "hint": "minor radius"},
    "rings": {"type": "int", "default": 18, "min": 3, "max": 1024},
    "segments": {"type": "int", "default": 24, "min": 4, "max": 2048},
    "angle_x": {"type": "float", "default": 70.0, "unit": "deg"},
    "angle_y": {"type": "float", "default": 10.0, "unit": "deg"},
    "angle_z": {"type": "float", "default": 0.0, "unit": "deg"},
    "mode": {"type": "enum", "default": "perspective", "choices": ["orthographic", "perspective", "isometric"]},
    "color": {"type": "color", "default": "#a8ffcf"},
    "width": {"type": "int", "default": 2, "min": 1, "max": 32},
    "depth_tint": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    "fov": {"type": "float", "default": 60.0, "unit": "deg"},
    "z_offset": {"type": "float", "default": 2.5},
})
@dataclass
class DrawTorusWire(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        p = dict(params or {})
        cx = _coerce_to_float(p.get("cx"), 0.6)
        cy = _coerce_to_float(p.get("cy"), 0.5)
        Rm = _coerce_to_float(p.get("radius"), 0.35)
        r = _coerce_to_float(p.get("tube"), 0.10)
        rings = max(3, _coerce_to_int(p.get("rings"), 18))
        segs = max(4, _coerce_to_int(p.get("segments"), 24))
        ax = math.radians(_coerce_to_float(p.get("angle_x"), 70.0))
        ay = math.radians(_coerce_to_float(p.get("angle_y"), 10.0))
        az = math.radians(_coerce_to_float(p.get("angle_z"), 0.0))
        mode = _as_mode(p, "perspective")
        color = _as_color(p, "color", (168, 255, 207, 255))
        width_px = _coerce_to_int(p.get("width"), 2)
        depth_tint = max(0.0, min(1.0, _coerce_to_float(p.get("depth_tint"), 0.5)))
        fov = _coerce_to_float(p.get("fov"), 60.0)
        z_off = _coerce_to_float(p.get("z_offset"), 2.5)

        R = _compose_rot(ax, ay, az)

        def torus_point(u, v):
            x = (Rm + r * math.cos(v)) * math.cos(u)
            y = (Rm + r * math.cos(v)) * math.sin(u)
            z = r * math.sin(v)
            return np.array([x, y, z], dtype=np.float32)

        # Rings (vary u; fixed v)
        for j in range(rings):
            v = 2 * math.pi * j / rings
            poly = np.array([torus_point(2 * math.pi * i / segs, v) for i in range(segs)], dtype=np.float32)
            poly = np.vstack([poly, poly[0]])
            V = poly @ R.T
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
            depths = (Z[:-1] + Z[1:]) * 0.5
            cols = _depth_tint(color, depths, k=depth_tint)
            for i in range(len(XY) - 1):
                draw.line([tuple(XY[i]), tuple(XY[i + 1])], fill=cols[i], width=width_px)

        # Segments (vary v; fixed u)
        for i0 in range(segs):
            u = 2 * math.pi * i0 / segs
            poly = np.array([torus_point(u, 2 * math.pi * j / rings) for j in range(rings)], dtype=np.float32)
            poly = np.vstack([poly, poly[0]])
            V = poly @ R.T
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
            depths = (Z[:-1] + Z[1:]) * 0.5
            cols = _depth_tint(color, depths, k=depth_tint)
            for i in range(len(XY) - 1):
                draw.line([tuple(XY[i]), tuple(XY[i + 1])], fill=cols[i], width=width_px)

        return img


@help("Camera frustum wireframe (near/far planes + side edges).")
@params({
    "cx": {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0},
    "cy": {"type": "float", "default": 0.28, "min": 0.0, "max": 1.0},
    "fov": {"type": "float", "default": 60.0, "unit": "deg"},
    "aspect": {"type": "float", "default": 16/9, "min": 1e-6, "max": 1e6},
    "near": {"type": "float", "default": 0.5, "min": 1e-6, "max": 1e6},
    "far": {"type": "float", "default": 1.5, "min": 1e-6, "max": 1e6},
    "angle_x": {"type": "float", "default": 12.0, "unit": "deg"},
    "angle_y": {"type": "float", "default": 25.0, "unit": "deg"},
    "angle_z": {"type": "float", "default": 0.0, "unit": "deg"},
    "mode": {"type": "enum", "default": "orthographic", "choices": ["orthographic", "perspective", "isometric"]},
    "color": {"type": "color", "default": "#ffa8a8"},
    "width": {"type": "int", "default": 2, "min": 1, "max": 32},
    "z_offset": {"type": "float", "default": 2.5},
})
@dataclass
class DrawFrustum(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        p = dict(params or {})
        cx = _coerce_to_float(p.get("cx"), 0.2)
        cy = _coerce_to_float(p.get("cy"), 0.28)
        fov = _coerce_to_float(p.get("fov"), 60.0)
        aspect = _coerce_to_float(p.get("aspect"), 16/9)
        n = _coerce_to_float(p.get("near"), 0.5)
        f = _coerce_to_float(p.get("far"), 1.5)
        ax = math.radians(_coerce_to_float(p.get("angle_x"), 12.0))
        ay = math.radians(_coerce_to_float(p.get("angle_y"), 25.0))
        az = math.radians(_coerce_to_float(p.get("angle_z"), 0.0))
        mode = _as_mode(p, "orthographic")
        color = _as_color(p, "color", (255, 168, 168, 255))
        width_px = _coerce_to_int(p.get("width"), 2)
        z_off = _coerce_to_float(p.get("z_offset"), 2.5)

        hfov = math.radians(fov) * 0.5
        nh = math.tan(hfov) * n
        nw = nh * aspect
        fh = math.tan(hfov) * f
        fw = fh * aspect

        near = np.array([[-nw,  nh, n], [ nw,  nh, n], [ nw, -nh, n], [-nw, -nh, n]], dtype=np.float32)
        far  = np.array([[-fw,  fh, f], [ fw,  fh, f], [ fw, -fh, f], [-fw, -fh, f]], dtype=np.float32)
        V = np.vstack([near, far])

        R = _compose_rot(ax, ay, az)
        V = V @ R.T

        ED = [
            (0,1),(1,2),(2,3),(3,0),
            (4,5),(5,6),(6,7),(7,4),
            (0,4),(1,5),(2,6),(3,7),
        ]

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        depths = (Z[[a for a, b in ED]] + Z[[b for a, b in ED]]) * 0.5
        cols = _depth_tint(color, depths, k=0.35)
        for i, (a, b) in enumerate(ED):
            draw.line([tuple(XY[a]), tuple(XY[b])], fill=cols[i], width=width_px)
        return img


@help("3D spiral/helix polyline.")
@params({
    "cx": {"type": "float", "default": 0.86, "min": 0.0, "max": 1.0},
    "cy": {"type": "float", "default": 0.72, "min": 0.0, "max": 1.0},
    "turns": {"type": "float", "default": 5.0, "min": 0.0, "max": 1e6},
    "height": {"type": "float", "default": 1.2, "min": 0.0, "max": 1e6},
    "radius": {"type": "float", "default": 0.22, "min": 0.0, "max": 1e6},
    "points": {"type": "int", "default": 260, "min": 8, "max": 100000},
    "angle_x": {"type": "float", "default": 0.0, "unit": "deg"},
    "angle_y": {"type": "float", "default": 0.0, "unit": "deg"},
    "angle_z": {"type": "float", "default": 0.0, "unit": "deg"},
    "mode": {"type": "enum", "default": "perspective", "choices": ["orthographic", "perspective", "isometric"]},
    "color": {"type": "color", "default": "#ff99ff"},
    "width": {"type": "int", "default": 3, "min": 1, "max": 64},
    "depth_tint": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
    "fov": {"type": "float", "default": 60.0, "unit": "deg"},
    "z_offset": {"type": "float", "default": 2.5},
})
@dataclass
class DrawSpiral3D(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        p = dict(params or {})
        cx = _coerce_to_float(p.get("cx"), 0.86)
        cy = _coerce_to_float(p.get("cy"), 0.72)
        turns = _coerce_to_float(p.get("turns"), 5.0)
        height_u = _coerce_to_float(p.get("height"), 1.2)
        rad = _coerce_to_float(p.get("radius"), 0.22)
        pts_n = max(8, _coerce_to_int(p.get("points"), 260))
        ax = math.radians(_coerce_to_float(p.get("angle_x"), 0.0))
        ay = math.radians(_coerce_to_float(p.get("angle_y"), 0.0))
        az = math.radians(_coerce_to_float(p.get("angle_z"), 0.0))
        mode = _as_mode(p, "perspective")
        color = _as_color(p, "color", (255, 153, 255, 255))
        width_px = _coerce_to_int(p.get("width"), 3)
        depth_tint = max(0.0, min(1.0, _coerce_to_float(p.get("depth_tint"), 0.5)))
        fov = _coerce_to_float(p.get("fov"), 60.0)
        z_off = _coerce_to_float(p.get("z_offset"), 2.5)

        t = np.linspace(0.0, 1.0, pts_n, dtype=np.float32)
        theta = 2 * math.pi * turns * t
        z = (t - 0.5) * height_u
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)
        V = np.stack([x, y, z], axis=1)

        R = _compose_rot(ax, ay, az)
        V = V @ R.T

        XY, Zs = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        depths = (Zs[:-1] + Zs[1:]) * 0.5
        cols = _depth_tint(color, depths, k=depth_tint)

        for i in range(len(XY) - 1):
            draw.line([tuple(XY[i]), tuple(XY[i + 1])], fill=cols[i], width=width_px)
        return img


@help("Cylinder wireframe: caps + verticals + optional intermediate rings.")
@params({
    "cx": {"type": "float", "default": 0.32, "min": 0.0, "max": 1.0},
    "cy": {"type": "float", "default": 0.38, "min": 0.0, "max": 1.0},
    "radius": {"type": "float", "default": 0.18, "min": 0.0, "max": 1e6},
    "height": {"type": "float", "default": 0.6, "min": 0.0, "max": 1e6},
    "rings": {"type": "int", "default": 6, "min": 1, "max": 2048, "hint": "horizontal rings along height (including caps are always drawn)"},
    "segments": {"type": "int", "default": 24, "min": 8, "max": 4096, "hint": "segments around circle"},
    "angle_x": {"type": "float", "default": 15.0, "unit": "deg"},
    "angle_y": {"type": "float", "default": 0.0, "unit": "deg"},
    "angle_z": {"type": "float", "default": 0.0, "unit": "deg"},
    "mode": {"type": "enum", "default": "perspective", "choices": ["orthographic", "perspective", "isometric"]},
    "color": {"type": "color", "default": "#88cfff"},
    "width": {"type": "int", "default": 2, "min": 1, "max": 32},
    "depth_tint": {"type": "float", "default": 0.45, "min": 0.0, "max": 1.0},
    "fov": {"type": "float", "default": 60.0, "unit": "deg"},
    "z_offset": {"type": "float", "default": 2.5},
})
@dataclass
class DrawCylinderWire(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        p = dict(params or {})
        cx = _coerce_to_float(p.get("cx"), 0.32)
        cy = _coerce_to_float(p.get("cy"), 0.38)
        Rr = _coerce_to_float(p.get("radius"), 0.18)
        H = _coerce_to_float(p.get("height"), 0.6)
        rings = max(1, _coerce_to_int(p.get("rings"), 6))
        segs = max(8, _coerce_to_int(p.get("segments"), 24))
        ax = math.radians(_coerce_to_float(p.get("angle_x"), 15.0))
        ay = math.radians(_coerce_to_float(p.get("angle_y"), 0.0))
        az = math.radians(_coerce_to_float(p.get("angle_z"), 0.0))
        mode = _as_mode(p, "perspective")
        color = _as_color(p, "color", (136, 207, 255, 255))
        width_px = _coerce_to_int(p.get("width"), 2)
        depth_tint = max(0.0, min(1.0, _coerce_to_float(p.get("depth_tint"), 0.45)))
        fov = _coerce_to_float(p.get("fov"), 60.0)
        z_off = _coerce_to_float(p.get("z_offset"), 2.5)

        R = _compose_rot(ax, ay, az)

        # Caps
        for z0 in (-H / 2, H / 2):
            ring = np.array(
                [[Rr * math.cos(2 * math.pi * i / segs),
                  Rr * math.sin(2 * math.pi * i / segs),
                  z0] for i in range(segs + 1)],
                dtype=np.float32,
            )
            V = ring @ R.T
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
            depths = (Z[:-1] + Z[1:]) * 0.5
            cols = _depth_tint(color, depths, k=depth_tint)
            for i in range(len(XY) - 1):
                draw.line([tuple(XY[i]), tuple(XY[i + 1])], fill=cols[i], width=width_px)

        # Vertical lines
        for i0 in range(segs):
            ang = 2 * math.pi * i0 / segs
            x = Rr * math.cos(ang)
            y = Rr * math.sin(ang)
            seg3 = np.array([[x, y, -H / 2], [x, y, H / 2]], dtype=np.float32)
            V = seg3 @ R.T
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
            col = _depth_tint(color, np.array([(Z[0] + Z[1]) * 0.5], dtype=np.float32), k=depth_tint)[0]
            draw.line([tuple(XY[0]), tuple(XY[1])], fill=col, width=width_px)

        # Intermediate rings
        for j in range(1, rings):
            z0 = -H / 2 + H * (j / rings)
            ring = np.array(
                [[Rr * math.cos(2 * math.pi * i / segs),
                  Rr * math.sin(2 * math.pi * i / segs),
                  z0] for i in range(segs + 1)],
                dtype=np.float32,
            )
            V = ring @ R.T
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
            depths = (Z[:-1] + Z[1:]) * 0.5
            cols = _depth_tint(color, depths, k=depth_tint)
            for i in range(len(XY) - 1):
                draw.line([tuple(XY[i]), tuple(XY[i + 1])], fill=cols[i], width=width_px)

        return img


@help("Oriented rectangle plane (optionally filled), projected from 3D.")
@params({
    "cx": {"type": "float", "default": 0.82, "min": 0.0, "max": 1.0},
    "cy": {"type": "float", "default": 0.22, "min": 0.0, "max": 1.0},
    "size_x": {"type": "float", "default": 0.45, "min": 0.0, "max": 1e6},
    "size_y": {"type": "float", "default": 0.25, "min": 0.0, "max": 1e6},
    "angle_x": {"type": "float", "default": 10.0, "unit": "deg"},
    "angle_y": {"type": "float", "default": 20.0, "unit": "deg"},
    "angle_z": {"type": "float", "default": 15.0, "unit": "deg"},
    "angle": {"type": "any", "default": None, "nullable": True, "hint": "tuple/list '(ax,ay,az)' overrides angle_x/y/z"},
    "tx": {"type": "float", "default": 0.0},
    "ty": {"type": "float", "default": 0.0},
    "tz": {"type": "float", "default": 0.0},
    "t": {"type": "any", "default": None, "nullable": True, "hint": "tuple/list '(tx,ty,tz)' overrides tx/ty/tz"},
    "mode": {"type": "enum", "default": "perspective", "choices": ["orthographic", "perspective", "isometric"]},
    "color": {"type": "color", "default": "#ffffff", "hint": "outline color"},
    "fill": {"type": "color", "default": "none", "hint": "set to a color to fill; use 'none' to disable"},
    "width": {"type": "int", "default": 2, "min": 1, "max": 64},
    "fov": {"type": "float", "default": 60.0, "unit": "deg"},
    "z_offset": {"type": "float", "default": 2.5},
})
@dataclass
class DrawPlane(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        try:
            p = dict(params or {})

            # If an RGBA/RGB-looking tuple landed in 'tx', treat it as 'color'
            _tx = p.get("tx")
            if (
                isinstance(_tx, tuple)
                and 3 <= len(_tx) <= 4
                and all(isinstance(x, int) and 0 <= x <= 255 for x in _tx)
                and "color" not in p
            ):
                p["color"] = _tx
                p["tx"] = 0.0

            cx = _coerce_to_float(p.get("cx"), 0.82)
            cy = _coerce_to_float(p.get("cy"), 0.22)
            sx = _coerce_to_float(p.get("size_x"), 0.45)
            sy = _coerce_to_float(p.get("size_y"), 0.25)

            ax_deg, ay_deg, az_deg = _vec3_from(p, prefix="angle_", default=(10.0, 20.0, 15.0), alt_key="angle")
            tx, ty, tz = _vec3_from(p, prefix="t", default=(0.0, 0.0, 0.0), alt_key="t")

            # Allow explicit overrides
            tx = _coerce_to_float(p.get("tx"), tx)
            ty = _coerce_to_float(p.get("ty"), ty)
            tz = _coerce_to_float(p.get("tz"), tz)

            fov = _coerce_to_float(p.get("fov"), 60.0)
            z_off = _coerce_to_float(p.get("z_offset"), 2.5)

            ax = math.radians(ax_deg)
            ay = math.radians(ay_deg)
            az = math.radians(az_deg)

            mode = _as_mode(p, "perspective")

            outline = _parse_color(p.get("color", "#ffffff"), (255, 255, 255, 255))
            outline = outline if isinstance(outline, tuple) and len(outline) >= 3 else (255, 255, 255, 255)

            fill_col = _parse_color(p.get("fill", "none"), None)
            if isinstance(fill_col, str) and fill_col.strip().lower() == "none":
                fill_col = None

            width_px = max(1, _coerce_to_int(p.get("width"), 2))

            V = np.array(
                [[-sx/2, -sy/2, 0],
                 [ sx/2, -sy/2, 0],
                 [ sx/2,  sy/2, 0],
                 [-sx/2,  sy/2, 0]], dtype=np.float32
            )
            R = _compose_rot(ax, ay, az)
            V = (V @ R.T) + np.array([tx, ty, tz], dtype=np.float32)
            XY, _Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)

            poly = [tuple(map(float, pt)) for pt in XY]

            if fill_col is not None and isinstance(fill_col, tuple) and 3 <= len(fill_col) <= 4:
                draw.polygon(poly, fill=fill_col)

            for i in range(4):
                a, b = i, (i + 1) % 4
                draw.line([poly[a], poly[b]], fill=outline, width=width_px)

            return img

        except Exception as e:
            print(f"[DrawPlane] ERROR: {e}", file=sys.stderr)
            print(f"[DrawPlane] Params received: {params}", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (width, height), (255, 0, 0, 128))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
REGISTRY.register("drawcube", DrawCube)
REGISTRY.register("drawpyramid", DrawPyramid)
REGISTRY.register("drawaxes3d", DrawAxes3D)
REGISTRY.register("drawspherepoints", DrawSpherePoints)
REGISTRY.register("polyline3d", Polyline3D)

REGISTRY.register("drawgrid3d", DrawGrid3D)
REGISTRY.register("drawicosahedron", DrawIcosahedron)
REGISTRY.register("drawtoruswire", DrawTorusWire)
REGISTRY.register("drawfrustum", DrawFrustum)
REGISTRY.register("drawspiral3d", DrawSpiral3D)
REGISTRY.register("drawcylinderwire", DrawCylinderWire)
REGISTRY.register("drawplane", DrawPlane)
