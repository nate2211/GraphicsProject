# threed.py — lightweight 3D wireframes for your GraphicsEngine
# Requires: numpy, Pillow
# Works with your graphics.py registry/engine.
#
# New blocks added:
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

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
import sys
import ast
from collections.abc import Sequence
# Pull base helpers/registry from graphics.py
try:
    from graphics import BaseBlock, REGISTRY, _ensure_image, _parse_color
except Exception:
    # Fallback stubs (lets file import without graphics.py for linting)
    class BaseBlock:  # type: ignore
        pass

    class _DummyReg:  # type: ignore
        def register(self, *_a, **_k): ...
    REGISTRY = _DummyReg()  # type: ignore

    def _ensure_image(img, w, h):
        return Image.new("RGBA", (w, h))

    def _parse_color(val: Any, default: Tuple[int, int, int, int]):
        return default


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
        # Place geometry in front of camera with z_offset
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
        d = 1.0 - k * ti
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


def _coerce_to_float(val, default_val):
    if val is None:
        return float(default_val)
    # allow "0.25" or "(0.25,0,0)" as strings
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            val = parsed
        except Exception:
            try:
                return float(val)
            except Exception:
                print(f"ERROR [DrawPlane]: could not parse string '{val}' -> float; default {default_val}", file=sys.stderr)
                return float(default_val)
    # allow sequences: take first numeric item
    if isinstance(val, Sequence) and not isinstance(val, (bytes, bytearray, str)):
        for x in val:
            if isinstance(x, (int, float)):
                return float(x)
        # looks like a color tuple (all ints 0..255)? fall through to default
        print(f"ERROR [DrawPlane]: sequence '{val}' not numeric-first; default {default_val}", file=sys.stderr)
        return float(default_val)
    if isinstance(val, (int, float)):
        return float(val)
    print(f"ERROR [DrawPlane]: unexpected type {type(val)} (value: {val}); default {default_val}", file=sys.stderr)
    return float(default_val)

def get_float_param(p_name, default_val):
    return _coerce_to_float(params.get(p_name), default_val)

def get_vec3(prefix, default=(0.0, 0.0, 0.0), alt_key=None):
    """
    Supports either separate keys like tx/ty/tz or a combined tuple/list via alt_key (e.g. 't').
    """
    if alt_key and alt_key in params:
        v = params.get(alt_key)
        # handle string "(0.1, 0.2, 0.0)"
        if isinstance(v, str):
            try: v = ast.literal_eval(v)
            except Exception: v = None
        if isinstance(v, Sequence) and len(v) >= 3:
            vx = _coerce_to_float(v[0], default[0])
            vy = _coerce_to_float(v[1], default[1])
            vz = _coerce_to_float(v[2], default[2])
            return vx, vy, vz
    # fallback to individual params
    return (
        get_float_param(f"{prefix}x", default[0]),
        get_float_param(f"{prefix}y", default[1]),
        get_float_param(f"{prefix}z", default[2]),
    )

# ---------------- Blocks (existing) ----------------

@dataclass
class DrawCube(BaseBlock):
    """Wireframe cube.
    Params:
      cx,cy (0..1) center | size (0..1 of min dimension)
      angle_x,angle_y,angle_z (deg) | mode: orthographic|perspective|isometric
      tx,ty,tz (world translation before projection)
      color | width(px) | depth_tint(0..1) | fov(deg, perspective only) | z_offset
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = float(params.get("cx", 0.5))
        cy = float(params.get("cy", 0.5))
        size = float(params.get("size", 0.5)) * 0.5  # since unit cube is side=2
        ax = math.radians(float(params.get("angle_x", 20.0)))
        ay = math.radians(float(params.get("angle_y", 35.0)))
        az = math.radians(float(params.get("angle_z", 0.0)))
        tx = float(params.get("tx", 0.0))
        ty = float(params.get("ty", 0.0))
        tz = float(params.get("tz", 0.0))
        mode = str(params.get("mode", "orthographic"))
        color = _parse_color(params.get("color", "#00ffff"), (0,255,255,255))
        width_px = int(params.get("width", 2))
        depth_tint = max(0.0, min(1.0, float(params.get("depth_tint", 0.45))))
        fov = float(params.get("fov", 60.0))
        z_off = float(params.get("z_offset", 2.5))

        V = _unit_cube() * size
        R = _compose_rot(ax, ay, az)
        V = V @ R.T
        V += np.array([tx, ty, tz], dtype=np.float32)

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        edge_depth = (Z[[a for a,b in _CUBE_EDGES]] + Z[[b for a,b in _CUBE_EDGES]]) * 0.5
        col_edges = _depth_tint(color, edge_depth, k=depth_tint)

        for (i, (a, b)) in enumerate(_CUBE_EDGES):
            c = col_edges[i] if i < len(col_edges) else color
            p0 = tuple(XY[a])
            p1 = tuple(XY[b])
            draw.line([p0, p1], fill=c, width=width_px)

        return img


@dataclass
class DrawPyramid(BaseBlock):
    """Wireframe square pyramid."""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = float(params.get("cx", 0.5)); cy = float(params.get("cy", 0.5))
        size = float(params.get("size", 0.6)) * 0.5
        ax = math.radians(float(params.get("angle_x", 15.0)))
        ay = math.radians(float(params.get("angle_y", 25.0)))
        az = math.radians(float(params.get("angle_z",  0.0)))
        tx = float(params.get("tx", 0.0)); ty = float(params.get("ty", 0.0)); tz = float(params.get("tz", 0.0))
        mode = str(params.get("mode", "orthographic"))
        color = _parse_color(params.get("color", "#ffcc66"), (255,204,102,255))
        width_px = int(params.get("width", 2))
        depth_tint = max(0.0, min(1.0, float(params.get("depth_tint", 0.4))))
        fov = float(params.get("fov", 60.0))
        z_off = float(params.get("z_offset", 2.5))

        V0, ED = _square_pyramid()
        V = V0 * size
        R = _compose_rot(ax, ay, az)
        V = V @ R.T
        V += np.array([tx, ty, tz], dtype=np.float32)

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        edge_depth = (Z[[a for a,b in ED]] + Z[[b for a,b in ED]]) * 0.5
        col_edges = _depth_tint(color, edge_depth, k=depth_tint)

        for i, (a, b) in enumerate(ED):
            c = col_edges[i] if i < len(col_edges) else color
            draw.line([tuple(XY[a]), tuple(XY[b])], fill=c, width=width_px)
        return img


@dataclass
class DrawAxes3D(BaseBlock):
    """XYZ axes from origin."""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = float(params.get("cx", 0.15)); cy = float(params.get("cy", 0.85))
        length = float(params.get("length", 0.35))
        width_px = int(params.get("width", 4))
        mode = str(params.get("mode", "orthographic"))
        fov = float(params.get("fov", 60.0))
        z_off = float(params.get("z_offset", 2.5))
        ax = math.radians(float(params.get("angle_x", 30.0)))
        ay = math.radians(float(params.get("angle_y", 30.0)))
        az = math.radians(float(params.get("angle_z",  0.0)))

        V, ED, cols = _axis_lines(length)
        R = _compose_rot(ax, ay, az)
        V = V @ R.T

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        # Per-edge colors (X,Y,Z) with mild depth tint
        depth = (Z[[a for a,b in ED]] + Z[[b for a,b in ED]]) * 0.5
        cols_tinted = [_depth_tint(c, np.array([d]), k=0.35)[0] for c, d in zip(cols, depth)]

        for (i, (a, b)) in enumerate(ED):
            draw.line([tuple(XY[a]), tuple(XY[b])], fill=cols_tinted[i], width=width_px)
        return img


@dataclass
class DrawSpherePoints(BaseBlock):
    """Dot sphere (latitude/longitude)."""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = float(params.get("cx", 0.5)); cy = float(params.get("cy", 0.5))
        rad = float(params.get("radius", 0.45))
        rings = max(2, int(params.get("rings", 14)))
        segs  = max(3, int(params.get("segments", 20)))
        dot   = max(1, int(params.get("dot", 2)))
        mode  = str(params.get("mode", "perspective"))
        fov   = float(params.get("fov", 60.0))
        z_off = float(params.get("z_offset", 2.5))
        ax = math.radians(float(params.get("angle_x", 0.0)))
        ay = math.radians(float(params.get("angle_y", 0.0)))
        az = math.radians(float(params.get("angle_z", 0.0)))
        color = _parse_color(params.get("color", "#ffffff"), (255,255,255,255))
        depth_tint = max(0.0, min(1.0, float(params.get("depth_tint", 0.5))))

        # Build points on a unit sphere, then scale
        pts: List[Tuple[float,float,float]] = []
        for i in range(1, rings):  # omit poles for neatness
            phi = math.pi * i / rings  # (0..pi)
            for j in range(segs):
                theta = 2 * math.pi * j / segs  # (0..2pi)
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


@dataclass
class Polyline3D(BaseBlock):
    """Draw a 3D polyline given points."""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        def parse_points(s: str) -> np.ndarray:
            out: List[Tuple[float,float,float]] = []
            for chunk in (s or "").split(","):
                chunk = chunk.strip()
                if not chunk: continue
                try:
                    xs, ys, zs = chunk.split(":")
                    out.append((float(xs), float(ys), float(zs)))
                except Exception:
                    pass
            if not out:
                # default little diamond
                out = [(-0.5,0,0),(0,0.5,0),(0,0,0.5),(0.5,0,0),(0,-0.5,0)]
            return np.array(out, dtype=np.float32)

        pts = parse_points(str(params.get("points", "")))
        closed = bool(params.get("closed", False))
        cx = float(params.get("cx", 0.5)); cy = float(params.get("cy", 0.5))
        mode = str(params.get("mode", "orthographic"))
        fov  = float(params.get("fov", 60.0))
        z_off = float(params.get("z_offset", 2.5))
        color = _parse_color(params.get("color", "#ff99ff"), (255,153,255,255))
        width_px = int(params.get("width", 3))
        ax = math.radians(float(params.get("angle_x", 0.0)))
        ay = math.radians(float(params.get("angle_y", 0.0)))
        az = math.radians(float(params.get("angle_z", 0.0)))
        tx = float(params.get("tx", 0.0)); ty = float(params.get("ty", 0.0)); tz = float(params.get("tz", 0.0))
        scale = float(params.get("scale", 1.0))
        depth_tint = max(0.0, min(1.0, float(params.get("depth_tint", 0.3))))

        V = pts * scale
        R = _compose_rot(ax, ay, az)
        V = V @ R.T
        V += np.array([tx, ty, tz], dtype=np.float32)

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        # Draw as segments to apply depth tint per segment
        indices = list(range(len(XY) - 1))
        if closed and len(XY) > 2:
            indices += [len(XY) - 1]  # last to first

        def seg(i: int) -> Tuple[int,int]:
            if i == len(XY) - 1 and closed:
                return i, 0
            return i, i + 1

        depths = np.array([(Z[a] + Z[b]) * 0.5 for a, b in (seg(i) for i in indices)], dtype=np.float32)
        cols = _depth_tint(color, depths, k=depth_tint)

        for (i, (a, b)) in enumerate(seg(j) for j in indices):
            draw.line([tuple(XY[a]), tuple(XY[b])], fill=cols[i], width=width_px)
        return img


# ---------------- New Blocks ----------------

@dataclass
class DrawGrid3D(BaseBlock):
    """Grid plane in 3D (like a ground grid).
    Params:
      cx,cy | spacing (world units between lines) | lines (count each side)
      angle_x/y/z (deg) orientation | tx/ty/tz | mode | color | width(px)
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = float(params.get("cx", 0.5)); cy = float(params.get("cy", 0.75))
        spacing = float(params.get("spacing", 0.1))
        lines = int(params.get("lines", 20))
        ax = math.radians(float(params.get("angle_x", 80.0)))  # tilt towards camera by default
        ay = math.radians(float(params.get("angle_y", 0.0)))
        az = math.radians(float(params.get("angle_z", 0.0)))
        tx = float(params.get("tx", 0.0)); ty = float(params.get("ty", -0.3)); tz = float(params.get("tz", 0.0))
        mode = str(params.get("mode", "perspective"))
        color = _parse_color(params.get("color", "#2a3355"), (42,51,85,255))
        width_px = int(params.get("width", 1))
        fov = float(params.get("fov", 60.0))
        z_off = float(params.get("z_offset", 2.5))

        # Build grid lines on XZ plane at y=0
        half = lines * spacing
        xs = np.linspace(-half, half, 2)
        coords: List[np.ndarray] = []
        # lines parallel to X (vary z)
        for i in range(-lines, lines + 1):
            z = i * spacing
            coords.append(np.array([[ -half, 0.0, z ],
                                    [  half, 0.0, z ]], dtype=np.float32))
        # lines parallel to Z (vary x)
        for i in range(-lines, lines + 1):
            x = i * spacing
            coords.append(np.array([[ x, 0.0, -half ],
                                    [ x, 0.0,  half ]], dtype=np.float32))

        R = _compose_rot(ax, ay, az)
        for seg3 in coords:
            V = (seg3 @ R.T) + np.array([tx, ty, tz], dtype=np.float32)
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
            col = _depth_tint(color, np.array([(Z[0]+Z[1])*0.5]), k=0.5)[0]
            draw.line([tuple(XY[0]), tuple(XY[1])], fill=col, width=width_px)
        return img


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

@dataclass
class DrawIcosahedron(BaseBlock):
    """Regular icosahedron wireframe.
    Params: cx,cy,size, angle_x/y/z, tx/ty/tz, mode, width, color, depth_tint, fov, z_offset
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = float(params.get("cx", 0.75)); cy = float(params.get("cy", 0.45))
        size = float(params.get("size", 0.45)) * 0.5
        ax = math.radians(float(params.get("angle_x", 10.0)))
        ay = math.radians(float(params.get("angle_y", 20.0)))
        az = math.radians(float(params.get("angle_z", 0.0)))
        tx = float(params.get("tx", 0.0)); ty = float(params.get("ty", 0.0)); tz = float(params.get("tz", 0.0))
        mode = str(params.get("mode", "perspective"))
        color = _parse_color(params.get("color", "#ffd166"), (255,209,102,255))
        width_px = int(params.get("width", 2))
        depth_tint = max(0.0, min(1.0, float(params.get("depth_tint", 0.35))))
        fov = float(params.get("fov", 60.0))
        z_off = float(params.get("z_offset", 2.5))

        V0, ED = _icosahedron()
        V = V0 * size
        R = _compose_rot(ax, ay, az)
        V = (V @ R.T) + np.array([tx, ty, tz], dtype=np.float32)

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        depths = (Z[[a for a,b in ED]] + Z[[b for a,b in ED]]) * 0.5
        cols = _depth_tint(color, depths, k=depth_tint)

        for (i, (a, b)) in enumerate(ED):
            draw.line([tuple(XY[a]), tuple(XY[b])], fill=cols[i], width=width_px)
        return img


@dataclass
class DrawTorusWire(BaseBlock):
    """Torus wireframe (rings × segments).
    Params:
      cx,cy | radius(major), tube(minor) | rings, segments
      angle_x/y/z | mode | color | width | depth_tint | fov | z_offset
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = float(params.get("cx", 0.6)); cy = float(params.get("cy", 0.5))
        Rm = float(params.get("radius", 0.35))
        r  = float(params.get("tube", 0.10))
        rings = max(3, int(params.get("rings", 18)))
        segs  = max(4, int(params.get("segments", 24)))
        ax = math.radians(float(params.get("angle_x", 70.0)))
        ay = math.radians(float(params.get("angle_y", 10.0)))
        az = math.radians(float(params.get("angle_z", 0.0)))
        mode = str(params.get("mode", "perspective"))
        color = _parse_color(params.get("color", "#a8ffcf"), (168,255,207,255))
        width_px = int(params.get("width", 2))
        depth_tint = max(0.0, min(1.0, float(params.get("depth_tint", 0.5))))
        fov = float(params.get("fov", 60.0))
        z_off = float(params.get("z_offset", 2.5))

        R = _compose_rot(ax, ay, az)

        # Build two families of polylines: rings (around major circle) and segments (around minor circle)
        def torus_point(u, v):
            # u: around major circle, v: around minor cross-section
            x = (Rm + r * math.cos(v)) * math.cos(u)
            y = (Rm + r * math.cos(v)) * math.sin(u)
            z = r * math.sin(v)
            return np.array([x, y, z], dtype=np.float32)

        # Rings (vary u; fixed v)
        for j in range(rings):
            v = 2 * math.pi * j / rings
            poly = np.array([torus_point(2*math.pi*i/segs, v) for i in range(segs)], dtype=np.float32)
            poly = np.vstack([poly, poly[0]])  # close
            V = (poly @ R.T)
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
            depths = (Z[:-1] + Z[1:]) * 0.5
            cols = _depth_tint(color, depths, k=depth_tint)
            for i in range(len(XY)-1):
                draw.line([tuple(XY[i]), tuple(XY[i+1])], fill=cols[i], width=width_px)

        # Segments (vary v; fixed u)
        for i0 in range(segs):
            u = 2 * math.pi * i0 / segs
            poly = np.array([torus_point(u, 2*math.pi*j/rings) for j in range(rings)], dtype=np.float32)
            poly = np.vstack([poly, poly[0]])
            V = (poly @ R.T)
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
            depths = (Z[:-1] + Z[1:]) * 0.5
            cols = _depth_tint(color, depths, k=depth_tint)
            for i in range(len(XY)-1):
                draw.line([tuple(XY[i]), tuple(XY[i+1])], fill=cols[i], width=width_px)

        return img


@dataclass
class DrawFrustum(BaseBlock):
    """Camera frustum visualization.
    Params:
      cx,cy | fov(deg) | aspect | near | far | angle_x/y/z | mode | color | width
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = float(params.get("cx", 0.2)); cy = float(params.get("cy", 0.28))
        fov = float(params.get("fov", 60.0))
        aspect = float(params.get("aspect", 16/9))
        n = float(params.get("near", 0.5))
        f = float(params.get("far",  1.5))
        ax = math.radians(float(params.get("angle_x", 12.0)))
        ay = math.radians(float(params.get("angle_y", 25.0)))
        az = math.radians(float(params.get("angle_z", 0.0)))
        mode = str(params.get("mode", "orthographic"))
        color = _parse_color(params.get("color", "#ffa8a8"), (255,168,168,255))
        width_px = int(params.get("width", 2))
        z_off = float(params.get("z_offset", 2.5))

        # Build 8 corner points in camera space looking down +Z
        hfov = math.radians(fov) * 0.5
        nh = math.tan(hfov) * n
        nw = nh * aspect
        fh = math.tan(hfov) * f
        fw = fh * aspect

        # Near plane corners (z = n)
        near = np.array([[-nw,  nh, n],
                         [ nw,  nh, n],
                         [ nw, -nh, n],
                         [-nw, -nh, n]], dtype=np.float32)
        # Far plane corners (z = f)
        far  = np.array([[-fw,  fh, f],
                         [ fw,  fh, f],
                         [ fw, -fh, f],
                         [-fw, -fh, f]], dtype=np.float32)
        V = np.vstack([near, far])  # 0..3 near, 4..7 far

        R = _compose_rot(ax, ay, az)
        V = V @ R.T

        # Edges: near rectangle, far rectangle, and 4 sides
        ED = [(0,1),(1,2),(2,3),(3,0),
              (4,5),(5,6),(6,7),(7,4),
              (0,4),(1,5),(2,6),(3,7)]

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        depths = (Z[[a for a,b in ED]] + Z[[b for a,b in ED]]) * 0.5
        cols = _depth_tint(color, depths, k=0.35)
        for i,(a,b) in enumerate(ED):
            draw.line([tuple(XY[a]), tuple(XY[b])], fill=cols[i], width=width_px)
        return img


@dataclass
class DrawSpiral3D(BaseBlock):
    """Helix/spiral polyline.
    Params:
      cx,cy | turns | height | radius | points | angle_x/y/z | mode | color | width | depth_tint
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = float(params.get("cx", 0.86)); cy = float(params.get("cy", 0.72))
        turns = float(params.get("turns", 5.0))
        height_u = float(params.get("height", 1.2))
        rad = float(params.get("radius", 0.22))
        pts_n = max(8, int(params.get("points", 260)))
        ax = math.radians(float(params.get("angle_x", 0.0)))
        ay = math.radians(float(params.get("angle_y", 0.0)))
        az = math.radians(float(params.get("angle_z", 0.0)))
        mode = str(params.get("mode", "perspective"))
        color = _parse_color(params.get("color", "#ff99ff"), (255,153,255,255))
        width_px = int(params.get("width", 3))
        depth_tint = max(0.0, min(1.0, float(params.get("depth_tint", 0.5))))
        fov = float(params.get("fov", 60.0))
        z_off = float(params.get("z_offset", 2.5))

        t = np.linspace(0.0, 1.0, pts_n, dtype=np.float32)
        theta = 2 * math.pi * turns * t
        z = (t - 0.5) * height_u
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)
        V = np.stack([x, y, z], axis=1)

        R = _compose_rot(ax, ay, az)
        V = V @ R.T

        XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
        depths = (Z[:-1] + Z[1:]) * 0.5
        cols = _depth_tint(color, depths, k=depth_tint)
        for i in range(len(XY)-1):
            draw.line([tuple(XY[i]), tuple(XY[i+1])], fill=cols[i], width=width_px)
        return img


@dataclass
class DrawCylinderWire(BaseBlock):
    """Cylinder wireframe (vertical axis).
    Params:
      cx,cy | radius | height | rings(int) | segments(int)
      angle_x/y/z | mode | color | width | depth_tint | fov | z_offset
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = float(params.get("cx", 0.32)); cy = float(params.get("cy", 0.38))
        Rr = float(params.get("radius", 0.18))
        H = float(params.get("height", 0.6))
        rings = max(1, int(params.get("rings", 6)))     # horizontal rings along height
        segs  = max(8, int(params.get("segments", 24))) # around circle
        ax = math.radians(float(params.get("angle_x", 15.0)))
        ay = math.radians(float(params.get("angle_y", 0.0)))
        az = math.radians(float(params.get("angle_z", 0.0)))
        mode = str(params.get("mode", "perspective"))
        color = _parse_color(params.get("color", "#88cfff"), (136,207,255,255))
        width_px = int(params.get("width", 2))
        depth_tint = max(0.0, min(1.0, float(params.get("depth_tint", 0.45))))
        fov = float(params.get("fov", 60.0))
        z_off = float(params.get("z_offset", 2.5))

        R = _compose_rot(ax, ay, az)

        # Caps (top/bottom rings)
        for z0 in (-H/2, H/2):
            ring = np.array([[Rr*math.cos(2*math.pi*i/segs),
                              Rr*math.sin(2*math.pi*i/segs),
                              z0] for i in range(segs+1)], dtype=np.float32)
            V = ring @ R.T
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
            depths = (Z[:-1] + Z[1:]) * 0.5
            cols = _depth_tint(color, depths, k=depth_tint)
            for i in range(len(XY)-1):
                draw.line([tuple(XY[i]), tuple(XY[i+1])], fill=cols[i], width=width_px)

        # Vertical lines at angle steps
        for i0 in range(segs):
            ang = 2*math.pi*i0/segs
            x = Rr * math.cos(ang); y = Rr * math.sin(ang)
            seg3 = np.array([[x,y,-H/2],[x,y, H/2]], dtype=np.float32)
            V = seg3 @ R.T
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
            col = _depth_tint(color, np.array([(Z[0]+Z[1])*0.5]), k=depth_tint)[0]
            draw.line([tuple(XY[0]), tuple(XY[1])], fill=col, width=width_px)

        # Horizontal rings along height
        for j in range(1, rings):
            z0 = -H/2 + H * (j / rings)
            ring = np.array([[Rr*math.cos(2*math.pi*i/segs),
                              Rr*math.sin(2*math.pi*i/segs),
                              z0] for i in range(segs+1)], dtype=np.float32)
            V = ring @ R.T
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)
            depths = (Z[:-1] + Z[1:]) * 0.5
            cols = _depth_tint(color, depths, k=depth_tint)
            for i in range(len(XY)-1):
                draw.line([tuple(XY[i]), tuple(XY[i+1])], fill=cols[i], width=width_px)
        return img


@dataclass
class DrawPlane(BaseBlock):
    """Simple oriented rectangle plane.
    Params:
      cx,cy | size_x,size_y | angle_x/y/z | tx/ty/tz or t=(x,y,z) | mode | color | width | fill(None to disable) | fov | z_offset
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        try:
            # ----------- Helpers (nested; close over local 'params') -----------
            params = dict(params or {})

            # If an RGBA/RGB-looking tuple landed in 'tx', treat it as 'color'
            _tx = params.get("tx")
            if (
                isinstance(_tx, tuple)
                and 3 <= len(_tx) <= 4
                and all(isinstance(x, int) and 0 <= x <= 255 for x in _tx)
                and "color" not in params
            ):
                params["color"] = _tx
                params["tx"] = 0.0  # reset translation x

            def _coerce_to_float(val, default_val):
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
                            print(f"ERROR [DrawPlane]: could not parse string '{val}' -> float; default {default_val}", file=sys.stderr)
                            return float(default_val)
                if isinstance(val, Sequence) and not isinstance(val, (bytes, bytearray, str)):
                    # take first numeric element if present
                    for x in val:
                        if isinstance(x, (int, float)):
                            return float(x)
                    print(f"ERROR [DrawPlane]: sequence '{val}' not numeric-first; default {default_val}", file=sys.stderr)
                    return float(default_val)
                print(f"ERROR [DrawPlane]: unexpected type {type(val)} (value: {val}); default {default_val}", file=sys.stderr)
                return float(default_val)

            def _get_float(name, default_val):
                return _coerce_to_float(params.get(name), default_val)

            def _get_vec3(prefix, default=(0.0, 0.0, 0.0), alt_key=None):
                """
                Use separate keys via 'prefix' ('t' -> tx,ty,tz) or combined tuple via alt_key (e.g., 't').
                """
                if alt_key and alt_key in params:
                    v = params.get(alt_key)
                    if isinstance(v, str):
                        try:
                            v = ast.literal_eval(v)
                        except Exception:
                            v = None
                    if isinstance(v, Sequence) and not isinstance(v, (bytes, bytearray, str)) and len(v) >= 3:
                        vx = _coerce_to_float(v[0], default[0])
                        vy = _coerce_to_float(v[1], default[1])
                        vz = _coerce_to_float(v[2], default[2])
                        return vx, vy, vz
                return (
                    _get_float(f"{prefix}x", default[0]),
                    _get_float(f"{prefix}y", default[1]),
                    _get_float(f"{prefix}z", default[2]),
                )

            # ----------------- Parse params (robust) -----------------
            cx = _get_float("cx", 0.82)
            cy = _get_float("cy", 0.22)
            sx = _get_float("size_x", 0.45)
            sy = _get_float("size_y", 0.25)

            ax_deg, ay_deg, az_deg = _get_vec3("angle_", default=(10.0, 20.0, 15.0), alt_key="angle")
            tx, ty, tz = _get_vec3("t", default=(0.0, 0.0, 0.0), alt_key="t")
            # allow individual overrides
            tx = _get_float("tx", tx)
            ty = _get_float("ty", ty)
            tz = _get_float("tz", tz)

            fov   = _get_float("fov", 60.0)
            z_off = _get_float("z_offset", 2.5)

            ax = math.radians(ax_deg)
            ay = math.radians(ay_deg)
            az = math.radians(az_deg)

            mode = str(params.get("mode", "perspective"))

            # Colors via _parse_color (accept tuples/strings/"none")
            color    = _parse_color(params.get("color", "#ffffff"), default=(255, 255, 255, 255))
            fill_col = _parse_color(params.get("fill", "none"),     default=None)

            try:
                width_px = int(params.get("width", 2))
            except (ValueError, TypeError):
                print(f"Warning [DrawPlane]: Could not convert param 'width' value '{params.get('width')}' to int. Using default 2.", file=sys.stderr)
                width_px = 2

            # ----------------- Geometry -----------------
            V = np.array(
                [[-sx/2, -sy/2, 0],
                 [ sx/2, -sy/2, 0],
                 [ sx/2,  sy/2, 0],
                 [-sx/2,  sy/2, 0]], dtype=np.float32
            )
            R = _compose_rot(ax, ay, az)
            V = (V @ R.T) + np.array([tx, ty, tz], dtype=np.float32)
            XY, Z = _project_points(V, mode, width, height, cx=cx, cy=cy, fov_deg=fov, z_offset=z_off)

            poly = [tuple(map(round, p)) for p in XY]

            # ----------------- Drawing -----------------
            # Fill
            if fill_col is not None:
                if isinstance(fill_col, tuple) and 3 <= len(fill_col) <= 4:
                    draw.polygon(poly, fill=fill_col)
                else:
                    print(f"Warning [DrawPlane]: Invalid fill color '{fill_col}' (type: {type(fill_col)}). Fill skipped.", file=sys.stderr)

            # Outline
            final_outline = color if (isinstance(color, tuple) and 3 <= len(color) <= 4) else None
            if final_outline:
                for i in range(4):
                    a, b = i, (i + 1) % 4
                    p_a = tuple(map(float, poly[a]))
                    p_b = tuple(map(float, poly[b]))
                    draw.line([p_a, p_b], fill=final_outline, width=width_px)
            else:
                print(f"Warning [DrawPlane]: Invalid outline color '{color}' (type: {type(color)}). Outline skipped.", file=sys.stderr)

            return img

        except Exception as e:
            # Make sure runtime doesn't crash the render loop
            print(f"!!! UNEXPECTED Error in DrawPlane.process: {e}", file=sys.stderr)
            print(f"    Params received: {params}", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (width, height), (255, 0, 0, 128))
# -------- Registration --------
REGISTRY.register("drawcube", DrawCube)
REGISTRY.register("drawpyramid", DrawPyramid)
REGISTRY.register("drawaxes3d", DrawAxes3D)
REGISTRY.register("drawspherepoints", DrawSpherePoints)
REGISTRY.register("polyline3d", Polyline3D)

# New ones:
REGISTRY.register("drawgrid3d", DrawGrid3D)
REGISTRY.register("drawicosahedron", DrawIcosahedron)
REGISTRY.register("drawtoruswire", DrawTorusWire)
REGISTRY.register("drawfrustum", DrawFrustum)
REGISTRY.register("drawspiral3d", DrawSpiral3D)
REGISTRY.register("drawcylinderwire", DrawCylinderWire)
REGISTRY.register("drawplane", DrawPlane)
