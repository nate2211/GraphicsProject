# designs.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageDraw

# Import from your existing graphics module
from graphics import BaseBlock, REGISTRY, _ensure_image, _parse_color, help, params


# ============================== Small helpers ==============================

def _px_or_norm(params: Dict[str, Any], key: str, size_px: int, default_norm: float) -> float:
    """
    Fetch a coordinate/length that may be given as absolute pixels (key+"_px")
    or normalized 0..1 (key). Returns float pixels.
    """
    if f"{key}_px" in params and params.get(f"{key}_px", None) is not None:
        return float(params[f"{key}_px"])
    v = params.get(key, default_norm)
    if v is None:
        v = default_norm
    return float(v) * float(size_px)


def _calc_star_points(cx: float, cy: float, outer_r: float, inner_r: float, num_points: int) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    step = math.pi / num_points
    for i in range(2 * num_points):
        r = outer_r if (i % 2 == 0) else inner_r
        ang = i * step - math.pi / 2.0
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    return pts


def _rotate_points(points: List[Tuple[float, float]], cx: float, cy: float, angle_deg: float) -> List[Tuple[float, float]]:
    if abs(angle_deg) < 1e-9:
        return points
    ang = math.radians(angle_deg)
    s, c = math.sin(ang), math.cos(ang)
    out: List[Tuple[float, float]] = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        xr = dx * c - dy * s
        yr = dx * s + dy * c
        out.append((cx + xr, cy + yr))
    return out


def _parse_points_list(points_str: str) -> List[Tuple[float, float]]:
    """
    Parse "x,y;x,y;..." into list of floats. Assumes normalized coordinates.
    """
    raw_pts: List[Tuple[float, float]] = []
    for token in (points_str or "").split(";"):
        token = token.strip()
        if not token or "," not in token:
            continue
        sx, sy = token.split(",", 1)
        try:
            raw_pts.append((float(sx), float(sy)))
        except Exception:
            pass
    return raw_pts


# ============================== Blocks ==============================

@help("Draw a star shape.\nParams: cx/cy (norm or *_px), outer_radius, inner_radius, points, rotation, fill, outline, outline_width.")
@params({
    "cx":            {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cx_px":         {"type": "int",   "default": None, "nullable": True},
    "cy":            {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy_px":         {"type": "int",   "default": None, "nullable": True},
    "outer_radius":  {"type": "float", "default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01},
    "inner_radius":  {"type": "float", "default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01},
    "points":        {"type": "int",   "default": 5,   "min": 3, "max": 64, "step": 1},
    "rotation":      {"type": "float", "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0},
    "fill":          {"type": "color", "default": "yellow"},
    "outline":       {"type": "color", "default": None, "nullable": True},
    "outline_width": {"type": "int",   "default": 1, "min": 0, "max": 200, "step": 1},
})
@dataclass
class DrawStarBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        outer_r = float(params.get("outer_radius", 0.2) or 0.2) * min(width, height)
        inner_r = float(params.get("inner_radius", 0.1) or 0.1) * min(width, height)
        points = max(3, int(params.get("points", 5) or 5))
        rot = float(params.get("rotation", 0.0) or 0.0)

        fill_color = _parse_color(params.get("fill", "yellow"), (255, 255, 0, 255))
        outline_color = _parse_color(params.get("outline", None), None)
        outline_width = int(params.get("outline_width", 1) or 1)

        pts = _calc_star_points(cx, cy, outer_r, inner_r, points)
        pts = _rotate_points(pts, cx, cy, rot)

        draw.polygon(
            pts,
            fill=fill_color if fill_color else None,
            outline=outline_color if outline_color else None,
            width=outline_width,
        )
        return img


@help("Draw a simple flower with circular petals.\nParams: cx/cy, petal_radius, center_radius, petals, rotation, petal_color, center_color.")
@params({
    "cx":            {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cx_px":         {"type": "int",   "default": None, "nullable": True},
    "cy":            {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy_px":         {"type": "int",   "default": None, "nullable": True},
    "petal_radius":  {"type": "float", "default": 0.15, "min": 0.0, "max": 2.0, "step": 0.01},
    "center_radius": {"type": "float", "default": 0.05, "min": 0.0, "max": 2.0, "step": 0.01},
    "petals":        {"type": "int",   "default": 6, "min": 3, "max": 128, "step": 1},
    "rotation":      {"type": "float", "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0},
    "petal_color":   {"type": "color", "default": "pink"},
    "center_color":  {"type": "color", "default": "yellow"},
})
@dataclass
class DrawFlowerBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        petal_r_rel = float(params.get("petal_radius", 0.15) or 0.15)
        center_r_rel = float(params.get("center_radius", 0.05) or 0.05)
        petal_r = petal_r_rel * min(width, height)
        center_r = center_r_rel * min(width, height)
        num_petals = max(3, int(params.get("petals", 6) or 6))
        rot = float(params.get("rotation", 0.0) or 0.0)

        petal_color = _parse_color(params.get("petal_color", "pink"), (255, 192, 203, 255))
        center_color = _parse_color(params.get("center_color", "yellow"), (255, 255, 0, 255))

        ang_step = 2.0 * math.pi / num_petals
        rot_rad = math.radians(rot)
        for i in range(num_petals):
            ang = i * ang_step + rot_rad
            px = cx + petal_r * math.cos(ang)
            py = cy + petal_r * math.sin(ang)
            bbox = [px - petal_r, py - petal_r, px + petal_r, py + petal_r]
            if petal_color:
                draw.ellipse(bbox, fill=petal_color)

        if center_r > 0 and center_color:
            cb = [cx - center_r, cy - center_r, cx + center_r, cy + center_r]
            draw.ellipse(cb, fill=center_color)

        return img


@help("Draw a regular polygon (n-gon).\nParams: cx/cy, radius, sides>=3, rotation, fill, outline, outline_width.")
@params({
    "cx":            {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cx_px":         {"type": "int",   "default": None, "nullable": True},
    "cy":            {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy_px":         {"type": "int",   "default": None, "nullable": True},
    "radius":        {"type": "float", "default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01},
    "sides":         {"type": "int",   "default": 6, "min": 3, "max": 128, "step": 1},
    "rotation":      {"type": "float", "default": -90.0, "min": -360.0, "max": 360.0, "step": 1.0},
    "fill":          {"type": "color", "default": None, "nullable": True},
    "outline":       {"type": "color", "default": "white"},
    "outline_width": {"type": "int",   "default": 2, "min": 0, "max": 200, "step": 1},
})
@dataclass
class DrawNGonBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        r = float(params.get("radius", 0.25) or 0.25) * min(width, height)
        sides = max(3, int(params.get("sides", 6) or 6))
        rot = float(params.get("rotation", -90.0) or -90.0)

        fill_color = _parse_color(params.get("fill", None), None)
        outline_color = _parse_color(params.get("outline", "white"), (255, 255, 255, 255))
        ow = int(params.get("outline_width", 2) or 2)

        pts = []
        for i in range(sides):
            a = 2.0 * math.pi * i / sides + math.radians(rot)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

        draw.polygon(
            pts,
            fill=fill_color if fill_color else None,
            outline=outline_color if outline_color else None,
            width=ow,
        )
        return img


@help(
    "Draw a polygon from a list of points.\n"
    "Params: points(\"x,y;x,y;...\") normalized, offset_x/offset_y(norm), rotation(deg), scale, fill, outline, outline_width."
)
@params({
    "points":        {"type": "str",   "default": "0.2,0.2;0.8,0.2;0.5,0.8"},
    "offset_x":      {"type": "float", "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
    "offset_y":      {"type": "float", "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01},
    "scale":         {"type": "float", "default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01},
    "rotation":      {"type": "float", "default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0},
    "fill":          {"type": "color", "default": None, "nullable": True},
    "outline":       {"type": "color", "default": "white"},
    "outline_width": {"type": "int",   "default": 2, "min": 0, "max": 200, "step": 1},
})
@dataclass
class DrawPolygonBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        pts_str = str(params.get("points", "0.2,0.2;0.8,0.2;0.5,0.8"))
        raw_pts = _parse_points_list(pts_str)
        if not raw_pts:
            return img

        offset_x = float(params.get("offset_x", 0.0) or 0.0) * width
        offset_y = float(params.get("offset_y", 0.0) or 0.0) * height
        scale = float(params.get("scale", 1.0) or 1.0)
        rot = float(params.get("rotation", 0.0) or 0.0)

        px_pts = [(x * width, y * height) for x, y in raw_pts]
        cx = sum(p[0] for p in px_pts) / len(px_pts)
        cy = sum(p[1] for p in px_pts) / len(px_pts)

        px_pts = [((x - cx) * scale + cx + offset_x, (y - cy) * scale + cy + offset_y) for x, y in px_pts]
        px_pts = _rotate_points(px_pts, cx + offset_x, cy + offset_y, rot)

        fill_color = _parse_color(params.get("fill", None), None)
        outline_color = _parse_color(params.get("outline", "white"), (255, 255, 255, 255))
        ow = int(params.get("outline_width", 2) or 2)

        draw.polygon(
            px_pts,
            fill=fill_color if fill_color else None,
            outline=outline_color if outline_color else None,
            width=ow,
        )
        return img


@help("Draw a heart curve.\nParams: cx/cy, size, rotation, steps, fill, outline, outline_width.")
@params({
    "cx":            {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cx_px":         {"type": "int",   "default": None, "nullable": True},
    "cy":            {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy_px":         {"type": "int",   "default": None, "nullable": True},
    "size":          {"type": "float", "default": 0.35, "min": 0.0, "max": 2.0, "step": 0.01},
    "rotation":      {"type": "float", "default": 0.0,  "min": -360.0, "max": 360.0, "step": 1.0},
    "steps":         {"type": "int",   "default": 200,  "min": 32, "max": 4096, "step": 1},
    "fill":          {"type": "color", "default": "red"},
    "outline":       {"type": "color", "default": "white"},
    "outline_width": {"type": "int",   "default": 2, "min": 0, "max": 200, "step": 1},
})
@dataclass
class DrawHeartBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        size = float(params.get("size", 0.35) or 0.35) * min(width, height)
        rot = float(params.get("rotation", 0.0) or 0.0)
        steps = max(64, int(params.get("steps", 200) or 200))

        fill_color = _parse_color(params.get("fill", "red"), (255, 0, 0, 255))
        outline_color = _parse_color(params.get("outline", "white"), (255, 255, 255, 255))
        ow = int(params.get("outline_width", 2) or 2)

        pts = []
        for i in range(steps):
            t = 2.0 * math.pi * i / steps
            x = 16 * (math.sin(t) ** 3)
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            x *= size / 18.0
            y *= -size / 18.0
            pts.append((cx + x, cy + y))

        pts = _rotate_points(pts, cx, cy, rot)

        if fill_color:
            draw.polygon(pts, fill=fill_color)
        if outline_color and ow > 0:
            draw.line(pts + [pts[0]], fill=outline_color, width=ow, joint="curve")
        return img


@help("Draw a ring/annulus as an outline with width.\nParams: cx/cy, outer_radius, inner_radius, ring_color, ring_width, fill.")
@params({
    "cx":           {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cx_px":        {"type": "int",   "default": None, "nullable": True},
    "cy":           {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy_px":        {"type": "int",   "default": None, "nullable": True},
    "outer_radius": {"type": "float", "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01},
    "inner_radius": {"type": "float", "default": 0.2, "min": 0.0, "max": 2.0, "step": 0.01},
    "ring_color":   {"type": "color", "default": "white"},
    "ring_width":   {"type": "int",   "default": None, "nullable": True, "min": 1, "max": 4096, "step": 1},
    "fill":         {"type": "color", "default": None, "nullable": True},
})
@dataclass
class DrawRingBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        r_out = float(params.get("outer_radius", 0.3) or 0.3) * min(width, height)
        r_in = float(params.get("inner_radius", 0.2) or 0.2) * min(width, height)

        ring_color = _parse_color(params.get("ring_color", "white"), (255, 255, 255, 255))
        fill_color = _parse_color(params.get("fill", None), None)

        # If ring_width is given, honor it. Otherwise approximate using radii difference.
        rw_raw = params.get("ring_width", None)
        if rw_raw is None:
            ring_width = max(1, int(abs(r_out - r_in)))
        else:
            ring_width = max(1, int(rw_raw))

        bbox_out = [cx - r_out, cy - r_out, cx + r_out, cy + r_out]
        if fill_color:
            draw.ellipse(bbox_out, fill=fill_color)

        draw.ellipse(bbox_out, outline=ring_color, width=ring_width)
        return img


@help("Draw an arrow from (x0,y0) to (x1,y1).\nParams: x0/y0/x1/y1 (norm or *_px), shaft_width_px, head_len_px, head_width_px, color.")
@params({
    "x0":             {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
    "x0_px":          {"type": "int",   "default": None, "nullable": True},
    "y0":             {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "y0_px":          {"type": "int",   "default": None, "nullable": True},
    "x1":             {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
    "x1_px":          {"type": "int",   "default": None, "nullable": True},
    "y1":             {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "y1_px":          {"type": "int",   "default": None, "nullable": True},
    "shaft_width_px": {"type": "float", "default": 6.0,  "min": 0.1, "max": 2000.0, "step": 0.1},
    "head_len_px":    {"type": "float", "default": 30.0, "min": 0.1, "max": 5000.0, "step": 0.1},
    "head_width_px":  {"type": "float", "default": 24.0, "min": 0.1, "max": 5000.0, "step": 0.1},
    "color":          {"type": "color", "default": "white"},
})
@dataclass
class DrawArrowBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        x0 = _px_or_norm(params, "x0", width, 0.2)
        y0 = _px_or_norm(params, "y0", height, 0.5)
        x1 = _px_or_norm(params, "x1", width, 0.8)
        y1 = _px_or_norm(params, "y1", height, 0.5)

        color = _parse_color(params.get("color", "white"), (255, 255, 255, 255))
        shaft_w = float(params.get("shaft_width_px", 6.0) or 6.0)
        head_len = float(params.get("head_len_px", 30.0) or 30.0)
        head_w = float(params.get("head_width_px", 24.0) or 24.0)

        dx, dy = x1 - x0, y1 - y0
        L = max(1e-6, math.hypot(dx, dy))
        ux, uy = dx / L, dy / L
        px, py = -uy, ux

        xe = x1 - ux * head_len
        ye = y1 - uy * head_len

        p1 = (x0 + px * shaft_w * 0.5, y0 + py * shaft_w * 0.5)
        p2 = (x0 - px * shaft_w * 0.5, y0 - py * shaft_w * 0.5)
        p3 = (xe - px * shaft_w * 0.5, ye - py * shaft_w * 0.5)
        p4 = (xe + px * shaft_w * 0.5, ye + py * shaft_w * 0.5)
        draw.polygon([p1, p2, p3, p4], fill=color)

        h1 = (xe + px * head_w * 0.5, ye + py * head_w * 0.5)
        h2 = (xe - px * head_w * 0.5, ye - py * head_w * 0.5)
        h3 = (x1, y1)
        draw.polygon([h1, h2, h3], fill=color)
        return img


@help("Draw a grid overlay.\nParams: rows, cols, color, width_px, margin_x, margin_y (norm).")
@params({
    "rows":     {"type": "int",   "default": 4, "min": 1, "max": 512, "step": 1},
    "cols":     {"type": "int",   "default": 6, "min": 1, "max": 512, "step": 1},
    "color":    {"type": "color", "default": "white"},
    "width_px": {"type": "int",   "default": 1, "min": 1, "max": 200, "step": 1},
    "margin_x": {"type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
    "margin_y": {"type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
})
@dataclass
class DrawGridBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        rows = max(1, int(params.get("rows", 4) or 4))
        cols = max(1, int(params.get("cols", 6) or 6))
        color = _parse_color(params.get("color", "white"), (255, 255, 255, 120))
        wpx = int(params.get("width_px", 1) or 1)
        mx = float(params.get("margin_x", 0.05) or 0.05) * width
        my = float(params.get("margin_y", 0.05) or 0.05) * height

        x0, y0 = mx, my
        x1, y1 = width - mx, height - my

        draw.rectangle([x0, y0, x1, y1], outline=color, width=wpx)

        for r in range(1, rows):
            yy = y0 + (y1 - y0) * r / rows
            draw.line([(x0, yy), (x1, yy)], fill=color, width=wpx)
        for c in range(1, cols):
            xx = x0 + (x1 - x0) * c / cols
            draw.line([(xx, y0), (xx, y1)], fill=color, width=wpx)

        return img


@help("Draw an Archimedean spiral.\nParams: cx/cy, turns, radius, thickness_px, color, rotation, steps.")
@params({
    "cx":           {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cx_px":        {"type": "int",   "default": None, "nullable": True},
    "cy":           {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy_px":        {"type": "int",   "default": None, "nullable": True},
    "turns":        {"type": "float", "default": 3.0, "min": 0.1, "max": 200.0, "step": 0.1},
    "radius":       {"type": "float", "default": 0.45, "min": 0.0, "max": 2.0, "step": 0.01},
    "rotation":     {"type": "float", "default": -90.0, "min": -360.0, "max": 360.0, "step": 1.0},
    "thickness_px": {"type": "int",   "default": 3, "min": 1, "max": 2000, "step": 1},
    "color":        {"type": "color", "default": "white"},
    "steps":        {"type": "int",   "default": 800, "min": 16, "max": 20000, "step": 1},
})
@dataclass
class DrawSpiralBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        turns = max(0.1, float(params.get("turns", 3.0) or 3.0))
        radius = float(params.get("radius", 0.45) or 0.45) * min(width, height)
        rot = math.radians(float(params.get("rotation", -90.0) or -90.0))
        thickness = int(params.get("thickness_px", 3) or 3)
        color = _parse_color(params.get("color", "white"), (255, 255, 255, 255))

        steps = int(params.get("steps", 800) or 800)
        a = 0.0
        b = radius / (2.0 * math.pi * turns)

        pts = []
        for i in range(steps + 1):
            t = (turns * 2.0 * math.pi) * (i / steps)
            r = a + b * t
            x = cx + r * math.cos(t + rot)
            y = cy + r * math.sin(t + rot)
            pts.append((x, y))

        draw.line(pts, fill=color, width=thickness, joint="curve")
        return img


@help("Draw a sunburst of alternating wedges.\nParams: cx/cy, inner_radius, outer_radius, wedges, start_deg, fill_a, fill_b.")
@params({
    "cx":           {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cx_px":        {"type": "int",   "default": None, "nullable": True},
    "cy":           {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy_px":        {"type": "int",   "default": None, "nullable": True},
    "inner_radius": {"type": "float", "default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01},
    "outer_radius": {"type": "float", "default": 0.48, "min": 0.0, "max": 2.0, "step": 0.01},
    "wedges":       {"type": "int",   "default": 16, "min": 1, "max": 1024, "step": 1},
    "start_deg":    {"type": "float", "default": -90.0, "min": -360.0, "max": 360.0, "step": 1.0},
    "fill_a":       {"type": "color", "default": "#FFD166"},
    "fill_b":       {"type": "color", "default": "#EF476F"},
})
@dataclass
class DrawSunburstBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        r0 = float(params.get("inner_radius", 0.1) or 0.1) * min(width, height)
        r1 = float(params.get("outer_radius", 0.48) or 0.48) * min(width, height)
        wedges = max(1, int(params.get("wedges", 16) or 16))
        start_deg = float(params.get("start_deg", -90.0) or -90.0)

        fa = _parse_color(params.get("fill_a", "#FFD166"), (255, 209, 102, 255))
        fb = _parse_color(params.get("fill_b", "#EF476F"), (239, 71, 111, 255))

        for i in range(wedges):
            a0 = math.radians(start_deg + (360.0 / wedges) * i)
            a1 = math.radians(start_deg + (360.0 / wedges) * (i + 1))
            pts = [
                (cx + r0 * math.cos(a0), cy + r0 * math.sin(a0)),
                (cx + r1 * math.cos(a0), cy + r1 * math.sin(a0)),
                (cx + r1 * math.cos(a1), cy + r1 * math.sin(a1)),
                (cx + r0 * math.cos(a1), cy + r0 * math.sin(a1)),
            ]
            draw.polygon(pts, fill=fa if (i % 2 == 0) else fb)
        return img


@help("Quadratic Bezier curve (polyline approximation).\nParams: x0/y0, x1/y1, cx/cy control, thickness_px, color, steps.")
@params({
    "x0":           {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
    "x0_px":        {"type": "int",   "default": None, "nullable": True},
    "y0":           {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
    "y0_px":        {"type": "int",   "default": None, "nullable": True},
    "x1":           {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
    "x1_px":        {"type": "int",   "default": None, "nullable": True},
    "y1":           {"type": "float", "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01},
    "y1_px":        {"type": "int",   "default": None, "nullable": True},
    "cx":           {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "cx_px":        {"type": "int",   "default": None, "nullable": True},
    "cy":           {"type": "float", "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
    "cy_px":        {"type": "int",   "default": None, "nullable": True},
    "thickness_px": {"type": "int",   "default": 3, "min": 1, "max": 2000, "step": 1},
    "color":        {"type": "color", "default": "white"},
    "steps":        {"type": "int",   "default": 200, "min": 8, "max": 20000, "step": 1},
})
@dataclass
class DrawBezierBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        x0 = _px_or_norm(params, "x0", width, 0.2)
        y0 = _px_or_norm(params, "y0", height, 0.8)
        x1 = _px_or_norm(params, "x1", width, 0.8)
        y1 = _px_or_norm(params, "y1", height, 0.8)
        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.2)

        steps = max(8, int(params.get("steps", 200) or 200))
        thickness = int(params.get("thickness_px", 3) or 3)
        color = _parse_color(params.get("color", "white"), (255, 255, 255, 255))

        pts = []
        for i in range(steps + 1):
            t = i / steps
            u = 1 - t
            x = u * u * x0 + 2 * u * t * cx + t * t * x1
            y = u * u * y0 + 2 * u * t * cy + t * t * y1
            pts.append((x, y))
        draw.line(pts, fill=color, width=thickness, joint="curve")
        return img


@help("Horizontal sine wave.\nParams: amp_px, freq(cycles), phase_deg, thickness_px, color, y(norm or y_px), steps.")
@params({
    "amp_px":       {"type": "float", "default": 20.0, "min": 0.0, "max": 5000.0, "step": 0.1},
    "freq":         {"type": "float", "default": 3.0,  "min": 0.0, "max": 200.0,  "step": 0.1},
    "phase_deg":    {"type": "float", "default": 0.0,  "min": -360.0, "max": 360.0, "step": 1.0},
    "y":            {"type": "float", "default": 0.5,  "min": 0.0, "max": 1.0, "step": 0.01},
    "y_px":         {"type": "int",   "default": None, "nullable": True},
    "thickness_px": {"type": "int",   "default": 3, "min": 1, "max": 2000, "step": 1},
    "color":        {"type": "color", "default": "white"},
    "steps":        {"type": "int",   "default": 500, "min": 16, "max": 20000, "step": 1},
})
@dataclass
class DrawWaveBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        amp = float(params.get("amp_px", 20.0) or 20.0)
        freq = float(params.get("freq", 3.0) or 3.0)
        phase = math.radians(float(params.get("phase_deg", 0.0) or 0.0))
        y = _px_or_norm(params, "y", height, 0.5)
        thickness = int(params.get("thickness_px", 3) or 3)
        color = _parse_color(params.get("color", "white"), (255, 255, 255, 255))
        steps = max(64, int(params.get("steps", 500) or 500))

        pts = []
        for i in range(steps + 1):
            t = i / steps
            xx = t * (width - 1)
            yy = y + amp * math.sin(2 * math.pi * freq * t + phase)
            pts.append((xx, yy))
        draw.line(pts, fill=color, width=thickness, joint="curve")
        return img


@help("Staggered dot pattern.\nParams: cell_px, radius_px, color, offset(0..1), margin_x, margin_y.")
@params({
    "cell_px":   {"type": "int",   "default": 24, "min": 2, "max": 4096, "step": 1},
    "radius_px": {"type": "int",   "default": 4,  "min": 1, "max": 2048, "step": 1},
    "color":     {"type": "color", "default": "white"},
    "offset":    {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
    "margin_x":  {"type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
    "margin_y":  {"type": "float", "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
})
@dataclass
class DrawDotsBlock(BaseBlock):
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cell = max(2, int(params.get("cell_px", 24) or 24))
        rad = max(1, int(params.get("radius_px", 4) or 4))
        color = _parse_color(params.get("color", "white"), (255, 255, 255, 180))
        mx = float(params.get("margin_x", 0.05) or 0.05) * width
        my = float(params.get("margin_y", 0.05) or 0.05) * height
        offset = float(params.get("offset", 0.5) or 0.5)

        x0, y0 = int(mx), int(my)
        x1, y1 = int(width - mx), int(height - my)
        shift = int(cell * offset)

        for j, yy in enumerate(range(y0, y1, cell)):
            row_shift = shift if (j % 2 == 1) else 0
            for xx in range(x0 + row_shift, x1, cell):
                draw.ellipse([xx - rad, yy - rad, xx + rad, yy + rad], fill=color)
        return img


# ============================== Registration ==============================

REGISTRY.register("star", DrawStarBlock)
REGISTRY.register("flower", DrawFlowerBlock)
REGISTRY.register("ngon", DrawNGonBlock)
REGISTRY.register("polygon", DrawPolygonBlock)
REGISTRY.register("heart", DrawHeartBlock)
REGISTRY.register("ring", DrawRingBlock)
REGISTRY.register("arrow", DrawArrowBlock)
REGISTRY.register("grid", DrawGridBlock)
REGISTRY.register("spiral", DrawSpiralBlock)
REGISTRY.register("sunburst", DrawSunburstBlock)
REGISTRY.register("bezier", DrawBezierBlock)
REGISTRY.register("wave", DrawWaveBlock)
REGISTRY.register("dots", DrawDotsBlock)
