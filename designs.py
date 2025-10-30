# designs.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageDraw

# Import from your existing graphics module
from graphics import BaseBlock, REGISTRY, _ensure_image, _parse_color


# ============================== Small helpers ==============================

def _px_or_norm(params: Dict[str, Any], key: str, size_px: int, default_norm: float) -> float:
    """
    Fetch a coordinate/length that may be given as absolute pixels (key+"_px")
    or normalized 0..1 (key). Returns float pixels.
    """
    if f"{key}_px" in params:
        return float(params[f"{key}_px"])
    return float(params.get(key, default_norm)) * float(size_px)

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
    out = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        xr = dx * c - dy * s
        yr = dx * s + dy * c
        out.append((cx + xr, cy + yr))
    return out


# ============================== Blocks ==============================

# --- Star Block (original with minor polish) ---
@dataclass
class DrawStarBlock(BaseBlock):
    """Draw a star shape (outer/inner radii)."""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        outer_r = float(params.get("outer_radius", 0.2)) * min(width, height)
        inner_r = float(params.get("inner_radius", 0.1)) * min(width, height)
        points = max(3, int(params.get("points", 5)))
        rot = float(params.get("rotation", 0.0))

        fill_color = _parse_color(params.get("fill", "yellow"), (255, 255, 0, 255))
        outline_color = _parse_color(params.get("outline", None), None)
        outline_width = int(params.get("outline_width", 1))

        pts = _calc_star_points(cx, cy, outer_r, inner_r, points)
        pts = _rotate_points(pts, cx, cy, rot)

        draw.polygon(
            pts,
            fill=fill_color if fill_color else None,
            outline=outline_color if outline_color else None,
            width=outline_width
        )
        return img


# --- Flower Block (original) ---
@dataclass
class DrawFlowerBlock(BaseBlock):
    """Draws a simple flower with circular petals around a center."""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        petal_r_rel = float(params.get("petal_radius", 0.15))
        center_r_rel = float(params.get("center_radius", 0.05))
        petal_r = petal_r_rel * min(width, height)
        center_r = center_r_rel * min(width, height)
        num_petals = max(3, int(params.get("petals", 6)))
        rot = float(params.get("rotation", 0.0))

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


# --- Regular N-Gon ---
@dataclass
class DrawNGonBlock(BaseBlock):
    """
    Draw a regular polygon.
    Params: cx, cy, radius, sides>=3, rotation (deg), fill, outline, outline_width
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        r = float(params.get("radius", 0.25)) * min(width, height)
        sides = max(3, int(params.get("sides", 6)))
        rot = float(params.get("rotation", -90.0))  # point-up default

        fill_color = _parse_color(params.get("fill", None), None)
        outline_color = _parse_color(params.get("outline", "white"), (255, 255, 255, 255))
        ow = int(params.get("outline_width", 2))

        pts = []
        for i in range(sides):
            a = 2.0 * math.pi * i / sides + math.radians(rot)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))

        draw.polygon(pts, fill=fill_color if fill_color else None,
                     outline=outline_color if outline_color else None, width=ow)
        return img


# --- Arbitrary Polygon (list of points) ---
@dataclass
class DrawPolygonBlock(BaseBlock):
    """
    Draw a polygon from a list of points.
    Params:
      points: list like "x1,y1;x2,y2;..." (normalized 0..1 unless *_px provided)
      offset_x, offset_y (norm), rotation (deg), scale (relative)
      fill, outline, outline_width
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        pts_str = str(params.get("points", "0.2,0.2;0.8,0.2;0.5,0.8"))
        # Parse "x,y" pairs; treat as normalized
        raw_pts: List[Tuple[float, float]] = []
        for token in pts_str.split(";"):
            token = token.strip()
            if not token or "," not in token:
                continue
            sx, sy = token.split(",", 1)
            try:
                raw_pts.append((float(sx), float(sy)))
            except Exception:
                pass

        if not raw_pts:
            return img

        offset_x = float(params.get("offset_x", 0.0)) * width
        offset_y = float(params.get("offset_y", 0.0)) * height
        scale = float(params.get("scale", 1.0))
        rot = float(params.get("rotation", 0.0))

        # Convert to pixels, apply scale about centroid, rotate
        px_pts = [(x * width, y * height) for x, y in raw_pts]
        cx = sum(p[0] for p in px_pts) / len(px_pts)
        cy = sum(p[1] for p in px_pts) / len(px_pts)
        px_pts = [((x - cx) * scale + cx + offset_x, (y - cy) * scale + cy + offset_y) for x, y in px_pts]
        px_pts = _rotate_points(px_pts, cx + offset_x, cy + offset_y, rot)

        fill_color = _parse_color(params.get("fill", None), None)
        outline_color = _parse_color(params.get("outline", "white"), (255, 255, 255, 255))
        ow = int(params.get("outline_width", 2))

        draw.polygon(px_pts, fill=fill_color if fill_color else None,
                     outline=outline_color if outline_color else None, width=ow)
        return img


# --- Heart (parametric) ---
@dataclass
class DrawHeartBlock(BaseBlock):
    """
    Draws a heart using a parametric curve, fitted inside a box.
    Params: cx, cy, size (0..1), rotation, fill, outline, outline_width
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        size = float(params.get("size", 0.35)) * min(width, height)
        rot = float(params.get("rotation", 0.0))
        steps = max(64, int(params.get("steps", 200)))

        fill_color = _parse_color(params.get("fill", "red"), (255, 0, 0, 255))
        outline_color = _parse_color(params.get("outline", "white"), (255, 255, 255, 255))
        ow = int(params.get("outline_width", 2))

        pts = []
        for i in range(steps):
            t = 2.0 * math.pi * i / steps
            # Classic heart curve (scaled tweak)
            x = 16 * (math.sin(t) ** 3)
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            x *= size / 18.0
            y *= -size / 18.0  # flip Y
            pts.append((cx + x, cy + y))

        pts = _rotate_points(pts, cx, cy, rot)

        if fill_color:
            draw.polygon(pts, fill=fill_color)
        if outline_color and ow > 0:
            draw.line(pts + [pts[0]], fill=outline_color, width=ow, joint="curve")
        return img


# --- Ring / Annulus ---
@dataclass
class DrawRingBlock(BaseBlock):
    """
    Draws a ring (annulus).
    Params: cx, cy, outer_radius, inner_radius, fill (inner), ring_color, ring_width
    If inner_radius<=0: behaves like a circle outline with ring_width.
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        r_out = float(params.get("outer_radius", 0.3)) * min(width, height)
        r_in = float(params.get("inner_radius", 0.2)) * min(width, height)

        ring_color = _parse_color(params.get("ring_color", "white"), (255, 255, 255, 255))
        fill_color = _parse_color(params.get("fill", None), None)
        ring_width = int(params.get("ring_width", max(1, int(r_out - r_in))))

        # Outer
        bbox_out = [cx - r_out, cy - r_out, cx + r_out, cy + r_out]
        if fill_color:
            draw.ellipse(bbox_out, fill=fill_color)
        # Inner punch-out via drawing a filled ellipse with transparent? PIL's ImageDraw has no compositing.
        # Simplify: draw just the ring as an outline width approximating (r_out - r_in):
        draw.ellipse(bbox_out, outline=ring_color, width=ring_width)
        return img


# --- Arrow (triangle head + shaft) ---
@dataclass
class DrawArrowBlock(BaseBlock):
    """
    Draw a simple arrow from (x0,y0) to (x1,y1).
    Params: x0,y0,x1,y1 (norm or *_px), shaft_width_px, head_len_px, head_width_px, color
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        x0 = _px_or_norm(params, "x0", width, 0.2)
        y0 = _px_or_norm(params, "y0", height, 0.5)
        x1 = _px_or_norm(params, "x1", width, 0.8)
        y1 = _px_or_norm(params, "y1", height, 0.5)

        color = _parse_color(params.get("color", "white"), (255, 255, 255, 255))
        shaft_w = float(params.get("shaft_width_px", 6.0))
        head_len = float(params.get("head_len_px", 30.0))
        head_w = float(params.get("head_width_px", 24.0))

        dx, dy = x1 - x0, y1 - y0
        L = max(1e-6, math.hypot(dx, dy))
        ux, uy = dx / L, dy / L  # direction
        # Perp
        px, py = -uy, ux

        # Shaft end before head
        xe = x1 - ux * head_len
        ye = y1 - uy * head_len

        # Shaft rectangle
        p1 = (x0 + px * shaft_w * 0.5, y0 + py * shaft_w * 0.5)
        p2 = (x0 - px * shaft_w * 0.5, y0 - py * shaft_w * 0.5)
        p3 = (xe - px * shaft_w * 0.5, ye - py * shaft_w * 0.5)
        p4 = (xe + px * shaft_w * 0.5, ye + py * shaft_w * 0.5)
        draw.polygon([p1, p2, p3, p4], fill=color)

        # Arrow head (triangle)
        h1 = (xe + px * head_w * 0.5, ye + py * head_w * 0.5)
        h2 = (xe - px * head_w * 0.5, ye - py * head_w * 0.5)
        h3 = (x1, y1)
        draw.polygon([h1, h2, h3], fill=color)
        return img


# --- Grid (rectangular) ---
@dataclass
class DrawGridBlock(BaseBlock):
    """
    Draw a grid overlay.
    Params: rows, cols, color, width_px, margin_x, margin_y (norm)
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        rows = max(1, int(params.get("rows", 4)))
        cols = max(1, int(params.get("cols", 6)))
        color = _parse_color(params.get("color", "white"), (255, 255, 255, 120))
        wpx = int(params.get("width_px", 1))
        mx = float(params.get("margin_x", 0.05)) * width
        my = float(params.get("margin_y", 0.05)) * height

        x0, y0 = mx, my
        x1, y1 = width - mx, height - my

        # Outer rect
        draw.rectangle([x0, y0, x1, y1], outline=color, width=wpx)

        # Internal lines
        for r in range(1, rows):
            yy = y0 + (y1 - y0) * r / rows
            draw.line([(x0, yy), (x1, yy)], fill=color, width=wpx)
        for c in range(1, cols):
            xx = x0 + (x1 - x0) * c / cols
            draw.line([(xx, y0), (xx, y1)], fill=color, width=wpx)
        return img


# --- Spiral (Archimedean) ---
@dataclass
class DrawSpiralBlock(BaseBlock):
    """
    Draw an Archimedean spiral.
    Params: cx, cy, turns, radius, thickness_px, color, rotation
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        turns = max(0.1, float(params.get("turns", 3.0)))
        radius = float(params.get("radius", 0.45)) * min(width, height)
        rot = math.radians(float(params.get("rotation", -90.0)))
        thickness = int(params.get("thickness_px", 3))
        color = _parse_color(params.get("color", "white"), (255, 255, 255, 255))

        steps = int(params.get("steps", 800))
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


# --- Sunburst (radial wedges) ---
@dataclass
class DrawSunburstBlock(BaseBlock):
    """
    Draw radial wedges around a center.
    Params: cx, cy, inner_radius, outer_radius, wedges, start_deg, fill_a, fill_b (alternating)
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.5)
        r0 = float(params.get("inner_radius", 0.1)) * min(width, height)
        r1 = float(params.get("outer_radius", 0.48)) * min(width, height)
        wedges = max(1, int(params.get("wedges", 16)))
        start_deg = float(params.get("start_deg", -90.0))

        fa = _parse_color(params.get("fill_a", "#FFD166"), (255, 209, 102, 255))
        fb = _parse_color(params.get("fill_b", "#EF476F"), (239, 71, 111, 255))

        for i in range(wedges):
            a0 = math.radians(start_deg + (360.0 / wedges) * i)
            a1 = math.radians(start_deg + (360.0 / wedges) * (i + 1))
            # Build wedge polygon
            pts = [(cx + r0 * math.cos(a0), cy + r0 * math.sin(a0)),
                   (cx + r1 * math.cos(a0), cy + r1 * math.sin(a0)),
                   (cx + r1 * math.cos(a1), cy + r1 * math.sin(a1)),
                   (cx + r0 * math.cos(a1), cy + r0 * math.sin(a1))]
            draw.polygon(pts, fill=fa if (i % 2 == 0) else fb)
        return img


# --- Quadratic Bezier curve (polyline approximation) ---
@dataclass
class DrawBezierBlock(BaseBlock):
    """
    Quadratic Bezier curve.
    Params: x0,y0, x1,y1, cx,cy (control); thickness_px, color, steps
    (All coords normalized unless *_px provided.)
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        x0 = _px_or_norm(params, "x0", width, 0.2)
        y0 = _px_or_norm(params, "y0", height, 0.8)
        x1 = _px_or_norm(params, "x1", width, 0.8)
        y1 = _px_or_norm(params, "y1", height, 0.8)
        cx = _px_or_norm(params, "cx", width, 0.5)
        cy = _px_or_norm(params, "cy", height, 0.2)

        steps = max(8, int(params.get("steps", 200)))
        thickness = int(params.get("thickness_px", 3))
        color = _parse_color(params.get("color", "white"), (255, 255, 255, 255))

        pts = []
        for i in range(steps + 1):
            t = i / steps
            u = 1 - t
            x = u*u*x0 + 2*u*t*cx + t*t*x1
            y = u*u*y0 + 2*u*t*cy + t*t*y1
            pts.append((x, y))
        draw.line(pts, fill=color, width=thickness, joint="curve")
        return img


# --- Sine Wave (polyline) ---
@dataclass
class DrawWaveBlock(BaseBlock):
    """
    Horizontal sine wave.
    Params: amp (px), freq (cycles across width), phase_deg, thickness_px, color, y (norm)
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        amp = float(params.get("amp_px", 20.0))
        freq = float(params.get("freq", 3.0))
        phase = math.radians(float(params.get("phase_deg", 0.0)))
        y = _px_or_norm(params, "y", height, 0.5)
        thickness = int(params.get("thickness_px", 3))
        color = _parse_color(params.get("color", "white"), (255, 255, 255, 255))
        steps = max(64, int(params.get("steps", 500)))

        pts = []
        for i in range(steps + 1):
            t = i / steps
            xx = t * (width - 1)
            yy = y + amp * math.sin(2 * math.pi * freq * t + phase)
            pts.append((xx, yy))
        draw.line(pts, fill=color, width=thickness, joint="curve")
        return img


# --- Dot Pattern (staggered) ---
@dataclass
class DrawDotsBlock(BaseBlock):
    """
    Staggered dot pattern.
    Params: cell_px, radius_px, color, offset (0..1), margin_x, margin_y (norm)
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        draw = ImageDraw.Draw(img)

        cell = max(2, int(params.get("cell_px", 24)))
        rad = max(1, int(params.get("radius_px", 4)))
        color = _parse_color(params.get("color", "white"), (255, 255, 255, 180))
        mx = float(params.get("margin_x", 0.05)) * width
        my = float(params.get("margin_y", 0.05)) * height
        offset = float(params.get("offset", 0.5))  # horizontal stagger [0..1]

        x0, y0 = int(mx), int(my)
        x1, y1 = int(width - mx), int(height - my)
        shift = int(cell * offset)

        for j, y in enumerate(range(y0, y1, cell)):
            row_shift = shift if (j % 2 == 1) else 0
            for x in range(x0 + row_shift, x1, cell):
                draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=color)
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
