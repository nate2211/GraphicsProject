# camera_advanced_v3_5.py — CameraPipeline v3.5 (cinema‑grade, animation‑aware, look/move rig)
# -----------------------------------------------------------------------------------
# New in v3.5 (drop‑in, backwards compatible):
# - Camera "rig" that can LOOK and MOVE around the scene with intuitive film terms
# - Modes: direct (legacy), target (look‑at), orbit, and path (waypoints)
# - Controls: look_yaw_deg (alias rotation), look_tilt_{x,y} (alias tilt), dolly, truck_x/y
# - Targeting on canvas or detected content bbox; composition pivot alignment retained
# - Simple path follower with normalized (0..1) points; clamp/loop/pingpong play modes
# - Orbit around a point (normalized) with optional face_target behavior
# - Dolly scales zoom; truck shifts camera after auto-fit and clamping
# -----------------------------------------------------------------------------------
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from PIL import Image, ImageFilter

# Graphics glue
from graphics import BaseBlock, REGISTRY, _ensure_image, _parse_color
# Animation glue (safe to import; your repo provides these)
from animations import ANIMATION_REGISTRY, AnimationContext

# =============================================================================
# Small helpers
# =============================================================================

def _norm01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _clamp(x: float, a: float, b: float) -> float:
    return max(a, min(b, x))


def _rng(seed: Optional[int]) -> random.Random:
    r = random.Random()
    if seed is not None:
        r.seed(int(seed) & 0xFFFFFFFF)
    return r


def _to_rgba(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGBA" else img.convert("RGBA")


def _content_bbox_alpha(img: Image.Image, *, alpha_thresh: int = 1) -> Optional[Tuple[int, int, int, int]]:
    a = _to_rgba(img).getchannel("A")
    if alpha_thresh <= 0:
        return a.getbbox()
    mask = a.point(lambda v: 255 if v > alpha_thresh else 0, mode="L")
    return mask.getbbox()


def _content_bbox_luma(img: Image.Image, *, thresh: float = 0.02) -> Optional[Tuple[int, int, int, int]]:
    arr = np.asarray(_to_rgba(img)).astype(np.float32) / 255.0
    lum = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    m = (lum > float(thresh)).astype(np.uint8) * 255
    pil = Image.fromarray(m, "L")
    return pil.getbbox()


def _dilate_bbox(b: Tuple[int, int, int, int], pad_px: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = b
    return (
        max(0, x0 - pad_px),
        max(0, y0 - pad_px),
        min(W, x1 + pad_px),
        min(H, y1 + pad_px),
    )

# ------------------------ Animation‑aware sub‑pipeline ------------------------

def _collect_prefixed(extras: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    pfx = f"{prefix}."
    for k, v in extras.items():
        if isinstance(k, str) and k.startswith(pfx):
            out[k[len(pfx):]] = v
    return out


def _sub_params_for(extras: Dict[str, Any], raw: str, idx1: int) -> Dict[str, Any]:
    p: Dict[str, Any] = {}
    v = extras.get(raw)
    if isinstance(v, dict):
        p.update(v)
    p.update(_collect_prefixed(extras, raw))
    p.update(_collect_prefixed(extras, str(idx1)))
    p.update(_collect_prefixed(extras, str(idx1 - 1)))
    v0 = extras.get(str(idx1 - 1))
    if isinstance(v0, dict):
        p.update(v0)
    v1 = extras.get(str(idx1))
    if isinstance(v1, dict):
        p.update(v1)
    return p


def _anim_ctx_from(anim_dict: Dict[str, Any], W: int, H: int) -> AnimationContext:
    a = anim_dict or {}
    return AnimationContext(
        frame=int(a.get("frame", 0)),
        total_frames=int(a.get("total_frames", 1)),
        time=float(a.get("time", 0.0)),
        duration=float(a.get("duration", 1.0)),
        fps=float(a.get("fps", 30.0)),
        width=int(a.get("width", W)),
        height=int(a.get("height", H)),
    )


def _run_sub_pipeline(pipeline: str, width: int, height: int, extras: Dict[str, Dict[str, Any]]) -> Image.Image:
    img: Optional[Image.Image] = None
    names = [s.strip().lower() for s in (pipeline or "").split("|") if s.strip()]
    anim_info = extras.get("__anim", {}) if isinstance(extras, dict) else {}
    ctx = _anim_ctx_from(anim_info, width, height)

    for idx1, raw in enumerate(names, start=1):
        params = _sub_params_for(extras, raw, idx1)
        if raw in ANIMATION_REGISTRY.names():
            anim_blk = ANIMATION_REGISTRY.create(raw)  # type: ignore
            img = anim_blk.process_frame(img, ctx, params, engine=None)
        else:
            blk = REGISTRY.create(raw)
            img = blk.process(img, width, height, params=params)
        if img is None:
            raise RuntimeError(f"Sub-pipeline block '{raw}' returned None")
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        if img.size != (width, height):
            img = img.resize((width, height), Image.Resampling.LANCZOS)
    return _ensure_image(img, width, height)

# ------------------------------ Backgrounds ------------------------------

def _make_background(width: int, height: int, kind: str,
                     c0: Tuple[int, int, int, int], c1: Tuple[int, int, int, int]) -> Image.Image:
    kind = (kind or "solid").lower()
    if kind == "solid":
        return Image.new("RGBA", (width, height), c0)
    if kind == "vgrad":
        top = np.array(c0, dtype=np.float32)
        bot = np.array(c1, dtype=np.float32)
        t = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        arr = (top * (1 - t) + bot * t).astype(np.uint8)
        arr = np.repeat(arr[None, :, :], width, axis=0).transpose(1, 0, 2)
        return Image.fromarray(arr, "RGBA")
    if kind == "rgrad":
        cx, cy = width * 0.5, height * 0.5
        yy, xx = np.indices((height, width), dtype=np.float32)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r /= (0.5 * math.hypot(width, height))
        r = np.clip(r, 0, 1)[..., None]
        c0a = np.array(c0, dtype=np.float32)
        c1a = np.array(c1, dtype=np.float32)
        arr = (c0a * (1 - r) + c1a * r).astype(np.uint8)
        return Image.fromarray(arr, "RGBA")
    return Image.new("RGBA", (width, height), c0)

# ------------------------------ Lens & Post ------------------------------

def _apply_perspective_hint(img: Image.Image, tilt_x: float, tilt_y: float) -> Image.Image:
    if abs(tilt_x) < 1e-4 and abs(tilt_y) < 1e-4:
        return img
    w, h = img.size
    ix = _clamp(float(tilt_x), -1, 1) * 0.12 * w
    iy = _clamp(float(tilt_y), -1, 1) * 0.12 * h
    dst = [
        (0 + ix, 0 + iy),
        (w - 1 - ix, 0 + iy),
        (w - 1 - ix, h - 1 - iy),
        (0 + ix, h - 1 - iy),
    ]
    return img.transform((w, h), Image.Transform.QUAD, data=sum(dst, ()), resample=Image.Resampling.BICUBIC)


def _compose_affine(img: Image.Image, *, scale: float, rot_deg: float,
                    trans: Tuple[float, float], center: Tuple[float, float]) -> Image.Image:
    cx, cy = center
    theta = math.radians(rot_deg)
    s = max(1e-6, float(scale))
    c, si = math.cos(theta), math.sin(theta)
    a = s * c
    b = -s * si
    d = s * si
    e = s * c
    tx, ty = trans
    c0 = -a * cx - b * cy + cx + tx
    f0 = -d * cx - e * cy + cy + ty
    det = a * e - b * d
    if abs(det) < 1e-12:
        det = 1e-12
    ia = e / det
    ib = -b / det
    id_ = -d / det
    ie = a / det
    ic = -(ia * c0 + ib * f0)
    if_ = -(id_ * c0 + ie * f0)
    return img.transform(img.size, Image.Transform.AFFINE,
                         (ia, ib, ic, id_, ie, if_), resample=Image.Resampling.BICUBIC)


def _lens_distort(img: Image.Image, k1: float, k2: float, center: Tuple[float, float]) -> Image.Image:
    if abs(k1) < 1e-6 and abs(k2) < 1e-6:
        return img
    w, h = img.size
    cx, cy = center
    arr = np.asarray(_to_rgba(img)).astype(np.float32)
    yy, xx = np.indices((h, w), dtype=np.float32)
    smax = float(max(w, h))
    x = (xx - cx) / smax
    y = (yy - cy) / smax
    r2 = x * x + y * y
    factor = 1 + k1 * r2 + k2 * r2 * r2
    u = cx + x * factor * smax
    v = cy + y * factor * smax
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)
    wa = (x1 - u) * (y1 - v)
    wb = (u - x0) * (y1 - v)
    wc = (x1 - u) * (v - y0)
    wd = (u - x0) * (v - y0)
    out = (
        arr[y0, x0] * wa[..., None] +
        arr[y0, x1] * wb[..., None] +
        arr[y1, x0] * wc[..., None] +
        arr[y1, x1] * wd[..., None]
    )
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def _chroma_ab(img: Image.Image, shift_x: float, shift_y: float) -> Image.Image:
    if abs(shift_x) <= 1e-6 and abs(shift_y) <= 1e-6:
        return img
    w, h = img.size
    sx = int(round(shift_x))
    sy = int(round(shift_y))
    arr = np.asarray(_to_rgba(img))
    r = np.roll(arr[..., 0], shift=sx, axis=1)
    r = np.roll(r, shift=sy, axis=0)
    b = np.roll(arr[..., 2], shift=-sx, axis=1)
    b = np.roll(b, shift=-sy, axis=0)
    out = arr.copy()
    out[..., 0] = r
    out[..., 2] = b
    return Image.fromarray(out, "RGBA")


def _bloom(img: Image.Image, threshold: float, radius: float, intensity: float) -> Image.Image:
    if intensity <= 1e-6 or radius <= 0:
        return img
    im = _to_rgba(img)
    arr = np.asarray(im).astype(np.float32) / 255.0
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mask = np.clip((lum - threshold) / max(1e-6, 1 - threshold), 0, 1)
    glow = arr[..., :3] * mask[..., None]
    glow_img = Image.fromarray(np.clip(glow * 255.0, 0, 255).astype(np.uint8), "RGB")
    blur = glow_img.filter(ImageFilter.GaussianBlur(radius=float(radius)))
    blur_arr = np.asarray(blur).astype(np.float32) / 255.0
    arr[..., :3] = np.clip(arr[..., :3] + blur_arr * intensity, 0, 1)
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "RGBA")


def _tone_map(img: Image.Image, exposure: float, contrast: float,
             wb: Tuple[float, float, float], lift: float, gamma: float, gain: float) -> Image.Image:
    arr = np.asarray(_to_rgba(img)).astype(np.float32) / 255.0
    e = 2.0 ** exposure
    arr[..., :3] *= e
    arr[..., 0] *= wb[0]
    arr[..., 1] *= wb[1]
    arr[..., 2] *= wb[2]
    arr[..., :3] = arr[..., :3] / (1.0 + arr[..., :3])
    if abs(lift) > 1e-5:
        arr[..., :3] = np.clip(arr[..., :3] + lift, 0, 1)
    if abs(gamma - 1.0) > 1e-5:
        arr[..., :3] = np.clip(arr[..., :3], 1e-6, 1.0) ** (1.0 / gamma)
    if abs(gain - 1.0) > 1e-5:
        arr[..., :3] = np.clip(arr[..., :3] * gain, 0, 1)
    c = _clamp(contrast, -1, 1)
    if abs(c) > 1e-3:
        mid = 0.5
        arr[..., :3] = np.clip((arr[..., :3] - mid) * (1 + 2.4 * c) + mid, 0, 1)
    return Image.fromarray((arr * 255.0).astype(np.uint8), "RGBA")


def _vignette(img: Image.Image, strength: float) -> Image.Image:
    if strength <= 1e-6:
        return img
    w, h = img.size
    cx, cy = w * 0.5, h * 0.5
    yy, xx = np.indices((h, w), dtype=np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r = np.clip(r / (0.72 * math.hypot(cx, cy)), 0, 1)
    ramp = (1 - r) ** (1.5 + 2.5 * _norm01(strength))
    arr = np.asarray(_to_rgba(img)).astype(np.float32) / 255.0
    arr[..., :3] *= ramp[..., None]
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "RGBA")


def _tilt_shift(img: Image.Image, band_y: float, band_h: float, blur_radius: float) -> Image.Image:
    if blur_radius <= 1e-6 or band_h <= 1e-6:
        return img
    h = img.size[1]
    im = _to_rgba(img)
    blurred = im.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    center = _clamp(band_y, 0, 1)
    half = max(1e-3, band_h * 0.5)
    d = np.abs(y - center)
    t = np.clip((d - half) / max(1e-3, (0.5 - half)), 0, 1)
    mix = (1 - np.cos(t * math.pi)) * 0.5
    base = np.asarray(im).astype(np.float32)
    bl = np.asarray(blurred).astype(np.float32)
    out = base * (1 - mix[..., None]) + bl * mix[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def _grain(img: Image.Image, amount: float, seed: Optional[int]) -> Image.Image:
    if amount <= 1e-6:
        return img
    w, h = img.size
    r = _rng(seed)
    noise = (np.random.RandomState(r.randint(0, 1 << 30)).randn(h, w, 1).astype(np.float32))
    arr = np.asarray(_to_rgba(img)).astype(np.float32) / 255.0
    sigma = amount * 0.10
    arr[..., :3] = np.clip(arr[..., :3] + sigma * noise, 0, 1)
    return Image.fromarray((arr * 255.0).astype(np.uint8), "RGBA")


def _rolling_shutter(img: Image.Image, px_perc: float) -> Image.Image:
    if abs(px_perc) < 1e-6:
        return img
    arr = np.asarray(_to_rgba(img))
    h, w = arr.shape[0], arr.shape[1]
    max_shift = int(round(px_perc * w))
    out = np.empty_like(arr)
    for y in range(h):
        s = int(round((y / max(1, h - 1)) * max_shift))
        out[y] = np.roll(arr[y], s, axis=0)
    return Image.fromarray(out, "RGBA")


def _letterbox(img: Image.Image, aspect: float, bg: Tuple[int, int, int, int]) -> Image.Image:
    if aspect <= 0:
        return img
    w, h = img.size
    cur = w / max(1, h)
    if abs(cur - aspect) < 1e-6:
        return img
    if cur > aspect:
        new_h = int(round(w / aspect))
        canvas = Image.new("RGBA", (w, new_h), bg)
        y = (new_h - h) // 2
        canvas.paste(img, (0, y))
        return canvas
    else:
        new_w = int(round(h * aspect))
        canvas = Image.new("RGBA", (new_w, h), bg)
        x = (new_w - w) // 2
        canvas.paste(img, (x, 0))
        return canvas

# ------------------------------ Rig helpers ------------------------------

def _parse_points(s: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    if not s:
        return pts
    for part in str(s).split(";"):
        part = part.strip()
        if not part:
            continue
        xy = part.split(",")
        if len(xy) != 2:
            continue
        try:
            x = _norm01(float(xy[0]))
            y = _norm01(float(xy[1]))
            pts.append((x, y))
        except Exception:
            pass
    return pts


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _polyline_eval(points: List[Tuple[float, float]], t: float, mode: str) -> Tuple[float, float]:
    if not points:
        return 0.5, 0.5
    n = len(points)
    if n == 1:
        return points[0]
    mode_l = (mode or "clamp").lower()
    # Build segment count and handle looping
    if mode_l == "loop":
        total_segs = n
        t = t % 1.0
        seg_f = t * total_segs
        i0 = int(math.floor(seg_f)) % n
        i1 = (i0 + 1) % n
        lt = seg_f - math.floor(seg_f)
        x = _lerp(points[i0][0], points[i1][0], lt)
        y = _lerp(points[i0][1], points[i1][1], lt)
        return x, y
    if mode_l == "pingpong":
        t2 = t % 2.0
        if t2 > 1.0:
            t2 = 2.0 - t2
        t = t2
    # clamp (or pingpong after folding) on n-1 segments
    total_segs = max(1, n - 1)
    seg_f = _clamp(t, 0.0, 1.0) * total_segs
    i0 = int(math.floor(seg_f))
    i1 = min(i0 + 1, n - 1)
    lt = seg_f - math.floor(seg_f)
    x = _lerp(points[i0][0], points[i1][0], lt)
    y = _lerp(points[i0][1], points[i1][1], lt)
    return x, y

# =============================================================================
# Camera block (with look/move rig)
# =============================================================================
@dataclass
class CameraPipeline(BaseBlock):
    """
    Cinematic view transform & post. **Backwards compatible** with v1/v2/v3.

    New rig parameters (all optional):
      cam_mode              : direct|target|orbit|path        [direct]
      # target mode
      cam_target_x/y        : float 0..1 (normalized)         [0.5, 0.5]
      cam_target_space      : canvas|content                   [canvas]
      face_target           : bool (rotate so target stays up) [False]
      # orbit mode
      orbit_center_x/y      : float 0..1                       [0.5, 0.5]
      orbit_radius_px       : float px                         [0]
      orbit_angle_deg       : float deg                        [0]
      orbit_face_target     : bool                             [True]
      # path mode
      path_points           : "x,y; x,y; ..." (normalized)
      path_t                : float 0..1                       [0]
      path_mode             : clamp|loop|pingpong              [clamp]
      # common movement
      truck_x_px/y_px       : float px                         [0, 0]
      dolly                 : float (zoom factor ~ 2**dolly)   [0]
      # look controls (aliases to legacy)
      look_yaw_deg          : float (alias rotation)
      look_tilt_x, _y       : float -1..1 (aliases tilt_x/y)
    """

    def process(self, img, width, height, *, params):
        W, H = width, height

        # -------------------- Background --------------------
        bg_kind = str(params.get("background_kind", "solid"))
        bg0 = _parse_color(params.get("background_color0", "#00000000"), (0, 0, 0, 0)) or (0, 0, 0, 0)
        bg1 = _parse_color(params.get("background_color1", "#00000000"), (0, 0, 0, 0)) or (0, 0, 0, 0)
        canvas = _make_background(W, H, bg_kind, bg0, bg1)

        # -------------------- Prepare design --------------------
        pre = str(params.get("pre_pipeline", "") or "")
        design_pipe = str(params.get("design_pipeline", "") or "")
        sub_extras: Dict[str, Any] = dict(params)

        if design_pipe:
            design = _run_sub_pipeline(design_pipe, W, H, sub_extras)
        else:
            design = _ensure_image(img, W, H)
            if pre:
                pre_extras = dict(sub_extras)
                design = _run_sub_pipeline(pre, W, H, pre_extras)
        design = _to_rgba(design)

        # -------------------- Core camera params (legacy + aliases) --------------------
        pan_x = float(params.get("pan_x", 0.0))
        pan_y = float(params.get("pan_y", 0.0))
        zoom = float(params.get("zoom", 1.0))
        rotation = float(params.get("rotation", params.get("look_yaw_deg", 0.0)))

        auto_fit = bool(params.get("auto_fit", True))
        fit_mode = str(params.get("fit_mode", "contain")).lower()
        pad_frac = _clamp(float(params.get("auto_fit_padding", 0.08)), 0.0, 0.45)
        athresh = int(params.get("track_alpha_thresh", 2))
        bthresh = float(params.get("track_brightness_thresh", 0.02))
        bbox_pad = max(0, int(params.get("bbox_pad_px", 4)))

        clamp_pan = bool(params.get("clamp_pan", True))
        safe_margin = _clamp(float(params.get("safe_margin", 0.05)), 0.0, 0.4)
        comp_px = _clamp(float(params.get("composition_pivot_x", 0.5)), 0.0, 1.0)
        comp_py = _clamp(float(params.get("composition_pivot_y", 0.5)), 0.0, 1.0)

        tilt_x = float(params.get("tilt_x", params.get("look_tilt_x", 0.0)))
        tilt_y = float(params.get("tilt_y", params.get("look_tilt_y", 0.0)))
        focal_len = max(0.2, float(params.get("focal_len", 1.0)))

        # -------------------- Auto‑fit scale --------------------
        eff_zoom = zoom
        bbox = _content_bbox_alpha(design, alpha_thresh=athresh) if auto_fit else None
        if auto_fit and not bbox:
            bbox = _content_bbox_luma(design, thresh=bthresh)

        if auto_fit:
            if bbox:
                x0, y0, x1, y1 = _dilate_bbox(bbox, bbox_pad, W, H)
                bw = max(1, x1 - x0)
                bh = max(1, y1 - y0)
                pad_x = pad_frac * W
                pad_y = pad_frac * H
                vW = max(1, W - 2 * pad_x)
                vH = max(1, H - 2 * pad_y)
                if fit_mode == "fill":
                    s_w = vW / bw
                    s_h = vH / bh
                    s_auto = (s_w + s_h) * 0.5
                elif fit_mode == "cover":
                    s_auto = max(vW / bw, vH / bh)
                else:  # contain
                    s_auto = min(vW / bw, vH / bh)
                eff_zoom = min(eff_zoom, s_auto)
            else:
                eff_zoom = min(eff_zoom, 0.95)

        # -------------------- Pan clamping & composition pivot (baseline) --------------------
        pan_scale = 0.33 / focal_len
        tx = pan_x * W * pan_scale
        ty = pan_y * H * pan_scale

        if auto_fit and clamp_pan and bbox:
            x0, y0, x1, y1 = bbox
            bw = max(1, x1 - x0)
            bh = max(1, y1 - y0)
            vw = bw * eff_zoom
            vh = bh * eff_zoom
            cx_target = (x0 + comp_px * bw) * eff_zoom
            cy_target = (y0 + comp_py * bh) * eff_zoom
            sx = W * 0.5
            sy = H * 0.5
            tx += (sx - cx_target)
            ty += (sy - cy_target)
            mx = max(0.0, (W - vw) * 0.5 - safe_margin * W)
            my = max(0.0, (H - vh) * 0.5 - safe_margin * H)
            tx = _clamp(tx, -mx, mx)
            ty = _clamp(ty, -my, my)

        # -------------------- RIG: look & move augmentations --------------------
        cam_mode = str(params.get("cam_mode", "direct")).lower()
        truck_x = float(params.get("truck_x_px", 0.0))
        truck_y = float(params.get("truck_y_px", 0.0))
        dolly = float(params.get("dolly", 0.0))  # zoom *= 2**dolly later

        # Target mode: look/move to put (cam_target_x,y) at screen center.
        if cam_mode == "target":
            tspace = str(params.get("cam_target_space", "canvas")).lower()
            tx0, ty0 = tx, ty
            if tspace == "content" and bbox:
                x0, y0, x1, y1 = bbox
                bw = max(1, x1 - x0)
                bh = max(1, y1 - y0)
                gx = x0 + _norm01(float(params.get("cam_target_x", 0.5))) * bw
                gy = y0 + _norm01(float(params.get("cam_target_y", 0.5))) * bh
            else:
                gx = _norm01(float(params.get("cam_target_x", 0.5))) * W
                gy = _norm01(float(params.get("cam_target_y", 0.5))) * H
            sx = W * 0.5
            sy = H * 0.5
            # move so (gx,gy) aligns with screen center (sx,sy)
            tx += (sx - gx * eff_zoom)
            ty += (sy - gy * eff_zoom)
            if bool(params.get("face_target", False)):
                # keep rotation minimal in 2D; alias to existing rotation
                rotation = rotation  # (placeholder — caller can animate look_yaw_deg)

        # Orbit mode: circular trucking around a center; optional face target
        elif cam_mode == "orbit":
            ocx = _norm01(float(params.get("orbit_center_x", 0.5))) * W
            ocy = _norm01(float(params.get("orbit_center_y", 0.5))) * H
            radius = float(params.get("orbit_radius_px", 0.0))
            angle = math.radians(float(params.get("orbit_angle_deg", 0.0)))
            tx += radius * math.cos(angle)
            ty += radius * math.sin(angle)
            if bool(params.get("orbit_face_target", True)):
                # Subtle parallax cue — rotate a touch as we orbit
                rotation += math.degrees(0.0)  # no forced spin; user animates look_yaw_deg if desired
            # Recenter on orbit center (so we actually "look at" it)
            sx = W * 0.5
            sy = H * 0.5
            tx += (sx - ocx * eff_zoom)
            ty += (sy - ocy * eff_zoom)

        # Path mode: follow normalized waypoints; t comes from path_t or __anim.time/duration
        elif cam_mode == "path":
            pts = _parse_points(str(params.get("path_points", "")).strip())
            mode = str(params.get("path_mode", "clamp"))
            # derive t if not provided
            if "path_t" in params:
                t = _norm01(float(params.get("path_t", 0.0)))
            else:
                anim = params.get("__anim", {})
                dur = float(anim.get("duration", 1.0))
                t = 0.0
                if dur > 1e-6:
                    t = (float(anim.get("time", 0.0)) / dur)
            px, py = _polyline_eval(pts, t, mode)
            gx = px * W
            gy = py * H
            sx = W * 0.5
            sy = H * 0.5
            tx += (sx - gx * eff_zoom)
            ty += (sy - gy * eff_zoom)

        # Truck (x/y) and dolly (zoom) post-adjustments
        tx += truck_x
        ty += truck_y
        if abs(dolly) > 1e-6:
            eff_zoom = max(1e-6, eff_zoom * (2.0 ** dolly))

        # Optional camera shake
        shake = float(params.get("shake_strength", 0.0))
        if shake > 1e-6:
            r = _rng(params.get("shake_seed"))
            tx += (r.random() - 0.5) * 2.0 * shake
            ty += (r.random() - 0.5) * 2.0 * shake

        # -------------------- Lens hint + affine --------------------
        cam = design
        cam = _apply_perspective_hint(cam, tilt_x, tilt_y)
        cam = _compose_affine(cam, scale=eff_zoom, rot_deg=rotation, trans=(tx, ty), center=(W / 2, H / 2))

        # -------------------- Lens distortion (radial) --------------------
        k1 = float(params.get("lens_k1", 0.0))
        k2 = float(params.get("lens_k2", 0.0))
        if abs(k1) > 0 or abs(k2) > 0:
            cam = _lens_distort(cam, k1, k2, (W * 0.5, H * 0.5))

        # Composite over background (if any alpha)
        out = canvas.copy()
        out.alpha_composite(cam)

        # -------------------- Post optics --------------------
        out = _vignette(out, float(params.get("vignette", 0.0)))
        out = _chroma_ab(
            out,
            float(params.get("chroma_shift_x_px", params.get("chroma_shift_px", 0.0))),
            float(params.get("chroma_shift_y_px", 0.0)),
        )
        out = _bloom(
            out,
            threshold=_clamp(float(params.get("bloom_threshold", 0.85)), 0.0, 1.0),
            radius=max(0.0, float(params.get("bloom_radius", 8.0))),
            intensity=max(0.0, float(params.get("bloom_intensity", 0.0))),
        )

        # -------------------- Tilt‑shift DOF --------------------
        out = _tilt_shift(
            out,
            band_y=_clamp(float(params.get("tiltshift_center_y", 0.5)), 0, 1),
            band_h=_clamp(float(params.get("tiltshift_band_h", 0.0)), 0, 1),
            blur_radius=max(0.0, float(params.get("tiltshift_blur_radius", 0.0))),
        )

        # -------------------- Tone/grade --------------------
        wb_str = str(params.get("white_balance", "1,1,1")).split(",")
        try:
            wb = (float(wb_str[0]), float(wb_str[1]), float(wb_str[2]))
        except Exception:
            wb = (1.0, 1.0, 1.0)

        out = _tone_map(
            out,
            exposure=float(params.get("exposure", 0.0)),
            contrast=float(params.get("contrast", 0.0)),
            wb=wb,
            lift=float(params.get("lift", 0.0)),
            gamma=max(1e-3, float(params.get("gamma", 1.0))),
            gain=max(1e-3, float(params.get("gain", 1.0))),
        )

        # -------------------- Film grain + rolling shutter --------------------
        out = _grain(out, float(params.get("grain", params.get("film_grain", 0.0))), params.get("grain_seed"))
        out = _rolling_shutter(out, float(params.get("rolling_shutter", 0.0)))

        # -------------------- Anamorphic + Letterbox --------------------
        anam = max(0.01, float(params.get("anamorphic_ratio", 1.0)))
        if abs(anam - 1.0) > 1e-3:
            new_h = int(round(H / anam))
            tmp = out.resize((W, max(1, new_h)), Image.Resampling.LANCZOS)
            out = tmp.resize((W, H), Image.Resampling.LANCZOS)
        aspect = float(params.get("letterbox_aspect", 0.0))
        if aspect > 0:
            lb_col = _parse_color(params.get("letterbox_color", "#000000"), (0, 0, 0, 255)) or (0, 0, 0, 255)
            out = _letterbox(out, aspect, lb_col)

        # -------------------- Optional post pipeline --------------------
        post = str(params.get("post_pipeline", "") or "")
        if post:
            local_params = dict(params)
            img_post: Optional[Image.Image] = out
            for raw in [s.strip().lower() for s in post.split("|") if s.strip()]:
                p = local_params.get(raw, {})
                blk = REGISTRY.create(raw)
                img_post = blk.process(img_post, W, H, params=p)
                if img_post is None:
                    img_post = out
                    break
                img_post = _to_rgba(img_post)
            out = img_post or out

        return out

# Register block name for the engine
REGISTRY.register("camerapipeline", CameraPipeline)
