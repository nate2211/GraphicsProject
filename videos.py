# videos.py
from __future__ import annotations

import os
import sys
import io
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from urllib.parse import urlparse, unquote

import numpy as np
import requests
from PIL import Image, ImageSequence

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
        _norm01,
    )
except Exception as e:
    raise ImportError(
        "videos.py must be able to import from graphics.py "
        "(REGISTRY, BaseBlock, params, help, _ensure_image, _norm01). "
        f"Original error: {e}"
    )

# ------------------------------------------------------------
# Path handling (quotes, file:// URIs, Windows quirks)
# ------------------------------------------------------------
def _sanitize_path(p: Any) -> str:
    """
    Accepts:
      - r'C:\\path\\file.mp4'
      - '"C:\\path\\file.mp4"' or "'C:\\path\\file.mp4'"
      - file:///C:/path/file.mp4
      - file:///C:/path/My%20File.mp4
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
        u = urlparse(str(s))
        return u.scheme.lower() in ("http", "https")
    except Exception:
        return False


# ------------------------------------------------------------
# ffmpeg / ffprobe
# ------------------------------------------------------------
def _which_ffmpeg() -> str:
    # If running as a PyInstaller bundle, use the internal temporary path
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, 'ffmpeg.exe')

    # Fallback for when you are just running it in PyCharm
    return r"C:\Users\natem\PycharmProjects\graphicsProject\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"

def _which_ffprobe() -> str:
    # Prefer sibling ffprobe.exe
    ffmpeg = _which_ffmpeg()
    sib = os.path.join(os.path.dirname(ffmpeg), "ffprobe.exe")
    if os.name == "nt" and os.path.exists(sib):
        return sib
    return "ffprobe"


def _probe_duration_ffprobe(path: str) -> float:
    """
    Best-effort duration probe using ffprobe.
    Returns 0.0 if not available or probe fails.
    """
    ffprobe = _which_ffprobe()
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        path,
    ]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = p.communicate(timeout=10)
        if p.returncode != 0:
            return 0.0
        s = out.decode("utf-8", "ignore").strip()
        return float(s) if s else 0.0
    except Exception:
        return 0.0


def _download_to_temp(url: str, suffix: str = ".mp4") -> str:
    """
    Download a URL to a temp file for ffmpeg input.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GraphicsEngine/1.0)"}
    r = requests.get(url, headers=headers, stream=True, timeout=30)
    r.raise_for_status()
    fd, outp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(outp, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)
    return outp


# ------------------------------------------------------------
# Geometry / placement helpers
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
    Compute resized (w,h) for a source into a destination box.
    mode: contain|cover|stretch|none
    """
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    dst_w = max(1, int(dst_w))
    dst_h = max(1, int(dst_h))
    m = (mode or "contain").strip().lower()

    if m == "none":
        return src_w, src_h
    if m == "stretch":
        return dst_w, dst_h

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    s = max(sx, sy) if m == "cover" else min(sx, sy)
    nw = max(1, int(round(src_w * s)))
    nh = max(1, int(round(src_h * s)))
    return nw, nh


def _time_from_ctx(ctx: Any, fallback: float = 0.0) -> float:
    if ctx is None:
        return fallback
    t = getattr(ctx, "time", None)
    try:
        return float(t) if t is not None else fallback
    except Exception:
        return fallback


def _duration_from_ctx(ctx: Any, fallback: float = 0.0) -> float:
    if ctx is None:
        return fallback
    d = getattr(ctx, "duration", None)
    try:
        return float(d) if d is not None else fallback
    except Exception:
        return fallback


# ------------------------------------------------------------
# Video decode (robust): cache frames as individual PNG bytes
# ------------------------------------------------------------
_VIDEO_CACHE_LOCK = threading.Lock()
_VIDEO_CACHE: Dict[Tuple[str, int, float], Dict[str, Any]] = {}
# key: (src, fps, scale) -> {"frames": [PIL.Image], "dur": float, "fps": int, "w": int, "h": int, "src": str}

def _decode_video_to_frames(path: str, fps: int, scale: float) -> Dict[str, Any]:
    """
    Robust approach: ffmpeg outputs an animated GIF to pipe, Pillow reads frames via ImageSequence.
    This avoids the 'concatenated PNG stream' parsing issues.

    Tradeoffs:
      - GIF conversion may slightly change colors and can be slower than raw frames.
      - For overlays/visualization it’s usually fine.
    """
    ffmpeg = _which_ffmpeg()
    sc = max(0.05, float(scale))
    fps = max(1, min(int(fps), 60))

    # Use palettegen/paletteuse for better GIF quality
    # scale keeps aspect, ensures even dims for some codecs
    vf_scale = f"scale=trunc(iw*{sc}/2)*2:trunc(ih*{sc}/2)*2"
    vf_fps = f"fps={fps}"
    vf = f"{vf_fps},{vf_scale}"

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-i", path,
        "-an",
        "-vf", vf,
        "-f", "gif",
        "pipe:1",
    ]

    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError(
            f"ffmpeg not found ('{ffmpeg}'). Put ffmpeg on PATH or set env FFMPEG_BIN to ffmpeg.exe."
        )

    data, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(
            f"ffmpeg video decode failed ({p.returncode}).\n"
            f"Path: {path}\n"
            f"Stderr: {err.decode('utf-8', 'ignore')[:2000]}"
        )
    if not data:
        raise RuntimeError("ffmpeg produced no video data (file may have no video track).")

    # Read frames
    bio = io.BytesIO(data)
    try:
        gif = Image.open(bio)
    except Exception as e:
        raise RuntimeError(f"Could not parse decoded video stream as GIF: {e}")

    frames = []
    for fr in ImageSequence.Iterator(gif):
        frames.append(fr.convert("RGBA"))

    if not frames:
        raise RuntimeError("Decoded video yielded 0 frames.")

    w, h = frames[0].size
    dur = _probe_duration_ffprobe(path)
    if dur <= 1e-6:
        dur = float(len(frames)) / float(fps)

    return {"frames": frames, "w": w, "h": h, "dur": float(dur), "fps": int(fps)}


def _load_video(src: Any, fps: int = 24, scale: float = 1.0) -> Dict[str, Any]:
    """
    Load + cache decoded frames for local path or http(s) URL.
    Caches frames in memory (fast playback, heavy RAM).
    """
    s = str(src).strip()
    if not s:
        raise ValueError("video source is empty.")

    # Strip wrapping quotes
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()

    fps = max(1, min(int(fps), 60))
    scale = float(scale)

    key = (s, fps, scale)
    with _VIDEO_CACHE_LOCK:
        cached = _VIDEO_CACHE.get(key)
        if cached is not None:
            return cached

    tmpfile = None
    try:
        if _is_url(s):
            tmpfile = _download_to_temp(s, suffix=".mp4")
            path = tmpfile
        else:
            path = _sanitize_path(s)
            ap = os.path.abspath(os.path.expanduser(path))
            if not os.path.exists(ap):
                raise FileNotFoundError(f"Video file not found: {ap}")
            path = ap

        payload = _decode_video_to_frames(path, fps=fps, scale=scale)
        payload.update({"src": s, "scale": scale})
    finally:
        if tmpfile is not None:
            try:
                os.remove(tmpfile)
            except Exception:
                pass

    with _VIDEO_CACHE_LOCK:
        _VIDEO_CACHE[key] = payload
    return payload


# ------------------------------------------------------------
# Block: VideoClip
# ------------------------------------------------------------
@helptext(
    """
    VideoClip: composite a video (path or http(s) URL) onto the current frame.

    Time mapping:
      - If ctx.duration is set, maps ctx.time across the video duration proportionally.
      - Else uses ctx.time directly (seconds) clamped to video duration.

    Performance:
      - Decodes frames into RAM on first use (cached).
      - Use fps + scale to reduce memory usage (e.g. fps=15, scale=0.5).
    """
)
@params({
    "src": {"type": "str", "default": None, "nullable": True, "hint": "file path, file:// URI, or https:// URL"},
    "fps": {"type": "int", "default": 24, "min": 1, "max": 60, "step": 1},
    "scale": {"type": "float", "default": 1.0, "min": 0.1, "max": 2.0, "step": 0.05, "hint": "decode scale (RAM saver)"},

    "fit": {"type": "str", "default": "contain", "hint": "contain|cover|stretch|none"},
    "w": {"type": "float", "default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01},
    "h": {"type": "float", "default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01},

    "x": {"type": "float", "default": 0.5, "min": -2.0, "max": 3.0, "step": 0.01},
    "y": {"type": "float", "default": 0.5, "min": -2.0, "max": 3.0, "step": 0.01},
    "anchor": {"type": "str", "default": "center", "hint": "center|topleft|topright|bottomleft|bottomright"},

    "opacity": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "blend": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},

    "loop": {"type": "bool", "default": True, "hint": "loop video when ctx.time exceeds duration"},
    "time_mode": {
        "type": "str",
        "default": "auto",
        "hint": "auto|realtime|proportional  (auto loops when output > clip)",
    },
})
@dataclass
class VideoClip(BaseBlock):
    def process(self, img, width, height, *, params, ctx=None, **kwargs):
        base = _ensure_image(img, width, height)

        src = params.get("src", None)
        if not src:
            return base

        fps = _clamp_int(params.get("fps", 24), 1, 60, 24)
        scale = max(0.1, min(2.0, _safe_float(params.get("scale", 1.0), 1.0)))
        loop = bool(params.get("loop", True))

        try:
            vid = _load_video(src, fps=fps, scale=scale)
        except Exception as e:
            print(f"[videos] VideoClip load failed: {e}", file=sys.stderr)
            return base

        frames = vid.get("frames") or []
        v_dur = float(vid.get("dur", 0.0))
        v_fps = int(vid.get("fps", fps))
        if not frames:
            return base

        if v_dur <= 1e-6:
            v_dur = float(len(frames)) / float(max(1, v_fps))

        # time mapping
        t = _time_from_ctx(ctx, 0.0)
        ctx_d = _duration_from_ctx(ctx, 0.0)

        mode = str(params.get("time_mode", "auto") or "auto").strip().lower()

        # Decide how to map timeline time -> video time
        # - proportional: fit whole clip into output duration (your old behavior)
        # - realtime: play at natural speed (trim if shorter, loop if longer)
        # - auto: proportional unless output is longer than clip AND loop=True (then realtime+loop)
        t_video = t

        if mode == "realtime":
            t_video = t

        elif mode == "proportional":
            if ctx_d > 1e-6:
                p = max(0.0, min(1.0, t / ctx_d))
                t_video = p * v_dur
            else:
                t_video = t

        else:  # auto
            if ctx_d > 1e-6:
                # If the output is longer than the clip, do NOT stretch the clip—loop it at normal speed.
                # Add a small epsilon (~1 frame) to avoid float edge cases.
                eps = 1.0 / float(max(1, v_fps))
                if loop and v_dur > 1e-6 and ctx_d > (v_dur + eps):
                    t_video = t  # natural speed, will loop below
                else:
                    # otherwise keep your existing "fit clip to output" behavior
                    p = max(0.0, min(1.0, t / ctx_d))
                    t_video = p * v_dur
            else:
                t_video = t

        # Apply looping/clamping
        if loop and v_dur > 1e-6:
            t_video = t_video % v_dur
        else:
            t_video = max(0.0, min(v_dur, t_video))

        # Frame index (floor is usually smoother than round for video sampling)
        idx = int(t_video * v_fps)
        idx = max(0, min(len(frames) - 1, idx))
        im = frames[idx].convert("RGBA")

        # destination box
        dst_w = max(1, int(round(_safe_float(params.get("w", 1.0), 1.0) * width)))
        dst_h = max(1, int(round(_safe_float(params.get("h", 1.0), 1.0) * height)))
        fit = str(params.get("fit", "contain") or "contain").strip().lower()

        nw, nh = _fit_rect(im.size[0], im.size[1], dst_w, dst_h, fit)
        if (nw, nh) != im.size:
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)

        # cover: crop to exact dst box
        if fit == "cover":
            cx = im.size[0] // 2
            cy = im.size[1] // 2
            x0 = max(0, cx - dst_w // 2)
            y0 = max(0, cy - dst_h // 2)
            x1 = min(im.size[0], x0 + dst_w)
            y1 = min(im.size[1], y0 + dst_h)
            im = im.crop((x0, y0, x1, y1))
            if im.size != (dst_w, dst_h):
                im = im.resize((dst_w, dst_h), Image.Resampling.LANCZOS)
        elif fit == "stretch":
            if im.size != (dst_w, dst_h):
                im = im.resize((dst_w, dst_h), Image.Resampling.LANCZOS)
        elif fit == "contain":
            canvas = Image.new("RGBA", (dst_w, dst_h), (0, 0, 0, 0))
            ox = (dst_w - im.size[0]) // 2
            oy = (dst_h - im.size[1]) // 2
            canvas.alpha_composite(im, (ox, oy))
            im = canvas
        else:
            # none: keep im as-is
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
        else:
            px, py = int(round(x - iw / 2)), int(round(y - ih / 2))

        opacity = _norm01(_safe_float(params.get("opacity", 1.0), 1.0))
        blend = _norm01(_safe_float(params.get("blend", 1.0), 1.0))

        if opacity < 0.999:
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
REGISTRY.register("videoclip", VideoClip)
