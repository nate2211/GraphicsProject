# visualizer.py
from __future__ import annotations

import os
import sys
import subprocess
import threading
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from urllib.parse import urlparse, unquote

import numpy as np
from PIL import Image, ImageDraw


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
        "visualizer.py must be able to import from graphics.py "
        "(REGISTRY, BaseBlock, params, help, _ensure_image, _parse_color, _norm01). "
        f"Original error: {e}"
    )


# ------------------------------------------------------------
# Path handling (quotes, file:// URIs, Windows quirks)
# ------------------------------------------------------------
def _sanitize_path(p: Any) -> str:
    """
    Accepts:
      - r'C:\\path\\file.wav'
      - '"C:\\path\\file.wav"' or "'C:\\path\\file.wav'"
      - file:///C:/path/file.wav
      - file:///C:/path/My%20File.wav
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


# ------------------------------------------------------------
# Audio decode + cache
# ------------------------------------------------------------
_AUDIO_CACHE_LOCK = threading.Lock()
_AUDIO_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}
# key: (abspath, sr) -> {"sr": int, "x": np.ndarray float32, "dur": float, "path": str}

def _which_ffmpeg() -> str:
    # If running as a PyInstaller bundle, use the internal temporary path
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, 'ffmpeg.exe')

    # Fallback for when you are just running it in PyCharm
    return r"C:\Users\natem\PycharmProjects\graphicsProject\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"

def _decode_audio_ffmpeg(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Decode any media file to mono float32 samples via ffmpeg.
    Returns (samples, sample_rate).
    """
    ffmpeg = _which_ffmpeg()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-i", path,
        "-vn",
        "-ac", "1",
        "-ar", str(int(target_sr)),
        "-f", "f32le",
        "pipe:1",
    ]

    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        raise RuntimeError(
            f"ffmpeg not found ('{ffmpeg}'). Put ffmpeg on PATH or set env FFMPEG_BIN to ffmpeg.exe."
        )

    out, err = p.communicate()
    if p.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decode failed ({p.returncode}).\n"
            f"Path: {path}\n"
            f"Stderr: {err.decode('utf-8', 'ignore')[:2000]}"
        )

    if not out:
        raise RuntimeError("ffmpeg produced no audio samples (file may have no audio track).")

    x = np.frombuffer(out, dtype=np.float32)
    return x, int(target_sr)

def _load_audio(path: Any, target_sr: int = 48000) -> Dict[str, Any]:
    """
    Load + cache mono float32 audio for the given path.
    Handles quoted paths and file:// URIs.
    """
    p = _sanitize_path(path)
    if not p:
        raise ValueError("audio path is empty.")

    ap = os.path.abspath(os.path.expanduser(p))
    key = (ap, int(target_sr))

    with _AUDIO_CACHE_LOCK:
        cached = _AUDIO_CACHE.get(key)
        if cached is not None:
            return cached

    if not os.path.exists(ap):
        raise FileNotFoundError(f"Audio/media file not found: {ap}")

    x, sr = _decode_audio_ffmpeg(ap, int(target_sr))
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    dur = float(len(x)) / float(sr) if sr > 0 else 0.0
    payload = {"sr": sr, "x": x, "dur": dur, "path": ap}

    with _AUDIO_CACHE_LOCK:
        _AUDIO_CACHE[key] = payload

    return payload


# ------------------------------------------------------------
# Context helpers (safe even if your engine doesn't pass ctx)
# ------------------------------------------------------------
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

def _clamp_int(v: Any, lo: int, hi: int, default: int) -> int:
    try:
        iv = int(v)
    except Exception:
        iv = int(default)
    return max(lo, min(hi, iv))

def _safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)

def _rms(frame: np.ndarray) -> float:
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(frame * frame)))


# ------------------------------------------------------------
# Blocks
# ------------------------------------------------------------
@helptext(
    """
    AudioWaveform: draw a waveform driven by an audio file (wav/mp4/etc).
    Uses ffmpeg to decode to mono float32.

    Notes:
      - Accepts quoted Windows paths: "C:\\Users\\...\\file.wav"
      - Accepts file:// URIs from drag/drop.
      - If ctx.duration is set, maps ctx.time proportionally across audio duration.

    Playhead:
      - Set show_playhead=false to hide it.
    """
)
@params({
    "path": {"type": "path", "default": None, "nullable": True, "hint": "wav/mp4/etc (ffmpeg-decodable)"},
    "sr": {"type": "int", "default": 48000, "min": 8000, "max": 192000, "step": 1000},

    "window_s": {"type": "float", "default": 1.5, "min": 0.05, "max": 30.0, "step": 0.05},
    "gain": {"type": "float", "default": 1.0, "min": 0.01, "max": 20.0, "step": 0.01},

    "thickness": {"type": "int", "default": 2, "min": 1, "max": 20, "step": 1},
    "color": {"type": "color", "default": "#FFFFFF"},
    "bg": {"type": "color", "default": "none"},

    # playhead options
    "show_playhead": {"type": "bool", "default": True},
    "playhead_color": {"type": "color", "default": "#FF4D4D"},
    "playhead_width": {"type": "int", "default": 2, "min": 1, "max": 20, "step": 1},

    "blend": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
})
@dataclass
class AudioWaveform(BaseBlock):
    def process(self, img, width, height, *, params, ctx=None, **kwargs):
        img = _ensure_image(img, width, height)

        path = _sanitize_path(params.get("path", None))
        if not path:
            return img

        sr = _clamp_int(params.get("sr", 48000), 8000, 192000, 48000)
        window_s = max(0.01, _safe_float(params.get("window_s", 1.5), 1.5))
        gain = max(0.001, _safe_float(params.get("gain", 1.0), 1.0))
        thickness = _clamp_int(params.get("thickness", 2), 1, 20, 2)
        blend = _norm01(_safe_float(params.get("blend", 1.0), 1.0))

        wave_color = _parse_color(params.get("color", "#FFFFFF"), (255, 255, 255, 255))
        bg_color = _parse_color(params.get("bg", "none"), default=None)

        show_playhead = bool(params.get("show_playhead", True))
        ph_color = _parse_color(params.get("playhead_color", "#FF4D4D"), (255, 77, 77, 255)) if show_playhead else None
        ph_w = _clamp_int(params.get("playhead_width", 2), 1, 20, 2)

        try:
            aud = _load_audio(path, target_sr=sr)
        except Exception as e:
            print(f"[visualizer] AudioWaveform load failed: {e}", file=sys.stderr)
            return img

        x = aud["x"]
        a_sr = aud["sr"]
        a_dur = aud["dur"]

        t = _time_from_ctx(ctx, 0.0)
        ctx_d = _duration_from_ctx(ctx, 0.0)
        if ctx_d > 1e-6:
            p = max(0.0, min(1.0, t / ctx_d))
            t_audio = p * a_dur
        else:
            t_audio = max(0.0, min(a_dur, t))

        half = window_s * 0.5
        t0 = max(0.0, t_audio - half)
        t1 = min(a_dur, t_audio + half)
        i0 = int(t0 * a_sr)
        i1 = int(t1 * a_sr)
        if i1 <= i0 + 8:
            return img

        seg = np.clip(x[i0:i1] * gain, -1.0, 1.0)

        # peak-per-column
        n = seg.size
        cols = max(1, width)
        step = max(1, n // cols)

        mins = np.zeros(cols, dtype=np.float32)
        maxs = np.zeros(cols, dtype=np.float32)
        for c in range(cols):
            a = c * step
            b = min(n, (c + 1) * step)
            chunk = seg[a:b]
            if chunk.size:
                mins[c] = float(chunk.min())
                maxs[c] = float(chunk.max())
            else:
                mins[c] = 0.0
                maxs[c] = 0.0

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)

        if bg_color is not None:
            d.rectangle([0, 0, width, height], fill=bg_color)

        midy = height * 0.5
        amp = height * 0.45

        if wave_color is not None:
            for xpix in range(cols):
                y_min = midy - (maxs[xpix] * amp)
                y_max = midy - (mins[xpix] * amp)
                y0 = float(min(y_min, y_max))
                y1 = float(max(y_min, y_max))
                d.line([(xpix, y0), (xpix, y1)], fill=wave_color, width=thickness)

        # playhead (optional)
        if show_playhead and ph_color is not None and ph_w > 0:
            cx = width // 2
            d.line([(cx, 0), (cx, height)], fill=ph_color, width=ph_w)

        if blend >= 0.999:
            return Image.alpha_composite(img, overlay)
        if blend <= 1e-6:
            return img

        # alpha-scale overlay
        ov = np.asarray(overlay).astype(np.float32)
        base = np.asarray(img).astype(np.float32)
        a = (ov[..., 3:4] / 255.0) * blend
        out = base.copy()
        out[..., :3] = base[..., :3] * (1.0 - a) + ov[..., :3] * a
        out[..., 3:4] = np.clip(base[..., 3:4] + ov[..., 3:4] * blend, 0, 255)
        return Image.fromarray(out.astype(np.uint8), "RGBA")


@helptext(
    """
    AudioBars: FFT spectrum bars driven by an audio file.

    - Accepts quoted Windows paths and file:// URIs.
    - Uses log-spaced frequency bins for nicer music visuals.
    """
)
@params({
    "path": {"type": "path", "default": None, "nullable": True},
    "sr": {"type": "int", "default": 48000, "min": 8000, "max": 192000, "step": 1000},

    "fft_size": {"type": "int", "default": 2048, "min": 256, "max": 65536, "step": 256},
    "gain": {"type": "float", "default": 2.0, "min": 0.01, "max": 50.0, "step": 0.01},

    "bars": {"type": "int", "default": 64, "min": 4, "max": 512, "step": 1},
    "min_hz": {"type": "float", "default": 30.0, "min": 0.0, "max": 20000.0, "step": 1.0},
    "max_hz": {"type": "float", "default": 12000.0, "min": 10.0, "max": 24000.0, "step": 10.0},

    "color": {"type": "color", "default": "#FFFFFF"},
    "bg": {"type": "color", "default": "none"},
    "fill_alpha": {"type": "float", "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
    "baseline": {"type": "float", "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
})
@dataclass
class AudioBars(BaseBlock):
    def process(self, img, width, height, *, params, ctx=None, **kwargs):
        img = _ensure_image(img, width, height)

        path = _sanitize_path(params.get("path", None))
        if not path:
            return img

        sr = _clamp_int(params.get("sr", 48000), 8000, 192000, 48000)
        fft_size = _clamp_int(params.get("fft_size", 2048), 256, 65536, 2048)
        bars = _clamp_int(params.get("bars", 64), 4, 512, 64)
        gain = max(0.001, _safe_float(params.get("gain", 2.0), 2.0))

        min_hz = max(0.0, _safe_float(params.get("min_hz", 30.0), 30.0))
        max_hz = max(min_hz + 1.0, _safe_float(params.get("max_hz", 12000.0), 12000.0))

        color = _parse_color(params.get("color", "#FFFFFF"), (255, 255, 255, 255))
        bg = _parse_color(params.get("bg", "none"), default=None)
        fill_alpha = _norm01(_safe_float(params.get("fill_alpha", 1.0), 1.0))
        baseline = _norm01(_safe_float(params.get("baseline", 0.95), 0.95))

        try:
            aud = _load_audio(path, target_sr=sr)
        except Exception as e:
            print(f"[visualizer] AudioBars load failed: {e}", file=sys.stderr)
            return img

        x = aud["x"]
        a_sr = aud["sr"]
        a_dur = aud["dur"]

        t = _time_from_ctx(ctx, 0.0)
        ctx_d = _duration_from_ctx(ctx, 0.0)
        if ctx_d > 1e-6:
            p = max(0.0, min(1.0, t / ctx_d))
            t_audio = p * a_dur
        else:
            t_audio = max(0.0, min(a_dur, t))

        center = int(t_audio * a_sr)
        half = int(fft_size // 2)
        i0 = max(0, center - half)
        i1 = min(x.size, center + half)
        frame = x[i0:i1]
        if frame.size < 32:
            return img

        if frame.size < fft_size:
            frame = np.pad(frame, (0, fft_size - frame.size), mode="constant")

        w = np.hanning(fft_size).astype(np.float32)
        mag = np.abs(np.fft.rfft(frame * w)).astype(np.float32)
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / a_sr)

        lo_i = int(np.searchsorted(freqs, min_hz, side="left"))
        hi_i = int(np.searchsorted(freqs, max_hz, side="right"))
        lo_i = max(0, min(lo_i, mag.size - 1))
        hi_i = max(lo_i + 1, min(hi_i, mag.size))

        mag = mag[lo_i:hi_i]
        freqs = freqs[lo_i:hi_i]

        f0 = max(min_hz, 1.0)
        f1 = max(max_hz, f0 + 1.0)
        edges = np.logspace(np.log10(f0), np.log10(f1), num=bars + 1)

        vals = np.zeros(bars, dtype=np.float32)
        for b in range(bars):
            e0, e1 = edges[b], edges[b + 1]
            a0 = int(np.searchsorted(freqs, e0, side="left"))
            a1 = int(np.searchsorted(freqs, e1, side="right"))
            a0 = max(0, min(a0, mag.size - 1))
            a1 = max(a0 + 1, min(a1, mag.size))
            chunk = mag[a0:a1]
            vals[b] = float(np.mean(chunk)) if chunk.size else 0.0

        ref = float(np.percentile(vals, 95)) + 1e-9
        vals = np.clip((vals / ref) * gain, 0.0, 1.0)

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)

        if bg is not None:
            d.rectangle([0, 0, width, height], fill=bg)

        if color is None:
            return Image.alpha_composite(img, overlay)

        r, g, b, a = color
        a = int(round(a * fill_alpha))
        bar_color = (r, g, b, a)

        base_y = int(round(height * baseline))
        max_h = int(round(height * 0.85))
        pad_px = 2
        total_pad = pad_px * (bars + 1)
        bar_w = max(1, (width - total_pad) // bars)

        xcur = pad_px
        for v in vals:
            h = int(round(v * max_h))
            d.rectangle([xcur, base_y - h, xcur + bar_w, base_y], fill=bar_color)
            xcur += bar_w + pad_px

        return Image.alpha_composite(img, overlay)


@helptext(
    """
    AudioRMS: simple RMS/energy meter driven by audio.

    - Accepts quoted Windows paths and file:// URIs.
    - Draws a horizontal loudness bar.
    """
)
@params({
    "path": {"type": "path", "default": None, "nullable": True},
    "sr": {"type": "int", "default": 48000, "min": 8000, "max": 192000, "step": 1000},
    "window_ms": {"type": "int", "default": 50, "min": 5, "max": 500, "step": 5},
    "gain": {"type": "float", "default": 4.0, "min": 0.01, "max": 50.0, "step": 0.01},

    "x": {"type": "float", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
    "y": {"type": "float", "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
    "w": {"type": "float", "default": 0.8, "min": 0.01, "max": 2.0, "step": 0.01},
    "h": {"type": "float", "default": 0.05, "min": 0.005, "max": 0.5, "step": 0.005},

    "fill": {"type": "color", "default": "#FFFFFF"},
    "bg": {"type": "color", "default": "#000000"},
    "bg_alpha": {"type": "float", "default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01},
    "outline": {"type": "color", "default": "none"},
    "outline_width": {"type": "int", "default": 2, "min": 0, "max": 20, "step": 1},
})
@dataclass
class AudioRMS(BaseBlock):
    def process(self, img, width, height, *, params, ctx=None, **kwargs):
        img = _ensure_image(img, width, height)

        path = _sanitize_path(params.get("path", None))
        if not path:
            return img

        sr = _clamp_int(params.get("sr", 48000), 8000, 192000, 48000)
        window_ms = _clamp_int(params.get("window_ms", 50), 5, 500, 50)
        gain = max(0.001, _safe_float(params.get("gain", 4.0), 4.0))

        x0 = float(params.get("x", 0.1)) * width
        y0 = float(params.get("y", 0.9)) * height
        ww = float(params.get("w", 0.8)) * width
        hh = float(params.get("h", 0.05)) * height

        fill = _parse_color(params.get("fill", "#FFFFFF"), (255, 255, 255, 255))
        bg = _parse_color(params.get("bg", "#000000"), (0, 0, 0, 255))
        bg_alpha = _norm01(_safe_float(params.get("bg_alpha", 0.35), 0.35))
        outline = _parse_color(params.get("outline", "none"), default=None)
        ow = _clamp_int(params.get("outline_width", 2), 0, 20, 2)

        try:
            aud = _load_audio(path, target_sr=sr)
        except Exception as e:
            print(f"[visualizer] AudioRMS load failed: {e}", file=sys.stderr)
            return img

        x = aud["x"]
        a_sr = aud["sr"]
        a_dur = aud["dur"]

        t = _time_from_ctx(ctx, 0.0)
        ctx_d = _duration_from_ctx(ctx, 0.0)
        if ctx_d > 1e-6:
            p = max(0.0, min(1.0, t / ctx_d))
            t_audio = p * a_dur
        else:
            t_audio = max(0.0, min(a_dur, t))

        center = int(t_audio * a_sr)
        half = int((window_ms / 1000.0) * a_sr / 2.0)
        i0 = max(0, center - half)
        i1 = min(x.size, center + half)
        frame = x[i0:i1]
        val = max(0.0, min(1.0, _rms(frame) * gain))

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)

        if bg is not None and bg_alpha > 1e-6:
            br, bgc, bb, ba = bg
            ba = int(round(ba * bg_alpha))
            d.rectangle([x0, y0, x0 + ww, y0 + hh], fill=(br, bgc, bb, ba))

        if fill is not None:
            d.rectangle([x0, y0, x0 + ww * val, y0 + hh], fill=fill)

        if outline is not None and ow > 0:
            d.rectangle([x0, y0, x0 + ww, y0 + hh], outline=outline, width=ow)

        return Image.alpha_composite(img, overlay)


# ------------------------------------------------------------
# Registration
# ------------------------------------------------------------
REGISTRY.register("audiowaveform", AudioWaveform)
REGISTRY.register("audiobars", AudioBars)
REGISTRY.register("audiorms", AudioRMS)
