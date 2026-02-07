# animations.py — Animation Blocks (help/params annotated)
from __future__ import annotations

import math
import sys
import os  # kept (font path checks, future use)
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Callable

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Graphics deps (import from graphics.py, with safe fallbacks for dev/testing)
# ---------------------------------------------------------------------------
try:
    from graphics import (
        BaseBlock as GfxBaseBlock,
        BlockRegistry,
        REGISTRY as GFX_REGISTRY,
        GraphicsEngine,
        _auto,
        _parse_color,
        help, params
    )
except ImportError as e:
    print(f"[animations] Warning: graphics import failed: {e}", file=sys.stderr)
    print("         Using dummy graphics components. Functionality will be limited.", file=sys.stderr)

    class GfxBaseBlock:  # type: ignore
        pass

    class BlockRegistry:  # type: ignore
        def __init__(self) -> None:
            self._by_name: Dict[str, type] = {}

        def register(self, name: str, cls: type) -> None:
            key = str(name).strip().lower()
            self._by_name[key] = cls

        def create(self, name: str, **kwargs: Any) -> Any:
            key = str(name).strip().lower()
            if key not in self._by_name:
                raise KeyError(f"[animations dummy] Block '{name}' not found.")
            return self._by_name[key](**kwargs)  # type: ignore

        def names(self) -> List[str]:
            return sorted(self._by_name.keys())

    GFX_REGISTRY = BlockRegistry()  # type: ignore
    GraphicsEngine = object         # type: ignore

    def _auto(v: Any) -> Any:  # type: ignore
        if isinstance(v, (int, float, bool)):
            return v
        return str(v)

    def _parse_color(v: Any, default: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:  # type: ignore
        return default if default is not None else (255, 255, 255, 255)

    # If graphics isn't available, help/params decorators might not exist.
    # Provide no-op fallbacks so import still works.
    def help(_s: str):  # type: ignore
        def deco(cls): return cls
        return deco

    def params(_d: Dict[str, Any]):  # type: ignore
        def deco(cls): return cls
        return deco


# ---------------------------------------------------------------------------
# Easing Functions (0..1 -> 0..1)
# ---------------------------------------------------------------------------
def ease_linear(t: float) -> float:
    return t

def ease_in_quad(t: float) -> float:
    return t * t

def ease_out_quad(t: float) -> float:
    return t * (2 - t)

def ease_in_out_quad(t: float) -> float:
    return 2 * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 2) / 2

def ease_in_sine(t: float) -> float:
    return 1 - math.cos((t * math.pi) / 2)

def ease_out_sine(t: float) -> float:
    return math.sin((t * math.pi) / 2)

def ease_in_out_sine(t: float) -> float:
    return -(math.cos(math.pi * t) - 1) / 2

def ease_in_out_cubic(t: float) -> float:
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

EASING_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    "linear": ease_linear,
    "in_quad": ease_in_quad, "quadin": ease_in_quad,
    "out_quad": ease_out_quad, "quadout": ease_out_quad,
    "in_out_quad": ease_in_out_quad, "quadinout": ease_in_out_quad,
    "in_sine": ease_in_sine, "sinein": ease_in_sine,
    "out_sine": ease_out_sine, "sineout": ease_out_sine,
    "in_out_sine": ease_in_out_sine, "sineinout": ease_in_out_sine,
    "in_out_cubic": ease_in_out_cubic, "cubicinout": ease_in_out_cubic,
}

EASING_CHOICES = sorted(set(EASING_FUNCTIONS.keys()))


# ---------------------------------------------------------------------------
# Small math/parsing helpers
# ---------------------------------------------------------------------------
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * float(t)

def _lerp_color(a_rgba: Tuple[int, int, int, int],
                b_rgba: Tuple[int, int, int, int],
                t: float) -> Tuple[int, int, int, int]:
    aa = np.array(a_rgba, dtype=np.float32)
    bb = np.array(b_rgba, dtype=np.float32)
    cc = aa + (bb - aa) * float(t)
    c = np.clip(cc, 0, 255).round().astype(np.int32)
    return (int(c[0]), int(c[1]), int(c[2]), int(c[3]))

def _as_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(_auto(str(v)))
    except (ValueError, TypeError):
        return default

def _as_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(round(v))
    try:
        return int(round(float(_auto(str(v)))))
    except (ValueError, TypeError):
        return default

def _get_easing(name: Any, default: Callable[[float], float] = ease_linear) -> Callable[[float], float]:
    try:
        s = str(name).strip().lower()
    except Exception:
        s = ""
    return EASING_FUNCTIONS.get(s, default)

def _parse_key_pairs(spec: Any) -> List[Tuple[float, Any]]:
    """
    Parses keyframe specifier:
      - string: "t:v, t:v, ..."
      - list: [(t, v), ...] or ["t:v", ...]
    Returns sorted list of (time_float, raw_value).
    """
    pairs: List[Tuple[float, Any]] = []
    if spec is None:
        return pairs

    items_to_parse: List[Any] = []
    if isinstance(spec, str):
        items_to_parse = [s.strip() for s in spec.split(",") if s.strip()]
    elif isinstance(spec, (list, tuple)):
        items_to_parse = list(spec)

    for item in items_to_parse:
        t_float: Optional[float] = None
        v_raw: Any = None

        if isinstance(item, str):
            if ":" in item:
                t_str, v_str = item.split(":", 1)
                try:
                    t_float = float(t_str.strip())
                except ValueError:
                    continue
                v_raw = v_str.strip()
            else:
                # string without ':', treat as value at t=1.0
                t_float = 1.0
                v_raw = item.strip()

        elif isinstance(item, (list, tuple)) and len(item) == 2:
            t_raw, v_raw = item
            try:
                t_float = float(_auto(t_raw))
            except (ValueError, TypeError):
                continue
        else:
            continue

        if t_float is not None:
            pairs.append((t_float, v_raw))

    pairs.sort(key=lambda kv: kv[0])
    return pairs

def _sample_keyframes_float(pairs: List[Tuple[float, Any]], tnorm: float, default: float = 0.0) -> float:
    if not pairs:
        return default
    t = _clamp01(tnorm)

    prev_t, prev_v_raw = pairs[0]
    next_t, next_v_raw = pairs[-1]
    if t <= prev_t:
        return _as_float(prev_v_raw, default)
    if t >= next_t:
        return _as_float(next_v_raw, default)

    for i in range(len(pairs) - 1):
        t0, v0_raw = pairs[i]
        t1, v1_raw = pairs[i + 1]
        if t0 <= t <= t1:
            v0 = _as_float(v0_raw, default)
            v1 = _as_float(v1_raw, default)
            interval = t1 - t0
            u = (t - t0) / max(1e-9, interval) if interval > 1e-9 else 0.0
            return _lerp(v0, v1, u)

    return _as_float(next_v_raw, default)

def _sample_keyframes_color(
    pairs: List[Tuple[float, Any]],
    tnorm: float,
    default: Tuple[int, int, int, int] = (0, 0, 0, 255),
) -> Tuple[int, int, int, int]:
    if not pairs:
        return default
    t = _clamp01(tnorm)

    prev_t, prev_v_raw = pairs[0]
    next_t, next_v_raw = pairs[-1]
    if t <= prev_t:
        return _parse_color(prev_v_raw, default) or default
    if t >= next_t:
        return _parse_color(next_v_raw, default) or default

    for i in range(len(pairs) - 1):
        t0, v0_raw = pairs[i]
        t1, v1_raw = pairs[i + 1]
        if t0 <= t <= t1:
            c0 = _parse_color(v0_raw, default) or default
            c1 = _parse_color(v1_raw, default) or default
            interval = t1 - t0
            u = (t - t0) / max(1e-9, interval) if interval > 1e-9 else 0.0
            return _lerp_color(c0, c1, u)

    return _parse_color(next_v_raw, default) or default


# ---------------------------------------------------------------------------
# Animation Context and Base Block
# ---------------------------------------------------------------------------
@dataclass
class AnimationContext:
    frame: int = 0
    total_frames: int = 1
    time: float = 0.0
    duration: float = 1.0
    fps: float = 30.0
    width: int = 640
    height: int = 480

@dataclass
class BaseAnimationBlock:
    """Base class for animation controllers."""
    def process_frame(
        self,
        img: Optional[Image.Image],
        ctx: AnimationContext,
        params: Dict[str, Any],
        engine: "GraphicsEngine",
    ) -> Image.Image:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Animation Registry Instance
# ---------------------------------------------------------------------------
ANIMATION_REGISTRY = BlockRegistry()


# ---------------------------------------------------------------------------
# AnimateParameter
# ---------------------------------------------------------------------------
@help(
    "Animate a single parameter on a target graphics block by lerping start->end over time.\n"
    "You typically place this inside a sub-pipeline (animation step) and set:\n"
    "  target_block, param_name, start_value, end_value, start_time, end_time, param_type, easing."
)
@params({
    "target_block": {"type": "str", "default": "", "hint": "Name of graphics block to run (e.g. 'camerapipeline')"},
    "param_name":   {"type": "str", "default": "", "hint": "Parameter name on target block to animate"},
    "start_value":  {"type": "any", "default": 0.0},
    "end_value":    {"type": "any", "default": 1.0},
    "start_time":   {"type": "float", "default": 0.0, "min": 0.0, "max": 1e9, "step": 0.01, "unit": "s"},
    "end_time":     {"type": "float", "default": 1.0, "min": 0.0, "max": 1e9, "step": 0.01, "unit": "s"},
    "param_type":   {"type": "enum", "default": "float", "choices": ["float", "int", "color"]},
    "easing":       {"type": "enum", "default": "linear", "choices": EASING_CHOICES},
})
@dataclass
class AnimateParameter(BaseAnimationBlock):
    """
    Interpolates a single parameter of a target graphics block.
    """
    def process_frame(self, img, ctx, params, engine):
        tb_name = params.get("target_block")
        param_name = params.get("param_name")
        start_val = params.get("start_value")
        end_val = params.get("end_value")

        if not tb_name or not param_name or start_val is None or end_val is None:
            print(
                f"[AnimateParameter] Warning: missing required params "
                f"(target_block='{tb_name}', param_name='{param_name}').",
                file=sys.stderr,
            )
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))

        start_time = _as_float(params.get("start_time", 0.0))
        end_time = _as_float(params.get("end_time", ctx.duration))
        param_type = str(params.get("param_type", "float")).lower()
        easing_func = _get_easing(params.get("easing", "linear"))

        seg_dur = max(1e-9, end_time - start_time)
        rel = ctx.time - start_time
        t_raw = _clamp01(rel / seg_dur)
        t_eased = _clamp01(easing_func(t_raw))

        try:
            if param_type == "color":
                c0 = _parse_color(start_val, (0, 0, 0, 255)) or (0, 0, 0, 255)
                c1 = _parse_color(end_val, (255, 255, 255, 255)) or (255, 255, 255, 255)
                current_value = _lerp_color(c0, c1, t_eased)
            elif param_type == "int":
                v0, v1 = _as_int(start_val), _as_int(end_val)
                current_value = int(round(_lerp(v0, v1, t_eased)))
            else:
                v0, v1 = _as_float(start_val), _as_float(end_val)
                current_value = _lerp(v0, v1, t_eased)
        except Exception as e_interp:
            print(f"[AnimateParameter] Error interpolating '{param_name}' ({param_type}): {e_interp}", file=sys.stderr)
            current_value = start_val

        try:
            target_block = GFX_REGISTRY.create(tb_name)  # type: ignore
            target_params = {str(param_name): current_value}
            if GfxBaseBlock and isinstance(target_block, GfxBaseBlock):
                return target_block.process(img, ctx.width, ctx.height, params=target_params)
            raise TypeError(f"Target '{tb_name}' is not a valid graphics block.")
        except Exception as e_draw:
            print(f"[AnimateParameter] Error running target block '{tb_name}': {e_draw}", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))

    def get_overrides(self, ctx: AnimationContext, params: Dict[str, Any], engine) -> tuple[str, Dict[str, Any]]:
        tb_name = params.get("target_block")
        param_name = params.get("param_name")
        start_val = params.get("start_value")
        end_val = params.get("end_value")

        if not tb_name or not param_name or start_val is None or end_val is None:
            return "", {}

        start_time = _as_float(params.get("start_time", 0.0))
        end_time = _as_float(params.get("end_time", ctx.duration))
        param_type = str(params.get("param_type", "float")).lower()
        easing_func = _get_easing(params.get("easing", "linear"))

        seg_dur = max(1e-9, end_time - start_time)
        rel = ctx.time - start_time
        t_raw = _clamp01(rel / seg_dur)
        t_eased = _clamp01(easing_func(t_raw))

        if param_type == "color":
            c0 = _parse_color(start_val, (0, 0, 0, 255)) or (0, 0, 0, 255)
            c1 = _parse_color(end_val, (255, 255, 255, 255)) or (255, 255, 255, 255)
            current_value = _lerp_color(c0, c1, t_eased)
        elif param_type == "int":
            v0, v1 = _as_int(start_val), _as_int(end_val)
            current_value = int(round(_lerp(v0, v1, t_eased)))
        else:
            v0, v1 = _as_float(start_val), _as_float(end_val)
            current_value = _lerp(v0, v1, t_eased)

        return str(tb_name).strip().lower(), {str(param_name): current_value}
ANIMATION_REGISTRY.register("animateparam", AnimateParameter)


# ---------------------------------------------------------------------------
# PingPongParameter
# ---------------------------------------------------------------------------
@help(
    "Oscillate a single parameter between start_value and end_value with a triangle wave.\n"
    "Use 'period' for cycle length, 'phase' for offset, and optional easing."
)
@params({
    "target_block": {"type": "str", "default": ""},
    "param_name":   {"type": "str", "default": ""},
    "start_value":  {"type": "any", "default": 0.0},
    "end_value":    {"type": "any", "default": 1.0},
    "period":       {"type": "float", "default": 2.0, "min": 1e-6, "max": 1e9, "step": 0.01, "unit": "s"},
    "phase":        {"type": "float", "default": 0.0, "min": -1e9, "max": 1e9, "step": 0.01, "unit": "s"},
    "param_type":   {"type": "enum", "default": "float", "choices": ["float", "int", "color"]},
    "easing":       {"type": "enum", "default": "linear", "choices": EASING_CHOICES},
})
@dataclass
class PingPongParameter(BaseAnimationBlock):
    """
    Oscillates a parameter between start and end values using a triangle wave.
    """
    def process_frame(self, img, ctx, params, engine):
        tb_name = params.get("target_block")
        param_name = params.get("param_name")
        start_val = params.get("start_value")
        end_val = params.get("end_value")

        if not tb_name or not param_name or start_val is None or end_val is None:
            print("[PingPongParameter] Warning: Missing required params. Skipping.", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))

        period = max(1e-9, _as_float(params.get("period", 2.0)))
        phase = _as_float(params.get("phase", 0.0))
        param_type = str(params.get("param_type", "float")).lower()
        easing_func = _get_easing(params.get("easing", "linear"))

        time_shifted = max(0.0, ctx.time + phase)
        cycle_pos = (time_shifted % period) / period
        tri = 1.0 - abs(2.0 * cycle_pos - 1.0)
        t_eased = _clamp01(easing_func(tri))

        try:
            if param_type == "color":
                c0 = _parse_color(start_val, (0, 0, 0, 255)) or (0, 0, 0, 255)
                c1 = _parse_color(end_val, (255, 255, 255, 255)) or (255, 255, 255, 255)
                current_value = _lerp_color(c0, c1, t_eased)
            elif param_type == "int":
                v0, v1 = _as_int(start_val), _as_int(end_val)
                current_value = int(round(_lerp(v0, v1, t_eased)))
            else:
                v0, v1 = _as_float(start_val), _as_float(end_val)
                current_value = _lerp(v0, v1, t_eased)
        except Exception as e_interp:
            print(f"[PingPongParameter] Error interpolating '{param_name}': {e_interp}", file=sys.stderr)
            current_value = start_val

        try:
            target_block = GFX_REGISTRY.create(tb_name)  # type: ignore
            target_params = {str(param_name): current_value}
            if GfxBaseBlock and isinstance(target_block, GfxBaseBlock):
                return target_block.process(img, ctx.width, ctx.height, params=target_params)
            raise TypeError(f"Target '{tb_name}' is not a valid graphics block.")
        except Exception as e_draw:
            print(f"[PingPongParameter] Error running target block '{tb_name}': {e_draw}", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))

ANIMATION_REGISTRY.register("pingpongparam", PingPongParameter)


# ---------------------------------------------------------------------------
# KeyframeParameter
# ---------------------------------------------------------------------------
@help(
    "Animate a single parameter through multiple keyframes.\n"
    "keys may be 't:v, t:v, ...' or a list of pairs.\n"
    "If normalized=true: key times are 0..1. Else: key times are absolute seconds."
)
@params({
    "target_block": {"type": "str", "default": ""},
    "param_name":   {"type": "str", "default": ""},
    "keys":         {"type": "any", "default": "", "hint": "e.g. '0:0, 0.5:1, 1:0'"},
    "normalized":   {"type": "bool", "default": False},
    "param_type":   {"type": "enum", "default": "float", "choices": ["float", "int", "color"]},
})
@dataclass
class KeyframeParameter(BaseAnimationBlock):
    """
    Interpolates a parameter based on keyframes.
    """
    _parsed_keys: List[Tuple[float, Any]] = field(init=False, default_factory=list)
    _last_keys_spec: Any = field(init=False, default=None)

    def _get_parsed_keys(self, keys_spec: Any) -> List[Tuple[float, Any]]:
        if keys_spec != self._last_keys_spec:
            self._parsed_keys = _parse_key_pairs(keys_spec)
            self._last_keys_spec = keys_spec
            if not self._parsed_keys:
                print(f"[KeyframeParameter] Warning: No valid keyframes parsed from spec: {keys_spec}", file=sys.stderr)
        return self._parsed_keys

    def process_frame(self, img, ctx, params, engine):
        tb_name = params.get("target_block")
        param_name = params.get("param_name")
        keys_spec = params.get("keys")

        if not tb_name or not param_name or keys_spec is None:
            print("[KeyframeParameter] Warning: Missing required params. Skipping.", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))

        param_type = str(params.get("param_type", "float")).lower()
        use_normalized_time = bool(params.get("normalized", False))

        pairs = self._get_parsed_keys(keys_spec)
        if not pairs:
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))

        # Build sample_pairs with times normalized to 0..1, and sample_time in 0..1
        if use_normalized_time:
            sample_pairs = pairs
            sample_time = _clamp01(ctx.time / max(1e-9, ctx.duration))
        else:
            min_t = pairs[0][0]
            max_t = pairs[-1][0]
            span = max(1e-9, max_t - min_t)
            sample_time = _clamp01((ctx.time - min_t) / span)
            sample_pairs = [((t - min_t) / span, v) for t, v in pairs]
            if sample_pairs:
                sample_pairs[0] = (0.0, sample_pairs[0][1])
                sample_pairs[-1] = (1.0, sample_pairs[-1][1])

        try:
            if param_type == "color":
                current_value = _sample_keyframes_color(sample_pairs, sample_time)
            elif param_type == "int":
                current_value = int(round(_sample_keyframes_float(sample_pairs, sample_time)))
            else:
                current_value = _sample_keyframes_float(sample_pairs, sample_time)
        except Exception as e_sample:
            print(f"[KeyframeParameter] Error sampling '{param_name}': {e_sample}", file=sys.stderr)
            current_value = pairs[0][1]

        try:
            target_block = GFX_REGISTRY.create(tb_name)  # type: ignore
            target_params = {str(param_name): current_value}
            if GfxBaseBlock and isinstance(target_block, GfxBaseBlock):
                return target_block.process(img, ctx.width, ctx.height, params=target_params)
            raise TypeError(f"Target '{tb_name}' is not a valid graphics block.")
        except Exception as e_draw:
            print(f"[KeyframeParameter] Error running target block '{tb_name}': {e_draw}", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))

ANIMATION_REGISTRY.register("keyframeparam", KeyframeParameter)


# ---------------------------------------------------------------------------
# WiggleParameter
# ---------------------------------------------------------------------------
@help(
    "Add a pseudo-random wiggle around a base value.\n"
    "- float/int: adds +/- amount\n"
    "- color: mixes base toward white/black with strength=amount\n"
    "freq_hz controls speed; seed makes it deterministic per-frame."
)
@params({
    "target_block": {"type": "str", "default": ""},
    "param_name":   {"type": "str", "default": ""},
    "base_value":   {"type": "any", "default": 0.0},
    "base":         {"type": "any", "default": None, "nullable": True, "hint": "alias for base_value"},
    "amount":       {"type": "float", "default": 0.1, "min": 0.0, "max": 1e9, "step": 0.001},
    "freq_hz":      {"type": "float", "default": 2.0, "min": 0.0, "max": 1e6, "step": 0.01},
    "param_type":   {"type": "enum", "default": "float", "choices": ["float", "int", "color"]},
    "seed":         {"type": "int", "default": None, "nullable": True},
})
@dataclass
class WiggleParameter(BaseAnimationBlock):
    """
    Adds time-varying noise (wiggle) to a base parameter value.
    """
    def process_frame(self, img, ctx, params, engine):
        tb_name = params.get("target_block")
        param_name = params.get("param_name")

        base_val = params.get("base_value")
        if base_val is None:
            base_val = params.get("base")

        if not tb_name or not param_name or base_val is None:
            print("[WiggleParameter] Warning: Missing required params. Skipping.", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))

        param_type = str(params.get("param_type", "float")).lower()
        freq = _as_float(params.get("freq_hz", 2.0))
        amount = _as_float(params.get("amount", 0.1))
        seed = params.get("seed")

        combined_seed = int(ctx.frame)
        if seed is not None:
            try:
                combined_seed += int(seed)
            except Exception:
                pass
        np.random.seed(combined_seed & 0xFFFFFFFF)

        time_phase = 2 * math.pi * freq * ctx.time
        sine_comp = 0.6 * math.sin(time_phase) + 0.4 * math.sin(time_phase * 1.37 + 1.23)
        noise_comp = float(np.random.rand() * 2.0 - 1.0)
        wobble = 0.7 * sine_comp + 0.3 * noise_comp
        wobble = max(-1.0, min(1.0, wobble))

        try:
            if param_type == "color":
                base_c = _parse_color(base_val, (128, 128, 128, 255)) or (128, 128, 128, 255)
                mix_target = (255, 255, 255, 255) if wobble > 0 else (0, 0, 0, 255)
                mix_amount = abs(wobble) * _clamp01(amount)
                current_value = _lerp_color(base_c, mix_target, mix_amount)
            elif param_type == "int":
                base_int = _as_int(base_val)
                current_value = base_int + int(round(wobble * amount))
            else:
                base_float = _as_float(base_val)
                current_value = base_float + wobble * amount
        except Exception as e_interp:
            print(f"[WiggleParameter] Error applying wiggle to '{param_name}': {e_interp}", file=sys.stderr)
            current_value = base_val

        try:
            target_block = GFX_REGISTRY.create(tb_name)  # type: ignore
            target_params = {str(param_name): current_value}
            if GfxBaseBlock and isinstance(target_block, GfxBaseBlock):
                return target_block.process(img, ctx.width, ctx.height, params=target_params)
            raise TypeError(f"Target '{tb_name}' is not a valid graphics block.")
        except Exception as e_draw:
            print(f"[WiggleParameter] Error running target block '{tb_name}': {e_draw}", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))

ANIMATION_REGISTRY.register("wiggleparam", WiggleParameter)


# ---------------------------------------------------------------------------
# XYParameter
# ---------------------------------------------------------------------------
@help(
    "Animate two parameters (e.g. cx/cy) together.\n"
    "mode='pingpong' oscillates between start/end using a triangle wave.\n"
    "mode='keyframes' uses keys_x/keys_y keyframe specs."
)
@params({
    "target_block": {"type": "str", "default": ""},
    "x_param":      {"type": "str", "default": "cx"},
    "y_param":      {"type": "str", "default": "cy"},
    "mode":         {"type": "enum", "default": "pingpong", "choices": ["pingpong", "keyframes"]},

    # pingpong extras
    "x_start":      {"type": "float", "default": 0.25, "min": -1e9, "max": 1e9, "step": 0.001},
    "x_end":        {"type": "float", "default": 0.75, "min": -1e9, "max": 1e9, "step": 0.001},
    "y_start":      {"type": "float", "default": 0.25, "min": -1e9, "max": 1e9, "step": 0.001},
    "y_end":        {"type": "float", "default": 0.75, "min": -1e9, "max": 1e9, "step": 0.001},
    "x0":           {"type": "float", "default": None, "nullable": True, "hint": "alias for x_start"},
    "x1":           {"type": "float", "default": None, "nullable": True, "hint": "alias for x_end"},
    "y0":           {"type": "float", "default": None, "nullable": True, "hint": "alias for y_start"},
    "y1":           {"type": "float", "default": None, "nullable": True, "hint": "alias for y_end"},
    "period":       {"type": "float", "default": 2.0, "min": 1e-6, "max": 1e9, "step": 0.01, "unit": "s"},
    "phase":        {"type": "float", "default": 0.0, "min": -1e9, "max": 1e9, "step": 0.01, "unit": "s"},
    "easing":       {"type": "enum", "default": "linear", "choices": EASING_CHOICES},

    # keyframes extras
    "keys_x":       {"type": "any", "default": "", "hint": "e.g. '0:0.2, 1:0.8'"},
    "keys_y":       {"type": "any", "default": "", "hint": "e.g. '0:0.2, 1:0.8'"},
    "normalized":   {"type": "bool", "default": False, "hint": "If true, key times are 0..1; else absolute seconds"},
})
@dataclass
class XYParameter(BaseAnimationBlock):
    """
    Animates two parameters simultaneously.
    """
    _parsed_keys_x: List[Tuple[float, Any]] = field(init=False, default_factory=list)
    _last_keys_spec_x: Any = field(init=False, default=None)
    _parsed_keys_y: List[Tuple[float, Any]] = field(init=False, default_factory=list)
    _last_keys_spec_y: Any = field(init=False, default=None)

    def _get_parsed_keys(self, keys_spec: Any, which: str) -> List[Tuple[float, Any]]:
        if which == "x":
            if keys_spec != self._last_keys_spec_x:
                self._parsed_keys_x = _parse_key_pairs(keys_spec)
                self._last_keys_spec_x = keys_spec
                if not self._parsed_keys_x:
                    print(f"[XYParameter] Warning: No valid keyframes parsed for keys_x: {keys_spec}", file=sys.stderr)
            return self._parsed_keys_x
        else:
            if keys_spec != self._last_keys_spec_y:
                self._parsed_keys_y = _parse_key_pairs(keys_spec)
                self._last_keys_spec_y = keys_spec
                if not self._parsed_keys_y:
                    print(f"[XYParameter] Warning: No valid keyframes parsed for keys_y: {keys_spec}", file=sys.stderr)
            return self._parsed_keys_y

    def process_frame(self, img, ctx, params, engine):
        tb_name = params.get("target_block")
        if not tb_name:
            print("[XYParameter] Warning: Missing target_block. Skipping.", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))

        x_param_name = str(params.get("x_param", "cx"))
        y_param_name = str(params.get("y_param", "cy"))
        mode = str(params.get("mode", "pingpong")).lower()

        val_x: Optional[float] = None
        val_y: Optional[float] = None

        try:
            if mode == "keyframes":
                keys_spec_x = params.get("keys_x")
                keys_spec_y = params.get("keys_y")
                if keys_spec_x is None or keys_spec_y is None:
                    raise ValueError("Keyframes mode requires 'keys_x' and 'keys_y'.")

                use_norm = bool(params.get("normalized", False))
                pairs_x = self._get_parsed_keys(keys_spec_x, "x")
                pairs_y = self._get_parsed_keys(keys_spec_y, "y")
                if not pairs_x or not pairs_y:
                    raise ValueError("No valid keyframes found for X or Y.")

                if use_norm:
                    t = _clamp01(ctx.time / max(1e-9, ctx.duration))
                    sample_pairs_x, sample_pairs_y = pairs_x, pairs_y
                    val_x = _sample_keyframes_float(sample_pairs_x, t)
                    val_y = _sample_keyframes_float(sample_pairs_y, t)
                else:
                    # Normalize X keys to 0..1 over their own span
                    min_t_x, max_t_x = pairs_x[0][0], pairs_x[-1][0]
                    span_x = max(1e-9, max_t_x - min_t_x)
                    tx = _clamp01((ctx.time - min_t_x) / span_x)
                    sx_pairs = [((tt - min_t_x) / span_x, vv) for tt, vv in pairs_x]
                    if sx_pairs:
                        sx_pairs[0] = (0.0, sx_pairs[0][1])
                        sx_pairs[-1] = (1.0, sx_pairs[-1][1])
                    val_x = _sample_keyframes_float(sx_pairs, tx)

                    # Normalize Y keys to 0..1 over their own span
                    min_t_y, max_t_y = pairs_y[0][0], pairs_y[-1][0]
                    span_y = max(1e-9, max_t_y - min_t_y)
                    ty = _clamp01((ctx.time - min_t_y) / span_y)
                    sy_pairs = [((tt - min_t_y) / span_y, vv) for tt, vv in pairs_y]
                    if sy_pairs:
                        sy_pairs[0] = (0.0, sy_pairs[0][1])
                        sy_pairs[-1] = (1.0, sy_pairs[-1][1])
                    val_y = _sample_keyframes_float(sy_pairs, ty)

            else:
                # pingpong defaults; allow aliases x0/x1/y0/y1
                x_start = _as_float(params.get("x_start", params.get("x0", 0.25)))
                x_end = _as_float(params.get("x_end", params.get("x1", 0.75)))
                y_start = _as_float(params.get("y_start", params.get("y0", 0.25)))
                y_end = _as_float(params.get("y_end", params.get("y1", 0.75)))
                period = max(1e-9, _as_float(params.get("period", 2.0)))
                phase = _as_float(params.get("phase", 0.0))
                easing_func = _get_easing(params.get("easing", "linear"))

                time_shifted = max(0.0, ctx.time + phase)
                cycle_pos = (time_shifted % period) / period
                tri = 1.0 - abs(2.0 * cycle_pos - 1.0)
                t_eased = _clamp01(easing_func(tri))

                val_x = _lerp(x_start, x_end, t_eased)
                val_y = _lerp(y_start, y_end, t_eased)

        except Exception as e_calc:
            print(f"[XYParameter] Error calculating values: {e_calc}", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))

        if val_x is not None and val_y is not None:
            try:
                target_block = GFX_REGISTRY.create(tb_name)  # type: ignore
                target_params = {x_param_name: val_x, y_param_name: val_y}
                if GfxBaseBlock and isinstance(target_block, GfxBaseBlock):
                    return target_block.process(img, ctx.width, ctx.height, params=target_params)
                raise TypeError(f"Target '{tb_name}' is not a valid graphics block.")
            except Exception as e_draw:
                print(f"[XYParameter] Error running target block '{tb_name}': {e_draw}", file=sys.stderr)

        return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0, 0, 0, 0))

    def get_overrides(
            self,
            ctx: AnimationContext,
            params: Dict[str, Any],
            engine,
    ) -> tuple[str, Dict[str, Any]]:
        tb_name = params.get("target_block")
        if not tb_name:
            return "", {}

        x_param_name = str(params.get("x_param", "cx"))
        y_param_name = str(params.get("y_param", "cy"))
        mode = str(params.get("mode", "pingpong")).lower()

        val_x: Optional[float] = None
        val_y: Optional[float] = None

        try:
            if mode == "keyframes":
                keys_spec_x = params.get("keys_x")
                keys_spec_y = params.get("keys_y")
                if keys_spec_x is None or keys_spec_y is None:
                    # keyframes mode requires both
                    return str(tb_name).strip().lower(), {}

                use_norm = bool(params.get("normalized", False))

                pairs_x = self._get_parsed_keys(keys_spec_x, "x")
                pairs_y = self._get_parsed_keys(keys_spec_y, "y")
                if not pairs_x or not pairs_y:
                    return str(tb_name).strip().lower(), {}

                if use_norm:
                    # normalized keys: time is 0..1 over duration
                    t = _clamp01(ctx.time / max(1e-9, ctx.duration))
                    val_x = _sample_keyframes_float(pairs_x, t)
                    val_y = _sample_keyframes_float(pairs_y, t)
                else:
                    # absolute time keys, normalized per axis span (same as process_frame)
                    min_t_x, max_t_x = pairs_x[0][0], pairs_x[-1][0]
                    span_x = max(1e-9, max_t_x - min_t_x)
                    tx = _clamp01((ctx.time - min_t_x) / span_x)
                    sx_pairs = [((tt - min_t_x) / span_x, vv) for tt, vv in pairs_x]
                    if sx_pairs:
                        sx_pairs[0] = (0.0, sx_pairs[0][1])
                        sx_pairs[-1] = (1.0, sx_pairs[-1][1])
                    val_x = _sample_keyframes_float(sx_pairs, tx)

                    min_t_y, max_t_y = pairs_y[0][0], pairs_y[-1][0]
                    span_y = max(1e-9, max_t_y - min_t_y)
                    ty = _clamp01((ctx.time - min_t_y) / span_y)
                    sy_pairs = [((tt - min_t_y) / span_y, vv) for tt, vv in pairs_y]
                    if sy_pairs:
                        sy_pairs[0] = (0.0, sy_pairs[0][1])
                        sy_pairs[-1] = (1.0, sy_pairs[-1][1])
                    val_y = _sample_keyframes_float(sy_pairs, ty)

            else:
                # pingpong defaults; allow aliases x0/x1/y0/y1
                x_start = _as_float(params.get("x_start", params.get("x0", 0.25)))
                x_end = _as_float(params.get("x_end", params.get("x1", 0.75)))
                y_start = _as_float(params.get("y_start", params.get("y0", 0.25)))
                y_end = _as_float(params.get("y_end", params.get("y1", 0.75)))

                period = max(1e-9, _as_float(params.get("period", 2.0)))
                phase = _as_float(params.get("phase", 0.0))
                easing_func = _get_easing(params.get("easing", "linear"))

                time_shifted = max(0.0, ctx.time + phase)
                cycle_pos = (time_shifted % period) / period
                tri = 1.0 - abs(2.0 * cycle_pos - 1.0)
                t_eased = _clamp01(easing_func(tri))

                val_x = _lerp(x_start, x_end, t_eased)
                val_y = _lerp(y_start, y_end, t_eased)

        except Exception as e_calc:
            print(f"[XYParameter.get_overrides] Error calculating values: {e_calc}", file=sys.stderr)
            return str(tb_name).strip().lower(), {}

        if val_x is None or val_y is None:
            return str(tb_name).strip().lower(), {}

        return str(tb_name).strip().lower(), {x_param_name: val_x, y_param_name: val_y}
ANIMATION_REGISTRY.register("xyparam", XYParameter)
