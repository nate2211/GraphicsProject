# animations.py
from __future__ import annotations

import math
import sys
import os # Added for font path check
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

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
            # Instantiate the dummy class or a generic object
            return self._by_name[key](**kwargs) # type: ignore

        def names(self) -> List[str]:
            return sorted(self._by_name.keys())

    GFX_REGISTRY = BlockRegistry()  # type: ignore
    GraphicsEngine = object         # type: ignore

    def _auto(v: Any) -> Any:  # type: ignore
        # Minimal string conversion for dummy
        if isinstance(v, (int, float, bool)): return v
        return str(v)

    def _parse_color(v: Any, default: Optional[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]: # type: ignore
        # Dummy color parser, just returns default
        return default if default is not None else (255, 255, 255, 255)

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

# Add more: cubic, expo, circ, back, elastic, bounce...
# Example: easeInOutCubic
def ease_in_out_cubic(t: float) -> float:
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2

EASING_FUNCTIONS: Dict[str, Any] = {
    "linear": ease_linear,
    "in_quad": ease_in_quad, "quadin": ease_in_quad,
    "out_quad": ease_out_quad, "quadout": ease_out_quad,
    "in_out_quad": ease_in_out_quad, "quadinout": ease_in_out_quad,
    "in_sine": ease_in_sine, "sinein": ease_in_sine,
    "out_sine": ease_out_sine, "sineout": ease_out_sine,
    "in_out_sine": ease_in_out_sine, "sineinout": ease_in_out_sine,
    "in_out_cubic": ease_in_out_cubic, "cubicinout": ease_in_out_cubic,
}

# ---------------------------------------------------------------------------
# Small math/parsing helpers
# ---------------------------------------------------------------------------
def _clamp01(x: float) -> float:
    # Optimized clamp
    return max(0.0, min(1.0, x))

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def _lerp_color(a_rgba: Tuple[int, int, int, int],
                b_rgba: Tuple[int, int, int, int],
                t: float) -> Tuple[int, int, int, int]:
    # Use numpy for potentially faster vectorized lerp if called often
    aa = np.array(a_rgba, dtype=np.float32)
    bb = np.array(b_rgba, dtype=np.float32)
    cc = aa + (bb - aa) * t
    # Clip, round, convert to int tuple
    c = np.clip(cc, 0, 255).round().astype(np.int32)
    return (int(c[0]), int(c[1]), int(c[2]), int(c[3]))

# More robust numeric conversion, handles None gracefully
def _as_float(v: Any, default: float = 0.0) -> float:
    if v is None: return default
    if isinstance(v, (int, float)): return float(v)
    try: return float(_auto(str(v))) # Ensure string input to _auto
    except (ValueError, TypeError): return default

def _as_int(v: Any, default: int = 0) -> int:
    if v is None: return default
    if isinstance(v, int): return v
    if isinstance(v, float): return int(round(v))
    try: return int(round(float(_auto(str(v)))))
    except (ValueError, TypeError): return default

def _get_easing(name: Any, default=ease_linear) -> Callable[[float], float]:
    # Ensure name is string before lower()
    try: s = str(name).strip().lower()
    except: s = ""
    return EASING_FUNCTIONS.get(s, default)

def _parse_key_pairs(spec: Any) -> List[Tuple[float, Any]]:
    """
    Parses keyframe specifier string "t:v, t:v, ..." or list [(t, v), ...].
    Returns sorted list of (time_float, raw_value) tuples.
    Values are kept raw (usually strings) for later type-specific sampling.
    """
    pairs: List[Tuple[float, Any]] = []
    if spec is None: return pairs

    items_to_parse: List[Any] = []
    if isinstance(spec, str):
        items_to_parse = [s.strip() for s in spec.split(',') if s.strip()]
    elif isinstance(spec, (list, tuple)):
        items_to_parse = spec # Assume it's already list/tuple of pairs or strings

    for item in items_to_parse:
        t_float: Optional[float] = None
        v_raw: Any = None

        if isinstance(item, str):
            if ":" in item:
                t_str, v_str = item.split(":", 1)
                try: t_float = float(t_str.strip())
                except ValueError: continue # Skip if time part isn't float
                v_raw = v_str.strip()
            else: # String without ':', treat as value at t=1.0
                t_float = 1.0
                v_raw = item.strip()
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            t_raw, v_raw = item
            try: t_float = float(_auto(t_raw)) # Coerce time part
            except (ValueError, TypeError): continue # Skip if time isn't numeric
        else:
            continue # Skip invalid item format

        # Add valid pair
        if t_float is not None:
             pairs.append((t_float, v_raw))

    # Sort pairs by time
    pairs.sort(key=lambda kv: kv[0])
    return pairs


def _sample_keyframes_float(pairs: List[Tuple[float, Any]], tnorm: float, default: float = 0.0) -> float:
    """Samples float keyframes at normalized time tnorm [0, 1]."""
    if not pairs: return default
    t = _clamp01(tnorm)

    # Find bounding keyframes
    prev_t, prev_v_raw = pairs[0]
    next_t, next_v_raw = pairs[-1]

    if t <= prev_t: return _as_float(prev_v_raw, default)
    if t >= next_t: return _as_float(next_v_raw, default)

    for i in range(len(pairs) - 1):
        t0, v0_raw = pairs[i]
        t1, v1_raw = pairs[i + 1]
        if t0 <= t <= t1:
            # Found the interval
            v0 = _as_float(v0_raw, default)
            v1 = _as_float(v1_raw, default)
            # Normalize time within this interval
            interval_dur = t1 - t0
            u = (t - t0) / max(1e-9, interval_dur) if interval_dur > 1e-9 else 0.0
            return _lerp(v0, v1, u)

    # Should be unreachable if t is clamped and list sorted, but return last value as fallback
    return _as_float(next_v_raw, default)


def _sample_keyframes_color(pairs: List[Tuple[float, Any]], tnorm: float, default: Tuple[int,int,int,int] = (0,0,0,255)) -> Tuple[int, int, int, int]:
    """Samples color keyframes at normalized time tnorm [0, 1]."""
    if not pairs: return default
    t = _clamp01(tnorm)

    # Find bounding keyframes
    prev_t, prev_v_raw = pairs[0]
    next_t, next_v_raw = pairs[-1]

    if t <= prev_t: return _parse_color(prev_v_raw, default) or default
    if t >= next_t: return _parse_color(next_v_raw, default) or default

    for i in range(len(pairs) - 1):
        t0, v0_raw = pairs[i]
        t1, v1_raw = pairs[i + 1]
        if t0 <= t <= t1:
            # Found interval
            c0 = _parse_color(v0_raw, default) or default
            c1 = _parse_color(v1_raw, default) or default
            interval_dur = t1 - t0
            u = (t - t0) / max(1e-9, interval_dur) if interval_dur > 1e-9 else 0.0
            return _lerp_color(c0, c1, u)

    # Fallback
    return _parse_color(next_v_raw, default) or default

# ---------------------------------------------------------------------------
# Animation Context and Base Block
# ---------------------------------------------------------------------------
@dataclass
class AnimationContext:
    frame: int = 0
    total_frames: int = 1
    time: float = 0.0         # Current time in seconds
    duration: float = 1.0     # Total animation duration in seconds
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
        engine: "GraphicsEngine", # Allow access to engine if needed
    ) -> Image.Image:
        """Process one frame. Input img might be None."""
        raise NotImplementedError

# Animation Registry Instance
ANIMATION_REGISTRY = BlockRegistry()

# ---------------------------------------------------------------------------
# AnimateParameter: Simple Lerp between two values over time
# ---------------------------------------------------------------------------
@dataclass
class AnimateParameter(BaseAnimationBlock):
    """
    Interpolates a single parameter of a target graphics block.
    Params:
      target_block: str (Name of graphics block)
      param_name:   str (Parameter to animate)
      start_value:  Any (Value at start_time)
      end_value:    Any (Value at end_time)
      start_time:   float (Seconds, default 0)
      end_time:     float (Seconds, default animation duration)
      param_type:   'float'|'int'|'color' (default 'float')
      easing:       str (Easing function name, default 'linear')
    """
    def process_frame(self, img, ctx, params, engine):
        tb_name = params.get("target_block")
        param_name = params.get("param_name")
        start_val = params.get("start_value")
        end_val = params.get("end_value")

        if None in (tb_name, param_name, start_val, end_val):
            print(f"[AnimateParameter] Warning: Missing required params (target_block='{tb_name}', param_name='{param_name}', start/end_value missing?). Skipping.", file=sys.stderr)
            # Return input image or a blank one if input is None
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0,0,0,0))

        start_time = _as_float(params.get("start_time", 0.0))
        end_time = _as_float(params.get("end_time", ctx.duration))
        param_type = str(params.get("param_type", "float")).lower()
        easing_func = _get_easing(params.get("easing", "linear"))

        # Calculate normalized time within the animation segment, apply easing
        segment_duration = max(1e-9, end_time - start_time)
        relative_time = ctx.time - start_time
        t_raw = _clamp01(relative_time / segment_duration)
        t_eased = _clamp01(easing_func(t_raw))

        # Interpolate based on type
        current_value: Any
        try:
            if param_type == "color":
                c0 = _parse_color(start_val, (0,0,0,255)) or (0,0,0,255)
                c1 = _parse_color(end_val, (255,255,255,255)) or (255,255,255,255)
                current_value = _lerp_color(c0, c1, t_eased)
            elif param_type == "int":
                v0, v1 = _as_int(start_val), _as_int(end_val)
                current_value = int(round(_lerp(v0, v1, t_eased)))
            else: # Default float
                v0, v1 = _as_float(start_val), _as_float(end_val)
                current_value = _lerp(v0, v1, t_eased)
        except Exception as e_interp:
            print(f"[AnimateParameter] Error interpolating '{param_name}' ({param_type}): {e_interp}. Using start value.", file=sys.stderr)
            current_value = start_val # Fallback to start value

        # Execute the target graphics block
        try:
            # Create a *new instance* of the graphics block each frame? Or reuse?
            # Reusing might be needed if the graphics block has state, but simpler to recreate.
            target_block = GFX_REGISTRY.create(tb_name) # type: ignore
            # Prepare params: use only the animated parameter for simplicity
            # More complex: merge with other extras for this block?
            target_params = {param_name: current_value}
            # Ensure target_block is the correct type
            if GfxBaseBlock and isinstance(target_block, GfxBaseBlock):
                 return target_block.process(img, ctx.width, ctx.height, params=target_params)
            else:
                 raise TypeError(f"Target '{tb_name}' is not a valid graphics block.")
        except Exception as e_draw:
            print(f"[AnimateParameter] Error running target block '{tb_name}': {e_draw}", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0,0,0,0))

ANIMATION_REGISTRY.register("animateparam", AnimateParameter)


# ---------------------------------------------------------------------------
# PingPongParameter: Oscillate a parameter using a triangle wave
# ---------------------------------------------------------------------------
@dataclass
class PingPongParameter(BaseAnimationBlock):
    """
    Oscillates a parameter between start and end values using a triangle wave.
    Params:
      target_block: str
      param_name:   str
      start_value:  Any
      end_value:    Any
      period:       float (Seconds for a full back-and-forth cycle, default 2.0)
      phase:        float (Time offset in seconds, default 0.0)
      param_type:   'float'|'int'|'color' (default 'float')
      easing:       str (Easing applied to the 0..1 triangle wave output, default 'linear')
    """
    def process_frame(self, img, ctx, params, engine):
        tb_name = params.get("target_block")
        param_name = params.get("param_name")
        start_val = params.get("start_value")
        end_val = params.get("end_value")

        if None in (tb_name, param_name, start_val, end_val):
            print(f"[PingPongParameter] Warning: Missing required params. Skipping.", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0,0,0,0))

        period = max(1e-9, _as_float(params.get("period", 2.0)))
        phase = _as_float(params.get("phase", 0.0))
        param_type = str(params.get("param_type", "float")).lower()
        easing_func = _get_easing(params.get("easing", "linear"))

        # Calculate triangle wave position (0..1)
        time_shifted = max(0.0, ctx.time + phase) # Apply phase shift
        cycle_pos = (time_shifted % period) / period # Position within the cycle [0, 1)
        triangle_val = 1.0 - abs(2.0 * cycle_pos - 1.0) # Map [0, 1) -> triangle wave [0, 1]

        # Apply easing to the triangle wave output
        t_eased = _clamp01(easing_func(triangle_val))

        # Interpolate based on type using the eased triangle value
        current_value: Any
        try:
            if param_type == "color":
                c0 = _parse_color(start_val, (0,0,0,255)) or (0,0,0,255)
                c1 = _parse_color(end_val, (255,255,255,255)) or (255,255,255,255)
                current_value = _lerp_color(c0, c1, t_eased)
            elif param_type == "int":
                v0, v1 = _as_int(start_val), _as_int(end_val)
                current_value = int(round(_lerp(v0, v1, t_eased)))
            else: # Default float
                v0, v1 = _as_float(start_val), _as_float(end_val)
                current_value = _lerp(v0, v1, t_eased)
        except Exception as e_interp:
            print(f"[PingPongParameter] Error interpolating '{param_name}': {e_interp}. Using start value.", file=sys.stderr)
            current_value = start_val

        # Execute target graphics block
        try:
            target_block = GFX_REGISTRY.create(tb_name) # type: ignore
            target_params = {param_name: current_value} # Pass only animated param
            if GfxBaseBlock and isinstance(target_block, GfxBaseBlock):
                 # --- FIX: Pass the correct parameters ---
                 # The error 'name xn is not defined' happened because the previous
                 # version incorrectly tried to use variables xn, yn etc.
                 # We need to pass the actual parameter name and calculated value.
                 # _acc_update is not needed if we just pass the single calculated param.
                 # merged = _acc_update(img, tb_name, {param_name: current_value}) # _acc_update was complex, simplify
                 return target_block.process(img, ctx.width, ctx.height, params=target_params)
            else:
                 raise TypeError(f"Target '{tb_name}' is not a valid graphics block.")
        except Exception as e_draw:
            # Print the specific NameError or other exception
            print(f"[PingPongParameter] Error running target block '{tb_name}': {e_draw}", file=sys.stderr)
            # import traceback; traceback.print_exc() # Uncomment for full traceback
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0,0,0,0))

ANIMATION_REGISTRY.register("pingpongparam", PingPongParameter)


# ---------------------------------------------------------------------------
# KeyframeParameter: Interpolate through multiple time:value pairs
# ---------------------------------------------------------------------------
@dataclass
class KeyframeParameter(BaseAnimationBlock):
    """
    Interpolates a parameter based on keyframes "t:v, t:v, ...".
    Params:
      target_block: str
      param_name:   str
      keys:         str or list (Keyframe specification)
      normalized:   bool (If true, key times are 0..1 relative to duration, else absolute seconds. Default false)
      param_type:   'float'|'int'|'color' (default 'float')
    """
    # Cache parsed keyframes for efficiency
    _parsed_keys: List[Tuple[float, Any]] = field(init=False, default_factory=list)
    _last_keys_spec: Any = field(init=False, default=None)

    def _get_parsed_keys(self, keys_spec: Any) -> List[Tuple[float, Any]]:
        """Parse keys spec only if it has changed."""
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

        if None in (tb_name, param_name, keys_spec):
            print(f"[KeyframeParameter] Warning: Missing required params. Skipping.", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0,0,0,0))

        param_type = str(params.get("param_type", "float")).lower()
        use_normalized_time = str(params.get("normalized", "false")).lower() in ('true', '1', 'yes')

        pairs = self._get_parsed_keys(keys_spec)
        if not pairs:
            # Return input or blank if no valid keys
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0,0,0,0))

        # Determine the normalized time [0, 1] to sample at
        if use_normalized_time:
            tnorm = _clamp01(ctx.time / max(1e-9, ctx.duration))
            # Key times in 'pairs' are already assumed to be 0..1
            sample_time = tnorm
            sample_pairs = pairs
        else:
            # Absolute time - normalize the *current time* relative to the *span* of the keys
            # Or should keys define the normalization range?
            # Let's assume keys times are absolute seconds. We sample at ctx.time.
            # We need to find the interval in absolute time.

            # Find interval based on absolute time ctx.time
            # Modify _sample functions to take absolute time and key times?
            # Easier: Normalize key times based on duration if !use_normalized_time
            # No, let _sample handle it based on tnorm. If times are absolute,
            # we need to map ctx.time to the range defined by keys.

            # Alternative: Assume keys time are relative 0..1 UNLESS normalized=false
            # If normalized=false, interpret key times as absolute seconds.
            # Then find where ctx.time fits.

            # Let's stick to normalized sampling time tnorm [0,1] for _sample functions,
            # and adjust how tnorm is derived or how keys are pre-processed.

            # If key times are absolute seconds:
            if not use_normalized_time:
                 # Find min/max time in keys to define the span for normalization
                 min_key_time = pairs[0][0]
                 max_key_time = pairs[-1][0]
                 key_span = max(1e-9, max_key_time - min_key_time)
                 # Normalize current ctx.time relative to this key span
                 sample_time = _clamp01((ctx.time - min_key_time) / key_span)
                 # Sample functions expect key times to be 0..1, so normalize pairs:
                 sample_pairs = [((t - min_key_time) / key_span, v) for t, v in pairs]
                 # Ensure first key is at 0 and last at 1 after normalization
                 if sample_pairs:
                      sample_pairs[0] = (0.0, sample_pairs[0][1])
                      sample_pairs[-1] = (1.0, sample_pairs[-1][1])

            else: # Key times are already normalized 0..1
                sample_time = _clamp01(ctx.time / max(1e-9, ctx.duration))
                sample_pairs = pairs # Use keys directly


        # Sample the value
        current_value: Any
        try:
            if param_type == "color":
                current_value = _sample_keyframes_color(sample_pairs, sample_time)
            elif param_type == "int":
                val_float = _sample_keyframes_float(sample_pairs, sample_time)
                current_value = int(round(val_float))
            else: # Default float
                current_value = _sample_keyframes_float(sample_pairs, sample_time)
        except Exception as e_sample:
            print(f"[KeyframeParameter] Error sampling '{param_name}': {e_sample}. Using first key value.", file=sys.stderr)
            current_value = pairs[0][1] # Fallback to first value

        # Execute target graphics block
        try:
            target_block = GFX_REGISTRY.create(tb_name) # type: ignore
            target_params = {param_name: current_value}
            if GfxBaseBlock and isinstance(target_block, GfxBaseBlock):
                return target_block.process(img, ctx.width, ctx.height, params=target_params)
            else:
                 raise TypeError(f"Target '{tb_name}' is not a valid graphics block.")
        except Exception as e_draw:
            print(f"[KeyframeParameter] Error running target block '{tb_name}': {e_draw}", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0,0,0,0))

ANIMATION_REGISTRY.register("keyframeparam", KeyframeParameter)


# ---------------------------------------------------------------------------
# WiggleParameter: Add pseudo-random noise to a parameter
# ---------------------------------------------------------------------------
@dataclass
class WiggleParameter(BaseAnimationBlock):
    """
    Adds time-varying noise (wiggle) to a base parameter value.
    Params:
      target_block: str
      param_name:   str
      base_value:   Any (The value around which to wiggle)
      amount:       float (Magnitude of wiggle for float/int)
                    float (Mix factor [0..1] towards white/black for color)
      freq_hz:      float (Speed of wiggle, default 2.0 Hz)
      param_type:   'float'|'int'|'color' (default 'float')
      seed:         int (Optional, for deterministic wiggle based on frame+seed)
    """
    def process_frame(self, img, ctx, params, engine):
        tb_name = params.get("target_block")
        param_name = params.get("param_name")
        base_val = params.get("base_value", params.get("base")) # Allow 'base' alias

        if None in (tb_name, param_name, base_val):
            print(f"[WiggleParameter] Warning: Missing required params. Skipping.", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0,0,0,0))

        param_type = str(params.get("param_type", "float")).lower()
        freq = _as_float(params.get("freq_hz", 2.0))
        amount = _as_float(params.get("amount", 0.1))
        seed = params.get("seed") # Keep as Any initially

        # Generate pseudo-random value [-1, 1] based on time (and optionally seed)
        # Combine sine wave with deterministic noise based on frame/seed
        combined_seed = ctx.frame
        if seed is not None:
            try: combined_seed += int(seed)
            except: pass # Ignore non-integer seeds
        np.random.seed(combined_seed & 0xFFFFFFFF) # Seed RNG for this frame

        # Blend sine wave with frame-based random noise for less repetitive wiggle
        # Higher freq -> faster sine, noise adds jitter
        time_phase = 2 * math.pi * freq * ctx.time
        # Use two sines with slightly different frequencies + noise
        sine_comp = 0.6 * math.sin(time_phase) + 0.4 * math.sin(time_phase * 1.37 + 1.23)
        noise_comp = np.random.rand() * 2.0 - 1.0
        # Blend sine and noise, scale to [-1, 1]
        wobble_norm = (0.7 * sine_comp + 0.3 * noise_comp) / 1.0 # Rough normalization
        wobble_norm = max(-1.0, min(1.0, wobble_norm))

        # Apply wobble to base value based on type
        current_value: Any
        try:
            if param_type == "color":
                base_c = _parse_color(base_val, (128,128,128,255)) or (128,128,128,255)
                # Wiggle color by mixing with white/black based on wobble sign/magnitude
                mix_target = (255,255,255,255) if wobble_norm > 0 else (0,0,0,255)
                mix_amount = abs(wobble_norm) * _clamp01(amount) # Amount controls max mix
                current_value = _lerp_color(base_c, mix_target, mix_amount)
            elif param_type == "int":
                base_int = _as_int(base_val)
                wiggle_delta = int(round(wobble_norm * amount)) # Amount is delta range
                current_value = base_int + wiggle_delta
            else: # Default float
                base_float = _as_float(base_val)
                wiggle_delta = wobble_norm * amount # Amount is delta range
                current_value = base_float + wiggle_delta
        except Exception as e_interp:
            print(f"[WiggleParameter] Error applying wiggle to '{param_name}': {e_interp}. Using base value.", file=sys.stderr)
            current_value = base_val

        # Execute target block
        try:
            target_block = GFX_REGISTRY.create(tb_name) # type: ignore
            target_params = {param_name: current_value}
            if GfxBaseBlock and isinstance(target_block, GfxBaseBlock):
                return target_block.process(img, ctx.width, ctx.height, params=target_params)
            else:
                 raise TypeError(f"Target '{tb_name}' is not a valid graphics block.")
        except Exception as e_draw:
            print(f"[WiggleParameter] Error running target block '{tb_name}': {e_draw}", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0,0,0,0))

ANIMATION_REGISTRY.register("wiggleparam", WiggleParameter)


# ---------------------------------------------------------------------------
# XYParameter: Animate two parameters (e.g., position) simultaneously
# ---------------------------------------------------------------------------
@dataclass
class XYParameter(BaseAnimationBlock):
    """
    Animates two parameters (e.g., 'cx', 'cy') using either pingpong or keyframes.
    Params:
      target_block: str
      x_param:      str (Name of the first parameter, default 'cx')
      y_param:      str (Name of the second parameter, default 'cy')
      mode:         'pingpong' (default) | 'keyframes'

    PingPong Mode Extras:
      x_start, x_end: Values for X parameter
      y_start, y_end: Values for Y parameter
      period:       float (Seconds for full cycle, default 2.0)
      easing:       str (Easing function name, default 'linear')
      phase:        float (Time offset in seconds, default 0.0)

    Keyframes Mode Extras:
      keys_x:       str or list (Keyframes for X parameter, "t:v,...")
      keys_y:       str or list (Keyframes for Y parameter, "t:v,...")
      normalized:   bool (Key times are 0..1 or absolute seconds, default false)
    """
    # Cache parsed keyframes
    _parsed_keys_x: List[Tuple[float, Any]] = field(init=False, default_factory=list)
    _last_keys_spec_x: Any = field(init=False, default=None)
    _parsed_keys_y: List[Tuple[float, Any]] = field(init=False, default_factory=list)
    _last_keys_spec_y: Any = field(init=False, default=None)

    def _get_parsed_keys(self, keys_spec: Any, which: str) -> List[Tuple[float, Any]]:
        """Get or parse keys for X or Y."""
        last_spec = self._last_keys_spec_x if which == 'x' else self._last_keys_spec_y
        if keys_spec != last_spec:
            parsed = _parse_key_pairs(keys_spec)
            if which == 'x':
                self._parsed_keys_x = parsed
                self._last_keys_spec_x = keys_spec
            else:
                self._parsed_keys_y = parsed
                self._last_keys_spec_y = keys_spec
            if not parsed:
                 print(f"[XYParameter] Warning: No valid keyframes parsed for keys_{which}: {keys_spec}", file=sys.stderr)
            return parsed
        return self._parsed_keys_x if which == 'x' else self._parsed_keys_y

    def process_frame(self, img, ctx, params, engine):
        tb_name = params.get("target_block")
        x_param_name = params.get("x_param", "cx") # Default to common position names
        y_param_name = params.get("y_param", "cy")
        mode = str(params.get("mode", "pingpong")).lower()

        if not tb_name:
            print(f"[XYParameter] Warning: Missing target_block. Skipping.", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0,0,0,0))

        val_x: Any = None
        val_y: Any = None

        try:
            if mode == "keyframes":
                keys_spec_x = params.get("keys_x")
                keys_spec_y = params.get("keys_y")
                if keys_spec_x is None or keys_spec_y is None:
                    raise ValueError("Keyframes mode requires 'keys_x' and 'keys_y' parameters.")

                use_normalized_time = str(params.get("normalized", "false")).lower() in ('true', '1', 'yes')
                pairs_x = self._get_parsed_keys(keys_spec_x, 'x')
                pairs_y = self._get_parsed_keys(keys_spec_y, 'y')

                if not pairs_x or not pairs_y: # Need both sets of keys
                     raise ValueError("No valid keyframes found for X or Y.")

                # Determine sampling time (normalized 0..1)
                if use_normalized_time:
                    sample_time = _clamp01(ctx.time / max(1e-9, ctx.duration))
                    sample_pairs_x, sample_pairs_y = pairs_x, pairs_y
                else: # Absolute time keys
                    min_t_x, max_t_x = pairs_x[0][0], pairs_x[-1][0]
                    span_x = max(1e-9, max_t_x - min_t_x)
                    sample_time_x = _clamp01((ctx.time - min_t_x) / span_x)
                    sample_pairs_x = [((t - min_t_x) / span_x, v) for t, v in pairs_x]
                    if sample_pairs_x: sample_pairs_x[0] = (0.0, sample_pairs_x[0][1]); sample_pairs_x[-1] = (1.0, sample_pairs_x[-1][1])

                    min_t_y, max_t_y = pairs_y[0][0], pairs_y[-1][0]
                    span_y = max(1e-9, max_t_y - min_t_y)
                    sample_time_y = _clamp01((ctx.time - min_t_y) / span_y)
                    sample_pairs_y = [((t - min_t_y) / span_y, v) for t, v in pairs_y]
                    if sample_pairs_y: sample_pairs_y[0] = (0.0, sample_pairs_y[0][1]); sample_pairs_y[-1] = (1.0, sample_pairs_y[-1][1])

                    # Sample using potentially different normalized times if spans differ
                    val_x = _sample_keyframes_float(sample_pairs_x, sample_time_x)
                    val_y = _sample_keyframes_float(sample_pairs_y, sample_time_y)

                # If times were normalized, sample using the single tnorm
                if use_normalized_time:
                     val_x = _sample_keyframes_float(sample_pairs_x, sample_time)
                     val_y = _sample_keyframes_float(sample_pairs_y, sample_time)

            else: # Default: pingpong mode
                x_start = _as_float(params.get("x_start", params.get("x0", 0.25))) # Allow x0 alias
                x_end = _as_float(params.get("x_end", params.get("x1", 0.75)))
                y_start = _as_float(params.get("y_start", params.get("y0", 0.25)))
                y_end = _as_float(params.get("y_end", params.get("y1", 0.75)))
                period = max(1e-9, _as_float(params.get("period", 2.0)))
                phase = _as_float(params.get("phase", 0.0))
                easing_func = _get_easing(params.get("easing", "linear"))

                time_shifted = max(0.0, ctx.time + phase)
                cycle_pos = (time_shifted % period) / period
                triangle_val = 1.0 - abs(2.0 * cycle_pos - 1.0)
                t_eased = _clamp01(easing_func(triangle_val))

                val_x = _lerp(x_start, x_end, t_eased)
                val_y = _lerp(y_start, y_end, t_eased)

        except Exception as e_calc:
            print(f"[XYParameter] Error calculating values: {e_calc}. Skipping.", file=sys.stderr)
            return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0,0,0,0))

        # Execute target block if values were calculated
        if val_x is not None and val_y is not None:
            try:
                target_block = GFX_REGISTRY.create(tb_name) # type: ignore
                target_params = {x_param_name: val_x, y_param_name: val_y}
                if GfxBaseBlock and isinstance(target_block, GfxBaseBlock):
                    return target_block.process(img, ctx.width, ctx.height, params=target_params)
                else:
                    raise TypeError(f"Target '{tb_name}' is not a valid graphics block.")
            except Exception as e_draw:
                print(f"[XYParameter] Error running target block '{tb_name}': {e_draw}", file=sys.stderr)
                # Fall through to return existing image or blank

        # Fallback return
        return img if img is not None else Image.new("RGBA", (ctx.width, ctx.height), (0,0,0,0))

ANIMATION_REGISTRY.register("xyparam", XYParameter)

