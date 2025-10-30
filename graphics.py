from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageEnhance

# =============== Registry ===============
class BlockRegistry:
    def __init__(self) -> None:
        self._by_name: Dict[str, type[BaseBlock]] = {}

    def register(self, name: str, cls: type["BaseBlock"]) -> None:
        key = name.strip().lower()
        self._by_name[key] = cls

    def names(self) -> List[str]:
        return sorted(self._by_name.keys())

    def create(self, name: str, **kwargs) -> "BaseBlock":
        key = name.strip().lower()
        if key not in self._by_name:
            msg = f"Unknown block '{name}'. Available: {', '.join(self.names()) or '(none)'}"
            raise KeyError(msg)
        return self._by_name[key](**kwargs)

REGISTRY = BlockRegistry()

# =============== Base & utils ===============
@dataclass
class BaseBlock:
    def process(
        self,
        img: Optional[Image.Image],
        width: int,
        height: int,
        *,
        params: Dict[str, Any],
    ) -> Image.Image:
        """Process or generate an image.

        img: None for a source (e.g. solidcolor), PIL Image for effects.
        Returns PIL Image object (RGBA).
        """
        raise NotImplementedError

# ---------------- type parsing for --extra ----------------
def _auto(v: str) -> Any:
    # Basic type coercion (int, float, bool, string)
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except Exception:
        # comma-list -> list of auto-typed items
        if "," in s:
            return [_auto(x) for x in s.split(",")]
        return s # Keep as string if no other type matches

def parse_extras(kvs: List[str]) -> Dict[str, Dict[str, Any]]:
    # Parses ["block.param=value", ...] into {"block": {"param": value}}
    out: Dict[str, Dict[str, Any]] = {}
    for item in kvs:
        if not item:
            continue
        if "=" not in item:
            key, val = item, "true" # Treat as flag
        else:
            key, val = item.split("=", 1)
        key = key.strip()
        val_t = _auto(val)
        if "." in key:
            blk, par = key.split(".", 1)
            out.setdefault(blk.strip().lower(), {})[par.strip()] = val_t
        else:
            # Stash top-level keys anyway (e.g., global settings)
            out.setdefault(key.strip().lower(), {})[""] = val_t
    return out

# ---------------- common helpers ----------------

def _parse_color(val: Any, default: Optional[Tuple[int, int, int, int]] = (0, 0, 0, 255)) -> Optional[Tuple[int, int, int, int]]:
    # Parses color string like "R,G,B,A", "R,G,B", "#RGB", "#RRGGBB", "#RRGGBBAA", or named colors
    # Returns None if default is None and parsing fails
    if val is None:
        return default
    if isinstance(val, (list, tuple)) and len(val) >= 3:
        r, g, b = int(val[0]), int(val[1]), int(val[2])
        a = int(val[3]) if len(val) > 3 else 255
        return (r, g, b, a)
    if isinstance(val, str):
        val = val.strip()
        if val.lower() == 'none': return default # Allow explicit 'none' string
        if val.startswith("#"):
            h = val.lstrip("#")
            try:
                if len(h) == 3: h = h[0]*2 + h[1]*2 + h[2]*2 + "FF"
                elif len(h) == 4: h = h[0]*2 + h[1]*2 + h[2]*2 + h[3]*2
                elif len(h) == 6: h += "FF"
                elif len(h) == 8: pass
                else: return default
                return tuple(int(h[i:i+2], 16) for i in (0, 2, 4, 6))
            except ValueError: return default # Handle invalid hex
        if "," in val:
            parts = [p.strip() for p in val.split(",")]
            if len(parts) >= 3:
                try:
                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                    a = int(parts[3]) if len(parts) > 3 else 255
                    return (r, g, b, a)
                except ValueError: return default # Handle non-integer parts
        # Basic named colors (add more if needed)
        named = {"red": (255,0,0,255), "green": (0,255,0,255), "blue": (0,0,255,255),
                 "white": (255,255,255,255), "black": (0,0,0,255), "transparent": (0,0,0,0),
                 "yellow": (255,255,0,255), "cyan": (0,255,255,255), "magenta": (255,0,255,255)}
        if val.lower() in named:
            return named[val.lower()]
    # If val is an int/float, treat as grayscale? (Could add this if needed)
    return default

def _norm01(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))

def _ensure_image(img: Optional[Image.Image], width: int, height: int) -> Image.Image:
    # Ensures an RGBA image exists, creating a transparent one if needed.
    if img is None:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if img.mode != "RGBA":
        return img.convert("RGBA")
    return img

# =============== Blocks ===============

@dataclass
class SolidColor(BaseBlock):
    """Generates a solid color image."""
    def process(self, img, width, height, *, params):
        color = _parse_color(params.get("color", "black"), default=(0, 0, 0, 255))
        return Image.new("RGBA", (width, height), color)

@dataclass
class DrawCircle(BaseBlock):
    """Draws a circle on the image."""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)

        center_x = float(params.get("cx", 0.5)) * width
        center_y = float(params.get("cy", 0.5)) * height
        radius = float(params.get("radius", 0.2)) * min(width, height)
        color = _parse_color(params.get("color", "white"), default=(255, 255, 255, 255))
        outline_color = _parse_color(params.get("outline", None), default=None) # Pass None default
        outline_width = int(params.get("outline_width", 1))

        draw = ImageDraw.Draw(img)
        x0 = center_x - radius
        y0 = center_y - radius
        x1 = center_x + radius
        y1 = center_y + radius
        bbox = [x0, y0, x1, y1]

        # Only draw fill/outline if color is not None
        _fill = color if color is not None else None
        _outline = outline_color if outline_color is not None else None
        draw.ellipse(bbox, fill=_fill, outline=_outline, width=outline_width)
        return img

@dataclass
class AddNoise(BaseBlock):
    """Adds random noise to the image."""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)

        amount = float(params.get("amount", 0.1)) # 0..1 scale
        mono = bool(params.get("mono", False)) # Grayscale noise if true

        # Convert to numpy array for faster processing
        data = np.array(img).astype(np.float32) / 255.0

        if mono:
            noise = np.random.rand(height, width, 1).astype(np.float32) * 2.0 - 1.0
            noise = np.repeat(noise, 3, axis=2) # Repeat for R, G, B
        else:
            noise = np.random.rand(height, width, 3).astype(np.float32) * 2.0 - 1.0

        # Add noise only to RGB channels, leave Alpha untouched
        data[:, :, :3] = data[:, :, :3] + noise * amount
        data = np.clip(data, 0.0, 1.0)

        # Convert back to PIL Image
        return Image.fromarray((data * 255.0).astype(np.uint8), "RGBA")


@dataclass
class LinearGradient(BaseBlock):
    """
    Create a linear gradient. Params:
      start: color (any format accepted by _parse_color)
      end:   color
      angle: degrees (0 = L→R, 90 = T→B)
    """
    def process(self, img, width, height, *, params):
        c0 = _parse_color(params.get("start", "#000000"), (0,0,0,255))
        c1 = _parse_color(params.get("end",   "#FFFFFF"), (255,255,255,255))
        ang = float(params.get("angle", 0.0)) * np.pi / 180.0
        # Unit vector for gradient direction
        dx, dy = np.cos(ang), np.sin(ang)
        # Build coordinate grid centered at (0,0) spanning -0.5 to 0.5
        xs = np.linspace(-0.5, 0.5, width, endpoint=True)
        ys = np.linspace(-0.5, 0.5, height, endpoint=True)
        X, Y = np.meshgrid(xs, ys)
        # Project coordinates onto gradient vector
        t = (X * dx + Y * dy)
        # Normalize t to 0..1 across the projection range
        min_t, max_t = t.min(), t.max()
        t = (t - min_t) / (max_t - min_t + 1e-12)

        c0 = np.array(c0, dtype=np.float32)
        c1 = np.array(c1, dtype=np.float32)
        # Interpolate colors based on t
        out = (c0[None, None, :] + (c1 - c0)[None, None, :] * t[..., None]).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGBA")


@dataclass
class RadialGradient(BaseBlock):
    """
    Radial gradient. Params:
      cx, cy: center (0..1 relative to image size)
      radius: relative size (0..1 where 1 touches corners from center)
      inner, outer: colors
    """
    def process(self, img, width, height, *, params):
        cx = float(params.get("cx", 0.5))
        cy = float(params.get("cy", 0.5))
        r  = float(params.get("radius", 0.707)) # Default 0.707 touches edges from center
        c0 = _parse_color(params.get("inner", "#FFFFFF"), (255,255,255,255))
        c1 = _parse_color(params.get("outer", "#000000"), (0,0,0,255))

        # Coordinates relative to center, scaled by max dimension for aspect correction
        max_dim = max(width, height)
        xs = ((np.arange(width) + 0.5) - cx * width) / max_dim
        ys = ((np.arange(height) + 0.5) - cy * height) / max_dim
        X, Y = np.meshgrid(xs, ys)

        # Distance from center, scaled by radius
        d = np.sqrt(X ** 2 + Y ** 2) / max(r, 1e-6)
        t = np.clip(d, 0.0, 1.0) # Clamp distance factor to [0, 1]

        c0 = np.array(c0, dtype=np.float32)
        c1 = np.array(c1, dtype=np.float32)
        # Interpolate colors based on t
        out = (c0[None, None, :] + (c1 - c0)[None, None, :] * t[..., None]).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGBA")


@dataclass
class Checkerboard(BaseBlock):
    """
    Checkerboard source. Params:
      size: int (size of squares, pixels)
      color_a, color_b: colors
    """
    def process(self, img, width, height, *, params):
        size = max(1, int(params.get("size", 32)))
        ca = _parse_color(params.get("color_a", "#222222"), (34, 34, 34, 255))
        cb = _parse_color(params.get("color_b", "#dddddd"), (221, 221, 221, 255))
        # Create indices grid
        yy, xx = np.indices((height, width))
        # Determine check pattern based on integer division by size
        mask = ((xx // size) + (yy // size)) % 2
        # Use numpy.where to select colors based on mask
        A = np.array(ca, dtype=np.uint8)
        B = np.array(cb, dtype=np.uint8)
        out = np.where(mask[..., None] == 0, A[None, None, :], B[None, None, :]).astype(np.uint8)
        return Image.fromarray(out, "RGBA")


@dataclass
class DrawRect(BaseBlock):
    """
    Draw a rectangle (optionally rounded). Params (normalized unless *_px):
      x, y, w, h: 0..1, or use x_px,y_px,w_px,h_px for absolute pixels
      fill, outline, outline_width, radius
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        # Prioritize pixel values if provided
        x = params.get("x_px", None)
        y = params.get("y_px", None)
        w = params.get("w_px", None)
        h = params.get("h_px", None)
        # Fallback to normalized values if pixel values are missing
        if x is None: x = float(params.get("x", 0.1)) * width
        if y is None: y = float(params.get("y", 0.1)) * height
        if w is None: w = float(params.get("w", 0.3)) * width
        if h is None: h = float(params.get("h", 0.3)) * height
        # Ensure coordinates are floats for drawing
        x, y, w, h = float(x), float(y), float(w), float(h)

        fill    = _parse_color(params.get("fill", None), default=None) # Default to no fill
        outline = _parse_color(params.get("outline", "white"), default=(255,255,255,255))
        ow      = int(params.get("outline_width", 2))
        rad     = int(params.get("radius", 0))

        draw = ImageDraw.Draw(img)
        bbox = [x, y, x + w, y + h]
        # Only draw fill/outline if color is not None
        _fill = fill if fill is not None else None
        _outline = outline if outline is not None else None

        if rad > 0 and hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle(bbox, radius=rad, fill=_fill, outline=_outline, width=ow)
        else:
            draw.rectangle(bbox, fill=_fill, outline=_outline, width=ow)
        return img


@dataclass
class DrawLine(BaseBlock):
    """
    Draw a line. Params:
      x0,y0,x1,y1: normalized 0..1 (or *_px variants)
      width, color
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        # Helper to get pixel or normalized coordinates
        def gv(name, axis_size, default_norm):
            px_val = params.get(name + "_px", None)
            if px_val is not None: return float(px_val)
            return float(params.get(name, default_norm)) * axis_size

        x0 = gv("x0", width, 0.1)
        y0 = gv("y0", height, 0.1)
        x1 = gv("x1", width, 0.9)
        y1 = gv("y1", height, 0.9)
        lw = int(params.get("width", 3))
        color = _parse_color(params.get("color", "white"), default=(255,255,255,255))

        if color is not None: # Only draw if color is valid
            ImageDraw.Draw(img).line([(x0, y0), (x1, y1)], fill=color, width=lw)
        return img


@dataclass
class DrawText(BaseBlock):
    """
    Draw text. Params:
      text: string
      x,y: normalized anchor position (0..1) or x_px,y_px
      size: px
      fill: color
      anchor: PIL anchor (e.g., 'mm','lt','rb' etc.) default 'mm' (middle-middle)
      font: optional path to .ttf/.otf
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        text   = str(params.get("text", "hello"))
        size   = int(params.get("size", 24))
        fill   = _parse_color(params.get("fill", "white"), default=(255,255,255,255))
        anchor = str(params.get("anchor", "mm")) # Middle Middle default
        # Get coordinates
        x = params.get("x_px", None); y = params.get("y_px", None)
        if x is None: x = float(params.get("x", 0.5)) * width
        if y is None: y = float(params.get("y", 0.5)) * height
        x, y = float(x), float(y)

        # Load font
        try:
            font_path = params.get("font", None)
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, size=size)
            else:
                font = ImageFont.load_default() # Fallback
                if font_path: print(f"Warning: Font not found at '{font_path}', using default.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error loading font '{params.get('font', 'default')}': {e}", file=sys.stderr)
            font = ImageFont.load_default()

        if fill is not None: # Only draw if color is valid
            draw = ImageDraw.Draw(img)
            # Use anchor if available, otherwise fallback to manual calculation
            try:
                draw.text((x, y), text, font=font, fill=fill, anchor=anchor)
            except TypeError: # Older PIL might not support anchor
                try:
                    # Calculate bounding box (may vary slightly across systems/fonts)
                    bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    # Adjust position based on common anchors
                    if anchor.startswith('m'): y -= text_height / 2
                    elif anchor.startswith('l'): pass # Top/Left is default (0,0) relative
                    elif anchor.startswith('r'): x -= text_width
                    elif anchor.startswith('b'): y -= text_height
                    if len(anchor) > 1:
                       if anchor[1] == 'm': x -= text_width / 2
                       elif anchor[1] == 't': pass
                       elif anchor[1] == 'l': pass
                       elif anchor[1] == 'r': x -= text_width
                       elif anchor[1] == 'b': y -= text_height

                    draw.text((x, y), text, font=font, fill=fill)
                except Exception as e_fallback:
                     print(f"Warning: Could not draw text with fallback anchor: {e_fallback}", file=sys.stderr)
        return img


@dataclass
class GaussianBlur(BaseBlock):
    """Gaussian blur. Params: radius (pixels)"""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        r = float(params.get("radius", 2.0))
        if r > 0: # Avoid error if radius is zero or negative
            return img.filter(ImageFilter.GaussianBlur(r))
        return img


@dataclass
class BoxBlur(BaseBlock):
    """Box blur. Params: radius (pixels, rounds down)"""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        r = int(params.get("radius", 2.0)) # Box blur radius is integer
        if r > 0:
            return img.filter(ImageFilter.BoxBlur(r))
        return img


@dataclass
class BrightnessContrast(BaseBlock):
    """Adjust brightness/contrast. Params: brightness, contrast (1.0 = no change)"""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        b = float(params.get("brightness", 1.0))
        c = float(params.get("contrast", 1.0))
        # Apply brightness
        if abs(b - 1.0) > 1e-6:
             enhancer = ImageEnhance.Brightness(img)
             img = enhancer.enhance(b)
        # Apply contrast
        if abs(c - 1.0) > 1e-6:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(c)
        return img


@dataclass
class Saturation(BaseBlock):
    """Adjust color saturation. Params: amount (0=grayscale, 1=no change, >1=boost)"""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        s = float(params.get("amount", 1.0))
        if abs(s - 1.0) > 1e-6:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(s)
        return img


@dataclass
class Invert(BaseBlock):
    """Invert RGB channels, keep alpha."""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        # Separate RGB and Alpha
        if img.mode == 'RGB':
            rgb = img
            a = Image.new('L', img.size, 255) # Create opaque alpha
        else: # Assumes RGBA or similar with alpha
            rgb = img.convert("RGB")
            try:
                a = img.getchannel("A")
            except ValueError: # Handle modes without explicit alpha like P
                a = Image.new('L', img.size, 255)

        # Invert RGB
        rgb_inverted = ImageOps.invert(rgb)

        # Merge back with original alpha
        out = Image.merge("RGBA", (*rgb_inverted.split(), a))
        return out


@dataclass
class Gamma(BaseBlock):
    """Apply gamma correction to RGB. Params: gamma (e.g., 0.45 for linear->sRGB approx, 2.2 for sRGB->linear)"""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        g = max(1e-6, float(params.get("gamma", 1.0)))
        if abs(g - 1.0) < 1e-6: return img # No change if gamma is 1

        # Use lookup table for efficiency if PIL supports it, otherwise numpy
        inv_gamma = 1.0 / g
        try:
             # Create a gamma lookup table (LUT) for 8-bit channels
             lut = [pow(i / 255.0, inv_gamma) * 255.0 for i in range(256)]
             lut = [int(round(v)) for v in lut] # Ensure integer values

             if img.mode == 'RGB':
                 # Apply LUT to each RGB channel
                 channels = img.split()
                 corrected_channels = [ch.point(lut) for ch in channels]
                 return Image.merge('RGB', corrected_channels)
             elif img.mode == 'RGBA':
                 # Apply LUT only to RGB, keep Alpha
                 channels = img.split()
                 corrected_rgb = [ch.point(lut) for ch in channels[:3]]
                 return Image.merge('RGBA', (*corrected_rgb, channels[3]))
             else: # Fallback for other modes if necessary
                 raise NotImplementedError("LUT method only implemented for RGB/RGBA")

        except Exception: # Fallback to numpy if LUT fails or mode not supported
            arr = np.asarray(img).astype(np.float32) / 255.0
            # Apply gamma only to color channels (assuming first 3 are color)
            num_color_channels = min(3, arr.shape[2])
            arr[..., :num_color_channels] = np.power(arr[..., :num_color_channels], inv_gamma)
            return Image.fromarray((arr * 255.0).clip(0,255).astype(np.uint8), img.mode)


@dataclass
class Vignette(BaseBlock):
    """Radial darkening. Params: strength (0..1), smooth (feather 0..1), cx(0.5), cy(0.5)"""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        strength = _norm01(float(params.get("strength", 0.6)))
        if strength < 1e-6: return img # No effect if strength is zero

        smooth   = _norm01(float(params.get("smooth", 0.8)), 0.01, 1.0) # Ensure smooth > 0
        cx = float(params.get("cx", 0.5))
        cy = float(params.get("cy", 0.5))

        # Coordinates relative to center, normalized 0..1 axis lengths
        xs = ((np.arange(width) + 0.5) / width) - cx
        ys = ((np.arange(height) + 0.5) / height) - cy
        X, Y = np.meshgrid(xs, ys)

        # Distance from center, scaled to roughly 1.0 at corners
        d = np.sqrt(X**2 + Y**2) / np.sqrt(0.5**2 + 0.5**2)

        # Create mask: 1.0 at center, falling off based on distance and smoothness
        # Power function creates the falloff curve, controlled by 'smooth'
        # Higher smooth -> faster falloff (sharper vignette edge)
        # Lower smooth -> slower falloff (softer vignette edge)
        mask = 1.0 - (d ** (1.0 / max(smooth, 1e-6))) # Use 1/smooth for intuitive control
        mask = np.clip(mask, 0.0, 1.0)

        # Apply vignette: Lerp between original color and darkened color based on mask and strength
        # Formula: final = original * (mask * (1 - strength) + strength)
        # Simplifies to: final = original * (1 - strength * (1 - mask))
        vignette_factor = 1.0 - strength * (1.0 - mask)

        arr = np.asarray(img).astype(np.float32)
        # Apply only to RGB channels
        arr[..., :3] *= vignette_factor[..., None]

        return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")


@dataclass
class Rotate(BaseBlock):
    """Rotate image. Params: angle (degrees), expand (bool)"""
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        ang = float(params.get("angle", 15.0))
        expand = bool(params.get("expand", False))

        # Rotate using bilinear resampling, fill transparent if expanding
        rotated = img.rotate(ang, resample=Image.Resampling.BILINEAR, expand=expand, fillcolor=(0,0,0,0))

        # If size changed due to expand=True, resize back to original dimensions
        if rotated.size != (width, height):
            # Crop/pad back to original size centered
            old_w, old_h = img.size
            new_w, new_h = rotated.size

            # Create a new transparent image of the target size
            final_img = Image.new("RGBA", (width, height), (0,0,0,0))

            # Calculate pasting position to center the rotated image
            paste_x = (width - new_w) // 2
            paste_y = (height - new_h) // 2

            # Paste the rotated image onto the transparent background
            final_img.paste(rotated, (paste_x, paste_y), rotated) # Use alpha mask from rotated
            return final_img
        else:
            return rotated # Return directly if size didn't change


@dataclass
class BlendColor(BaseBlock):
    """
    Alpha blend with a flat color over the current image.
    Params:
      color: color
      alpha: 0..1 (0=original, 1=overlay color)
    """
    def process(self, img, width, height, *, params):
        img = _ensure_image(img, width, height)
        color = _parse_color(params.get("color", "black"), default=(0,0,0,255))
        alpha = _norm01(float(params.get("alpha", 0.25)))

        if alpha < 1e-6: return img # No effect
        if alpha > 0.9999: return Image.new("RGBA", (width, height), color) # Fully overlay

        # Create the solid color overlay image
        overlay = Image.new("RGBA", (width, height), color)

        # Blend using PIL's blend function
        return Image.blend(img, overlay, alpha)


# =============== Engine ===============

class GraphicsEngine:
    def __init__(self, *, width: int = 640, height: int = 480) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))

    def render(
        self,
        *,
        pipeline: str,
        extras: Dict[str, Dict[str, Any]] | None = None,
    ) -> Image.Image:
        width = self.width
        height = self.height
        blocks = [b.strip().lower() for b in (pipeline or "").split("|") if b.strip()]
        if not blocks:
            blocks = ["solidcolor"] # Default pipeline if empty

        img: Optional[Image.Image] = None
        for name in blocks:
            params = (extras or {}).get(name, {})
            block = REGISTRY.create(name)
            img = block.process(img, width, height, params=params)
            if img is None:
                raise RuntimeError(f"Block '{name}' returned None")
            # Ensure consistent mode and size AFTER processing
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            # Only resize if block changed size unexpectedly (like rotate without expand=False)
            if img.size != (width, height):
                 print(f"Warning: Block '{name}' changed image size to {img.size}, resizing back to {width}x{height}.", file=sys.stderr)
                 img = img.resize((width, height), Image.Resampling.LANCZOS) # Use Lanczos for quality

        # Fallback if the entire pipeline somehow yielded None
        if img is None:
             img = Image.new("RGBA", (width, height), (255, 0, 255, 255)) # Magenta error

        return img

# =============== Registration Calls ===============
# Originals
REGISTRY.register("solidcolor", SolidColor)
REGISTRY.register("drawcircle", DrawCircle)
REGISTRY.register("addnoise", AddNoise)

# New additions
REGISTRY.register("lineargradient", LinearGradient)
REGISTRY.register("radialgradient", RadialGradient)
REGISTRY.register("checkerboard", Checkerboard)
REGISTRY.register("drawrect", DrawRect)
REGISTRY.register("drawline", DrawLine)
REGISTRY.register("drawtext", DrawText)
REGISTRY.register("gaussianblur", GaussianBlur)
REGISTRY.register("boxblur", BoxBlur)
REGISTRY.register("brightnesscontrast", BrightnessContrast)
REGISTRY.register("saturation", Saturation)
REGISTRY.register("invert", Invert)
REGISTRY.register("gamma", Gamma)
REGISTRY.register("vignette", Vignette)
REGISTRY.register("rotate", Rotate)
REGISTRY.register("blendcolor", BlendColor)