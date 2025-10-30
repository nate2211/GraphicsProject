#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import subprocess # For ffmpeg
import time # For animation timing
from typing import Dict, Any, List, Optional

import numpy as np
from PIL import Image
import shutil
import subprocess, sys, threading, time
import designs
import detail
import warp
import camera
import animations
# Make sure graphics.py and animations.py are importable
try:
    from graphics import GraphicsEngine as GraphicsEngineBase, parse_extras, REGISTRY as GFX_REGISTRY, BaseBlock as GfxBaseBlock
    # Import animation registry and context
    from animations import ANIMATION_REGISTRY, AnimationContext, BaseAnimationBlock
except ImportError as e:
    print(f"Error: Could not import from graphics.py or animations.py.")
    print(f"Details: {e}")
    sys.exit(1)

# ----- Modified Graphics Engine -----
class GraphicsEngine(GraphicsEngineBase):
    """Extended engine to handle animation context and block types."""

    def render_frame(
        self,
        *,
        pipeline: List[str], # Use pre-parsed list
        extras: List[Dict[str, Any]], # Use pre-split extras
        ctx: AnimationContext,
    ) -> Image.Image:
        """Renders a single frame for animation."""
        img: Optional[Image.Image] = None

        for i, name in enumerate(pipeline):
            params = extras[i]
            # Check if it's an animation block first
            if name in ANIMATION_REGISTRY.names():
                block = ANIMATION_REGISTRY.create(name)
                if isinstance(block, BaseAnimationBlock):
                    img = block.process_frame(img, ctx, params, self) # Pass engine ref
                else:
                     print(f"Warning: Block '{name}' in animation registry but not a BaseAnimationBlock.", file=sys.stderr)
                     # Attempt to process as graphics block if possible (might fail)
                     try:
                         gfx_block = GFX_REGISTRY.create(name)
                         img = gfx_block.process(img, ctx.width, ctx.height, params=params)
                     except Exception as e_inner:
                          print(f"Error trying gfx fallback for '{name}': {e_inner}", file=sys.stderr)
                          # Keep previous image state if error occurs
            # Otherwise, assume it's a graphics block
            elif name in GFX_REGISTRY.names():
                block = GFX_REGISTRY.create(name)
                img = block.process(img, ctx.width, ctx.height, params=params)
            else:
                # Should not happen if validation is done beforehand
                raise KeyError(f"Unknown block type '{name}' in pipeline.")

            if img is None:
                raise RuntimeError(f"Block '{name}' returned None during frame {ctx.frame}")

            # Ensure consistent mode and size after each block
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            if img.size != (ctx.width, ctx.height):
                 print(f"Warning: Block '{name}' changed size on frame {ctx.frame}, resizing.", file=sys.stderr)
                 img = img.resize((ctx.width, ctx.height), Image.Resampling.LANCZOS)

        # Fallback if pipeline yields None (e.g., empty pipeline)
        if img is None:
            img = Image.new("RGBA", (ctx.width, ctx.height), (255, 0, 255, 255)) # Magenta error

        return img

# ----------------------------- FFMPEG Video Output -----------------------------

def _drain_stream(stream):
    try:
        # Read & discard to avoid PIPE backpressure
        for _ in iter(lambda: stream.read(4096), b""):
            pass
    except Exception:
        pass

def start_ffmpeg_process(outfile: str, width: int, height: int, fps: float, ffmpeg_bin: str | None = None) -> subprocess.Popen:
    bin_path = ffmpeg_bin or os.environ.get("FFMPEG_BIN") or "ffmpeg"
    command = [
        bin_path,
        "-y",
        "-loglevel", "error",     # reduce stderr noise
        "-hide_banner",
        "-nostats",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgba",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        outfile,
    ]
    print("Starting ffmpeg:", " ".join(command), "->", outfile, file=sys.stderr)
    try:
        # On Windows, avoid creating a console window (optional)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,           # keep PIPE but drain it in a thread
            creationflags=creationflags
        )
        # Drain stderr to prevent deadlock
        threading.Thread(target=_drain_stream, args=(proc.stderr,), daemon=True).start()
        return proc
    except FileNotFoundError:
        raise RuntimeError(f"ffmpeg not found: {bin_path}")
    except Exception as e:
        raise RuntimeError(f"Error starting ffmpeg: {e}")

def feed_ffmpeg(proc: subprocess.Popen, frame):
    try:
        if frame.mode != "RGBA":
            frame = frame.convert("RGBA")
        proc.stdin.write(frame.tobytes())  # type: ignore
    except (BrokenPipeError, OSError) as e:
        raise RuntimeError(f"Error writing to ffmpeg stdin: {e}")

def finish_ffmpeg_process(proc: subprocess.Popen, *, grace_sec: float = 15.0) -> bool:
    # Close stdin so ffmpeg can finalize
    try:
        if proc.stdin:
            try:
                proc.stdin.flush()
            except Exception:
                pass
            proc.stdin.close()
    except Exception:
        pass

    # Poll with a deadline; then terminate/kill if needed
    deadline = time.time() + grace_sec
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            return rc == 0
        time.sleep(0.1)

    # Try graceful terminate
    try:
        proc.terminate()
    except Exception:
        pass

    deadline = time.time() + 5.0
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            return rc == 0
        time.sleep(0.1)

    # Force kill as last resort
    try:
        proc.kill()
    except Exception:
        pass
    return False

def feed_ffmpeg(proc: subprocess.Popen, frame: Image.Image):
    """Sends a single RGBA frame to ffmpeg's stdin."""
    try:
        # Ensure frame is RGBA before getting bytes
        if frame.mode != "RGBA":
            frame = frame.convert("RGBA")
        proc.stdin.write(frame.tobytes()) # type: ignore
    except (BrokenPipeError, OSError) as e:
        print(f"\nError writing to ffmpeg stdin: {e}", file=sys.stderr)
        print("FFmpeg might have crashed. Check stderr:", file=sys.stderr)
        stderr = proc.communicate()[1] # Get stderr output
        if stderr:
             print(stderr.decode(errors='ignore'), file=sys.stderr)
        raise # Re-raise the exception to stop the process

def finish_ffmpeg_process(proc: subprocess.Popen) -> bool:
    """Closes stdin and waits for ffmpeg to finish."""
    if proc.stdin:
        proc.stdin.close()
    retcode = proc.wait(timeout=30) # Wait up to 30s
    if retcode != 0:
        print(f"ffmpeg process exited with error code {retcode}.", file=sys.stderr)
        # stderr might have already been printed in feed_ffmpeg on error
        stderr = proc.stderr.read() if proc.stderr else b""
        if stderr:
            print("FFmpeg stderr:", file=sys.stderr)
            print(stderr.decode(errors='ignore'), file=sys.stderr)
        return False
    print("ffmpeg finished successfully.", file=sys.stderr)
    return True

# ----------------------------- CLI -----------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Graphics pipeline engine using PIL (supports image and video output)."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- 'run' command ---
    r = sub.add_parser("run", help="Render an image or video using a pipeline")
    # Output options (choose one)
    out_group = r.add_mutually_exclusive_group(required=True)
    out_group.add_argument("--out", help="Output image path (e.g., image.png)")
    out_group.add_argument("--out-video", help="Output video path (e.g., animation.mp4)")

    r.add_argument("--width", type=int, default=640, help="Width in pixels")
    r.add_argument("--height", type=int, default=480, help="Height in pixels")
    r.add_argument(
        "--pipeline",
        required=True,
        help="Pipeline string, e.g., 'solidcolor|drawcircle|animateparam'"
    )
    r.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra parameters ('block.param=value')"
    )
    # Video specific options
    r.add_argument("--duration", type=float, default=5.0, help="Animation duration in seconds (for video output)")
    r.add_argument("--fps", type=float, default=30.0, help="Frames per second (for video output)")
    r.add_argument(
        "--ffmpeg-bin",
        default=os.environ.get("FFMPEG_BIN", "ffmpeg"),
        help="Path or name of ffmpeg executable (defaults to $FFMPEG_BIN or 'ffmpeg')",
    )
    r.set_defaults(func=cmd_run)

    # --- 'list' command ---
    l = sub.add_parser("list", help="List available graphics and animation blocks")
    l.set_defaults(func=cmd_list)

    return p

# ---------------- Command Functions ----------------

def cmd_run(args: argparse.Namespace) -> int:
    """Handles the 'run' command for both image and video."""
    try:
        extras_dict: Dict[str, Dict[str, Any]] = parse_extras(args.extra)
        pipeline_names = [b.strip().lower() for b in (args.pipeline or "").split("|") if b.strip()]

        # Validate pipeline blocks exist
        all_known_blocks = set(GFX_REGISTRY.names()) | set(ANIMATION_REGISTRY.names())
        unknown = [name for name in pipeline_names if name not in all_known_blocks]
        if unknown:
            print(f"Error: Unknown block(s) in pipeline: {', '.join(unknown)}", file=sys.stderr)
            print(f"Available blocks are: {', '.join(sorted(all_known_blocks))}", file=sys.stderr)
            return 1

        # Prepare extras per stage (needed for render_frame)
        extras_list: List[Dict[str, Any]] = []
        for idx, name in enumerate(pipeline_names):
            stage_params: Dict[str, Any] = {}
            # Name-based (applies to all stages of same name)
            stage_params.update(extras_dict.get(name, {}))
            # 0-based stage index: "0.foo=bar"
            stage_params.update(extras_dict.get(str(idx), {}))
            # 1-based stage index: "1.foo=bar"
            stage_params.update(extras_dict.get(str(idx + 1), {}))
            extras_list.append(stage_params)

        engine = GraphicsEngine(width=args.width, height=args.height)

        # --- Video Output ---
        if args.out_video:
            fps = max(1.0, args.fps)
            duration = max(1.0 / fps, args.duration)
            total_frames = int(round(duration * fps))

            print(f"Rendering video: {args.out_video}")
            print(f"Dimensions: {args.width}x{args.height}, Duration: {duration:.2f}s, FPS: {fps:.1f}, Frames: {total_frames}")
            print(f"Pipeline: {args.pipeline}")
            # print(f"Extras: {extras_dict}") # Can be verbose

            ffmpeg_proc = start_ffmpeg_process(
                ffmpeg_bin=args.ffmpeg_bin,  # NEW
                outfile=args.out_video,
                width=args.width,
                height=args.height,
                fps=fps,
            )
            start_time = time.time()

            try:
                for i in range(total_frames):
                    current_time = i / fps
                    ctx = AnimationContext(
                        frame=i,
                        total_frames=total_frames,
                        time=current_time,
                        duration=duration,
                        fps=fps,
                        width=args.width,
                        height=args.height
                    )
                    # Use render_frame
                    img = engine.render_frame(pipeline=pipeline_names, extras=extras_list, ctx=ctx)
                    feed_ffmpeg(ffmpeg_proc, img)

                    # Progress indicator
                    elapsed = time.time() - start_time
                    percent = (i + 1) / total_frames * 100
                    eta = (elapsed / (i + 1)) * (total_frames - (i + 1)) if i > 0 else 0
                    print(f"\rFrame {i+1}/{total_frames} ({percent:.1f}%) | Time: {current_time:.2f}s | Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s ", end="")

            except Exception as e_render:
                print(f"\nError during frame rendering or piping: {e_render}", file=sys.stderr)
                # Attempt to cleanup ffmpeg
                if ffmpeg_proc and ffmpeg_proc.poll() is None:
                    try:
                        ffmpeg_proc.stdin.close() # type: ignore
                    except: pass
                    try:
                        ffmpeg_proc.kill()
                    except: pass
                return 1
            finally:
                print() # Newline after progress

            if not finish_ffmpeg_process(ffmpeg_proc):
                print(f"Video saving failed. Check ffmpeg errors above.", file=sys.stderr)
                # Optionally remove potentially corrupt output file
                # try: os.remove(args.out_video)
                # except: pass
                return 1
            else:
                print(f"Successfully saved video to {args.out_video}")

        # --- Image Output ---
        elif args.out:
            print(f"Rendering image: {args.out}")
            print(f"Dimensions: {args.width}x{args.height}")
            print(f"Pipeline: {args.pipeline}")
            # print(f"Extras: {extras_dict}")

            # For single image, render frame 0 at time 0
            ctx = AnimationContext(
                frame=0, total_frames=1, time=0.0, duration=1.0, fps=1.0, # Dummy values for single frame
                width=args.width, height=args.height
            )
            # Use render_frame for consistency
            img = engine.render_frame(pipeline=pipeline_names, extras=extras_list, ctx=ctx)

            os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
            img.save(args.out)
            print(f"Saved image to {args.out}")

        return 0

    except Exception as e:
        print(f"[Error] Failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc() # Show full traceback for debugging
        return 1

def cmd_list(args: argparse.Namespace) -> int:
    """Handles the 'list' command."""
    print("Available Graphics Blocks:")
    gfx_available = GFX_REGISTRY.names()
    if not gfx_available: print("  (None)")
    else:
        for name in gfx_available: print(f"  - {name}")

    print("\nAvailable Animation Blocks:")
    anim_available = ANIMATION_REGISTRY.names()
    if not anim_available: print("  (None)")
    else:
        for name in anim_available: print(f"  - {name}")

    return 0

# ---------------- Main Execution ----------------

def main(argv: Optional[List[str]] = None) -> int:
    """Parses arguments and calls the appropriate command function."""
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()

    if not argv:  # Show help if no arguments are given
        parser.print_help(sys.stderr)
        return 1

    try:
        args = parser.parse_args(argv)
        return args.func(args)
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}", file=sys.stderr)
        # import traceback # Uncomment for debugging top-level errors
        # traceback.print_exc()
        return 2

if __name__ == "__main__":
    # Ensure necessary modules are imported to register blocks
    import graphics # Ensures graphics blocks are registered
    import animations # Ensures animation blocks are registered
    try:
        import threed # Try importing 3d if it exists
    except ImportError:
        pass # Optional module

    raise SystemExit(main())