###gui.py####
from __future__ import annotations

import sys
import time
import dataclasses
import traceback
import json
import os # Added for path checks in font loading

from typing import Dict, Any, List, Type, Optional

from PIL import Image, ImageDraw
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QCoreApplication, QUrl
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QPushButton, QComboBox, QLabel, QLineEdit,
    QColorDialog, QSplitter, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QSlider, QFileDialog,
    QProgressDialog, QMessageBox,
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
# Import all block modules BEFORE GUI boots (they register into registries)
# These imports ensure your blocks are registered with GFX_REGISTRY and ANIMATION_REGISTRY
from animations import ANIMATION_REGISTRY, AnimationContext
import animations
import graphics
import designs
import detail
import camera
import threed
import warp
import visualizer
import images
import videos
# Registries / engine
from graphics import REGISTRY as GFX_REGISTRY


# Import the EXTENDED GraphicsEngine and ffmpeg helpers from main.py
# This ensures we use the animation-aware engine and video export utilities.
from main import GraphicsEngine, start_ffmpeg_process, feed_ffmpeg, finish_ffmpeg_process


# --------------------------- registry schema helpers ---------------------------

def _has_registry_schema(registry) -> bool:
    return hasattr(registry, "params_schema") and callable(getattr(registry, "params_schema"))


def _get_schema_for_block(block_name: str) -> Dict[str, Any]:
    """
    Pull schema from graphics.py-style @params() if present.
    For animation blocks, we try the same API if it exists.
    """
    name = (block_name or "").strip().lower()

    if _has_registry_schema(GFX_REGISTRY) and name in getattr(GFX_REGISTRY, "_by_name", {}):
        try:
            return GFX_REGISTRY.params_schema(name) or {}
        except Exception:
            return {}

    if _has_registry_schema(ANIMATION_REGISTRY) and name in getattr(ANIMATION_REGISTRY, "_by_name", {}):
        try:
            return ANIMATION_REGISTRY.params_schema(name) or {}
        except Exception:
            return {}

    return {}


def _get_help_for_block(block_name: str) -> str:
    name = (block_name or "").strip().lower()

    if hasattr(GFX_REGISTRY, "help") and name in getattr(GFX_REGISTRY, "_by_name", {}):
        try:
            return GFX_REGISTRY.help(name) or ""
        except Exception:
            return ""

    if hasattr(ANIMATION_REGISTRY, "help") and name in getattr(ANIMATION_REGISTRY, "_by_name", {}):
        try:
            return ANIMATION_REGISTRY.help(name) or ""
        except Exception:
            return ""

    return ""

def _sanitize_media_path(p: str | None) -> str | None:
    if not p:
        return None
    s = str(p).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()

    # Support file:// URIs (same behavior as visualizer.py)
    if s.lower().startswith("file:"):
        from urllib.parse import urlparse, unquote
        import os
        u = urlparse(s)
        s = unquote(u.path)
        if os.name == "nt" and len(s) >= 3 and s[0] == "/" and s[2] == ":":
            s = s[1:]
        s = s.replace("/", os.sep)

    return s or None


def _find_visualizer_audio_path(pipeline_names: list[str], extras_list: list[dict]) -> str | None:
    # Your audio-driven blocks:
    audio_blocks = {"audiowaveform", "audiobars", "audiorms"}

    for name, params in zip(pipeline_names, extras_list):
        if (name or "").strip().lower() in audio_blocks:
            p = _sanitize_media_path(params.get("path"))
            if p:
                return p
    return None
# --------------------------- schema coercion + normalization ---------------------------

def _safe_json_dumps(v: Any) -> str:
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return repr(v)


def _normalize_enum_choices(spec: Dict[str, Any]) -> List[str]:
    """
    Supports either:
      - spec["choices"] (preferred)
      - spec["options"] (legacy)
    Allows: list/tuple/set/iterable.
    """
    raw = spec.get("choices", None)
    if raw is None:
        raw = spec.get("options", None)

    if raw is None:
        return []

    # If raw is something iterable but not string/bytes:
    if isinstance(raw, (list, tuple, set)):
        return [str(x) for x in raw]

    # Maybe a generator/other iterable
    try:
        return [str(x) for x in list(raw)]
    except Exception:
        # last resort: single value
        return [str(raw)]


def _coerce_value_from_text(text: str, spec: Dict[str, Any]) -> Any:
    """
    Convert UI input -> typed value based on schema spec.

    Supports:
      - nullable: empty string => None
      - any: tries JSON decode; falls back to raw string
      - enum: returns selected string (or None if nullable + '<None>')
      - int/float/bool/color/path/str
    """
    t = (spec.get("type") or "str").lower()
    nullable = bool(spec.get("nullable", False))
    s = (text or "").strip()

    # Nullable always wins on empty
    if nullable and s == "":
        return None

    if t == "enum":
        if nullable and s.lower() in ("<none>", "none", "(none)"):
            return None
        return s

    if t == "any":
        if s == "":
            return spec.get("default", None)

        # Try JSON decode for dict/list/bool/null/numbers/strings
        try:
            return json.loads(s)
        except Exception:
            # fallback: raw string
            return s

    if t in ("str", "string", "path"):
        return s

    if t == "color":
        if s == "" and spec.get("default") is not None:
            return spec.get("default")
        return s

    if t in ("bool", "boolean"):
        return s.lower() in ("1", "true", "yes", "y", "on", "checked")

    if t in ("int", "integer"):
        try:
            return int(float(s))
        except Exception:
            d = spec.get("default", 0)
            return 0 if d is None else int(d)

    if t in ("float", "number"):
        try:
            return float(s)
        except Exception:
            d = spec.get("default", 0.0)
            return 0.0 if d is None else float(d)

    # fallback
    return s


# --------------------------- Param widget ---------------------------

class ParameterWidget(QWidget):
    changed = pyqtSignal()

    def __init__(self, param_name: str, spec: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self.spec = dict(spec or {})
        self._type = (self.spec.get("type") or "str").lower()

        self.value = self.spec.get("default", "")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        label = QLabel(f"{param_name}:")
        label.setMinimumWidth(140)
        label.setStyleSheet("font-weight: bold; color: #aaa;")
        layout.addWidget(label)

        self.input = self._build_input_widget(self.value)
        layout.addWidget(self.input, 1)

        # Hint + unit (nice for "unit": "s")
        hint = self.spec.get("hint", "") or ""
        unit = self.spec.get("unit", "") or ""
        if unit and hint:
            hint = f"{hint}  [{unit}]"
        elif unit and not hint:
            hint = f"[{unit}]"

        if hint:
            hint_lbl = QLabel(hint)
            hint_lbl.setStyleSheet("color: #777; font-size: 10px;")
            hint_lbl.setWordWrap(True)
            layout.addWidget(hint_lbl)

    def _build_input_widget(self, default: Any):
        t = self._type
        nullable = bool(self.spec.get("nullable", False))

        # --- ANY: freeform JSON-ish input ---
        if t == "any":
            if default is None:
                start = ""
            elif isinstance(default, (dict, list, bool, int, float)):
                start = _safe_json_dumps(default)
            else:
                start = str(default)

            w = QLineEdit(start)
            if nullable:
                w.setPlaceholderText('None / JSON (e.g. {"k":1}, [1,2], true, null)')
            else:
                w.setPlaceholderText("JSON or raw text")
            w.textChanged.connect(self._on_text_change)
            return w

        # Nullable numeric with default None => must allow blank -> None
        if nullable and default is None and t in ("int", "integer", "float", "number"):
            w = QLineEdit("")
            w.setPlaceholderText("None")
            w.textChanged.connect(self._on_text_change)
            return w

        # BOOL
        if t in ("bool", "boolean"):
            w = QCheckBox()
            w.setChecked(bool(default))
            w.stateChanged.connect(self._on_bool_change)
            return w

        # ENUM (supports 'choices' and legacy 'options')
        if t == "enum":
            w = QComboBox()
            w.setEditable(False)

            choices = _normalize_enum_choices(self.spec)

            # Nullable sentinel
            if nullable:
                w.addItem("<None>")

            for c in choices:
                w.addItem(str(c))

            # Set default
            if default is None and nullable:
                w.setCurrentText("<None>")
            elif default is not None:
                # If default isn't in list, still set text if possible
                sdef = str(default)
                if sdef in [w.itemText(i) for i in range(w.count())]:
                    w.setCurrentText(sdef)
                else:
                    # add it to avoid losing default
                    w.addItem(sdef)
                    w.setCurrentText(sdef)

            w.currentTextChanged.connect(self._on_text_change)
            return w

        # COLOR button
        if t == "color":
            w = QPushButton("" if default is None else str(default))
            w.setStyleSheet(f"text-align: left; padding: 6px; background-color: {default if default else '#333'};")
            w.clicked.connect(self._on_color_click)
            return w

        # INT
        if t in ("int", "integer"):
            if default is None:
                default = self.spec.get("min", 0)

            w = QSpinBox()
            w.setMinimum(int(self.spec.get("min", -2_147_483_648)))
            w.setMaximum(int(self.spec.get("max", 2_147_483_647)))
            step = int(self.spec.get("step", 1) or 1)
            w.setSingleStep(max(1, step))

            try:
                w.setValue(int(default))
            except Exception:
                fallback = self.spec.get("default", 0)
                if fallback is None:
                    fallback = self.spec.get("min", 0)
                w.setValue(int(fallback))

            w.valueChanged.connect(self._on_spin_change)
            return w

        # FLOAT
        if t in ("float", "number"):
            if default is None:
                default = self.spec.get("min", 0.0)

            w = QDoubleSpinBox()
            w.setDecimals(6)
            w.setMinimum(float(self.spec.get("min", -1e12)))
            w.setMaximum(float(self.spec.get("max", 1e12)))
            step = float(self.spec.get("step", 0.01) or 0.01)
            w.setSingleStep(step)

            try:
                w.setValue(float(default))
            except Exception:
                fallback = self.spec.get("default", 0.0)
                if fallback is None:
                    fallback = self.spec.get("min", 0.0)
                w.setValue(float(fallback))

            w.valueChanged.connect(self._on_spin_change)
            return w

        # STR/PATH fallback (also good for weird types)
        w = QLineEdit("" if default is None else str(default))
        if nullable:
            w.setPlaceholderText("None")
        w.textChanged.connect(self._on_text_change)
        return w

    def _on_text_change(self, text: str):
        self.value = _coerce_value_from_text(text, self.spec)
        self.changed.emit()

    def _on_spin_change(self, v):
        self.value = v
        self.changed.emit()

    def _on_bool_change(self, state: int):
        self.value = (state == Qt.CheckState.Checked.value)
        self.changed.emit()

    def _on_color_click(self):
        # If you want "none", user can type it for type=str; for type=color, picker is main UX.
        initial_color = QColor(self.input.text() if self.input.text() else "#000000")
        color = QColorDialog.getColor(initial_color)
        if color.isValid():
            hex_color = color.name()
            self.value = hex_color
            self.input.setText(hex_color)
            self.input.setStyleSheet(f"text-align: left; padding: 6px; background-color: {hex_color};")
            self.changed.emit()


# --------------------------- Block stage control ---------------------------

class BlockControl(QGroupBox):
    remove_requested = pyqtSignal(object)
    moved_up = pyqtSignal(object)
    moved_down = pyqtSignal(object) # Added for moving down
    params_changed = pyqtSignal()

    def __init__(self, block_name: str, block_cls: Type, index: int, parent=None):
        super().__init__(parent)
        self.block_name = (block_name or "").strip().lower()
        self.block_cls = block_cls
        self.param_widgets: List[ParameterWidget] = []

        self.setTitle(f"Stage {index}: {self.block_name.upper()}")
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #555; margin-top: 10px; padding-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }"
        )

        layout = QVBoxLayout(self)

        header = QHBoxLayout()
        up_btn = QPushButton("▲")
        down_btn = QPushButton("▼") # Added down button
        rm_btn = QPushButton("✕")
        rm_btn.setFixedSize(24, 24)
        up_btn.setFixedSize(24, 24)
        down_btn.setFixedSize(24, 24)
        rm_btn.setStyleSheet("background-color: #772222; color: white; border-radius: 4px;")
        up_btn.setStyleSheet("background-color: #333; color: white; border-radius: 4px;")
        down_btn.setStyleSheet("background-color: #333; color: white; border-radius: 4px;")


        up_btn.clicked.connect(lambda: self.moved_up.emit(self))
        down_btn.clicked.connect(lambda: self.moved_down.emit(self)) # Connect down button
        rm_btn.clicked.connect(lambda: self.remove_requested.emit(self))

        header.addWidget(up_btn)
        header.addWidget(down_btn) # Add down button to header
        header.addStretch()
        header.addWidget(rm_btn)
        layout.addLayout(header)

        help_txt = _get_help_for_block(self.block_name)
        if help_txt:
            help_lbl = QLabel(help_txt)
            help_lbl.setWordWrap(True)
            help_lbl.setStyleSheet("color: #888; font-weight: normal; font-style: italic;")
            layout.addWidget(help_lbl)

        self.param_layout = QVBoxLayout()
        layout.addLayout(self.param_layout)

        self.generate_ui()

    def clear_params_ui(self):
        while self.param_layout.count():
            item = self.param_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self.param_widgets.clear()

    def generate_ui(self):
        """
        Prefer schema from @params().
        Fallback: dataclass fields (legacy).
        """
        self.clear_params_ui()

        schema = _get_schema_for_block(self.block_name)
        if schema:
            # dict order preserved in modern python
            for pname, spec in schema.items():
                pw = ParameterWidget(pname, spec)
                pw.changed.connect(self.params_changed.emit)
                self.param_layout.addWidget(pw)
                self.param_widgets.append(pw)
            return

        # Legacy fallback: dataclass field introspection
        if dataclasses.is_dataclass(self.block_cls):
            for f in dataclasses.fields(self.block_cls):
                if f.name.startswith("_"):
                    continue
                val = ""
                if f.default is not dataclasses.MISSING:
                    val = f.default
                elif f.default_factory is not dataclasses.MISSING:  # type: ignore
                    try:
                        val = f.default_factory()  # type: ignore
                    except Exception:
                        val = ""
                pw = ParameterWidget(f.name, {"type": "str", "default": val})
                pw.changed.connect(self.params_changed.emit)
                self.param_layout.addWidget(pw)
                self.param_widgets.append(pw)
        else:
            # Minimal fallback: common cx/cy
            for pname, val in (("cx", 0.5), ("cy", 0.5)):
                pw = ParameterWidget(pname, {"type": "float", "default": val, "min": 0.0, "max": 1.0, "step": 0.01})
                pw.changed.connect(self.params_changed.emit)
                self.param_layout.addWidget(pw)
                self.param_widgets.append(pw)

    def get_stage_extras(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for w in self.param_widgets:
            out[w.param_name] = w.value
        return out


# --------------------------- Export Thread ---------------------------

class ExportThread(QThread):
    progress_update = pyqtSignal(int, str) # percent, status_message
    export_finished = pyqtSignal(bool, str) # success, message

    def __init__(self, engine: GraphicsEngine, pipeline: List[str], extras: List[Dict[str, Any]],
                 outfile: str, width: int, height: int, duration: float, fps: float, audio_path=None, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.pipeline = pipeline
        self.extras = extras
        self.outfile = outfile
        self.width = width
        self.height = height
        self.duration = duration
        self.fps = fps
        self._is_cancelled = False
        self.audio_path = audio_path
    def cancel(self):
        self._is_cancelled = True

    def run(self):
        total_frames = int(round(self.duration * self.fps))
        ffmpeg_proc = None
        success = False
        message = ""

        try:
            self.progress_update.emit(0, "Starting FFmpeg process...")
            ffmpeg_proc = start_ffmpeg_process(
                outfile=self.outfile,
                width=self.width,
                height=self.height,
                fps=self.fps,
                duration=self.duration,
                audio_path=self.audio_path,
                ffmpeg_bin="ffmpeg"
            )
            start_time = time.time()

            for i in range(total_frames):
                if self._is_cancelled:
                    message = "Export cancelled by user."
                    break

                current_time = i / self.fps
                ctx = AnimationContext(
                    frame=i,
                    total_frames=total_frames,
                    time=current_time,
                    duration=self.duration,
                    fps=self.fps,
                    width=self.width,
                    height=self.height
                )
                img = self.engine.render_frame(pipeline=self.pipeline, extras=self.extras, ctx=ctx)
                feed_ffmpeg(ffmpeg_proc, img)

                percent = int((i + 1) / total_frames * 100)
                self.progress_update.emit(percent, f"Rendering frame {i+1}/{total_frames}...")
                QCoreApplication.processEvents() # Keep GUI responsive during heavy loop

            else: # Only runs if loop completed without break
                self.progress_update.emit(100, "Finalizing video...")
                if finish_ffmpeg_process(ffmpeg_proc):
                    success = True
                    message = f"Successfully exported video to {self.outfile}"
                else:
                    message = "FFmpeg failed to finalize video. Check console for errors."

        except Exception as e:
            message = f"Export failed: {type(e).__name__}: {e}"
            print(f"Export thread error: {traceback.format_exc()}", file=sys.stderr)
        finally:
            if ffmpeg_proc and ffmpeg_proc.poll() is None:
                # If ffmpeg is still running due to an error/cancellation
                try:
                    ffmpeg_proc.stdin.close()
                    ffmpeg_proc.kill()
                except Exception as e_kill:
                    print(f"Error during ffmpeg cleanup: {e_kill}", file=sys.stderr)
            self.export_finished.emit(success, message)


# --------------------------- Main GUI ---------------------------

class GraphicsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gemini Engine: Generative Lab v2.0")
        self.resize(1400, 900)

        # Animation state variables
        self._current_time: float = 0.0
        self._is_playing: bool = False
        self._animation_duration: float = 5.0 # seconds
        self._animation_fps: float = 30.0    # frames per second

        # Instantiate the extended GraphicsEngine from main.py
        self.engine = GraphicsEngine(width=800, height=600)
        self.blocks: List[BlockControl] = []

        self.init_ui()
        self.audio_output = QAudioOutput()
        self.audio_player = QMediaPlayer()
        self.audio_player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.75)  # 0.0 to 1.0

        self._current_audio_path: str | None = None
        # Timer for live preview updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_tick)
        self.timer.start(int(1000 / self._animation_fps)) # Initial interval, will be updated

        # Export thread
        self.export_thread: Optional[ExportThread] = None
        self.progress_dialog: Optional[QProgressDialog] = None

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Left panel (Controls)
        left_panel = QWidget()
        left_panel.setMinimumWidth(480)
        left_panel.setMaximumWidth(600)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        # Block picker
        block_picker_group = QGroupBox("Add Block")
        block_picker_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }"
                                         "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }")
        block_picker_layout = QHBoxLayout(block_picker_group)
        self.block_combo = QComboBox()

        gfx_keys = list(getattr(GFX_REGISTRY, "_by_name", {}).keys())
        anim_keys = list(getattr(ANIMATION_REGISTRY, "_by_name", {}).keys())
        all_keys = sorted(list(set(gfx_keys + anim_keys)))
        self.block_combo.addItems(all_keys)
        self.block_combo.setPlaceholderText("Select a block...")
        self.block_combo.setEditable(True) # Allow typing to filter/search
        self.block_combo.lineEdit().setPlaceholderText("Type to filter or select...")


        add_btn = QPushButton("Add Block")
        add_btn.setStyleSheet("height: 30px; background-color: #2a52be; color: white; border-radius: 5px;")
        add_btn.clicked.connect(self.add_block)

        block_picker_layout.addWidget(self.block_combo, 1)
        block_picker_layout.addWidget(add_btn)
        left_layout.addWidget(block_picker_group)

        # Animation Controls
        anim_controls_group = QGroupBox("Animation Controls")
        anim_controls_group.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }"
                                          "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }")
        anim_controls_layout = QVBoxLayout(anim_controls_group)

        # Play/Pause and Duration
        play_dur_layout = QHBoxLayout()
        self.play_pause_button = QPushButton("▶ Play")
        self.play_pause_button.setFixedSize(60, 30)
        self.play_pause_button.setStyleSheet("font-weight: bold; background-color: #333; color: white; border-radius: 5px;")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        play_dur_layout.addWidget(self.play_pause_button)

        play_dur_layout.addWidget(QLabel("Duration (s):"))
        self.duration_spinbox = QDoubleSpinBox()
        self.duration_spinbox.setMinimum(0.1)
        self.duration_spinbox.setMaximum(3600.0) # 1 hour
        self.duration_spinbox.setSingleStep(0.5)
        self.duration_spinbox.setValue(self._animation_duration)
        self.duration_spinbox.valueChanged.connect(self.set_duration)
        play_dur_layout.addWidget(self.duration_spinbox)

        play_dur_layout.addWidget(QLabel("FPS:"))
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setMinimum(1)
        self.fps_spinbox.setMaximum(120)
        self.fps_spinbox.setSingleStep(1)
        self.fps_spinbox.setValue(int(self._animation_fps))
        self.fps_spinbox.valueChanged.connect(self.set_fps)
        play_dur_layout.addWidget(self.fps_spinbox)

        anim_controls_layout.addLayout(play_dur_layout)

        # Time slider and label
        time_slider_layout = QHBoxLayout()
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(int(self._animation_duration * self._animation_fps)) # Max frames
        self.time_slider.setSingleStep(1)
        self.time_slider.setPageStep(self.fps_spinbox.value()) # Jump by 1 second
        self.time_slider.sliderPressed.connect(self.pause_animation)
        self.time_slider.sliderMoved.connect(self.set_current_frame_from_slider)
        self.time_slider.sliderReleased.connect(self.update_preview) # Update preview only after release

        self.time_label = QLabel("Time: 0.00s (Frame 0)")
        self.time_label.setMinimumWidth(150)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.time_label.setStyleSheet("font-size: 11px; color: #bbb;")

        time_slider_layout.addWidget(self.time_slider)
        time_slider_layout.addWidget(self.time_label)
        anim_controls_layout.addLayout(time_slider_layout)

        # Export button
        self.export_button = QPushButton("Export Animation (MP4)")
        self.export_button.setStyleSheet("height: 30px; background-color: #007bff; color: white; font-weight: bold; border-radius: 5px;")
        self.export_button.clicked.connect(self.export_animation_video)
        anim_controls_layout.addWidget(self.export_button)

        left_layout.addWidget(anim_controls_group)

        # Stack of blocks
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("background-color: #222; border: 1px solid #444; border-radius: 5px;")
        self.stack_widget = QWidget()
        self.stack_layout = QVBoxLayout(self.stack_widget)
        self.stack_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.stack_layout.setContentsMargins(5, 5, 5, 5)
        self.stack_layout.setSpacing(5)
        self.scroll.setWidget(self.stack_widget)
        left_layout.addWidget(self.scroll, 1)

        # Right preview
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background: black; border: 5px solid #111; border-radius: 5px;")
        self.preview_label.setMinimumSize(800, 600) # Ensure a base size for the preview

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.preview_label)
        splitter.setSizes([500, 900]) # Initial distribution for splitter
        main_layout.addWidget(splitter)

        # Initialize time label and slider
        self._update_time_slider_range()
        self._update_time_slider_position()
        self._update_time_label()
        self.update_preview() # Render initial frame
    def _get_preview_audio_path(self) -> str | None:
        """Infer the audio file from the current pipeline (same as export)."""
        if not self.blocks:
            return None
        pipeline_names = [b.block_name for b in self.blocks]
        extras_list = [b.get_stage_extras() for b in self.blocks]
        p = _find_visualizer_audio_path(pipeline_names, extras_list)
        if p and os.path.exists(p):
            return p
        return None

    def _ensure_audio_loaded(self):
        """Load (or clear) audio based on current pipeline."""
        p = self._get_preview_audio_path()
        if p == self._current_audio_path:
            return

        self._current_audio_path = p

        # stop any existing audio
        try:
            self.audio_player.stop()
        except Exception:
            pass

        if not p:
            self.audio_player.setSource(QUrl())  # clear source
            return

        self.audio_player.setSource(QUrl.fromLocalFile(p))

    def _audio_seek_to_current_time(self):
        """Seek audio player to self._current_time."""
        if not self._current_audio_path:
            return
        ms = int(max(0.0, self._current_time) * 1000.0)
        # Only seek if we’re meaningfully out of sync (prevents spam-seeking)
        if abs(self.audio_player.position() - ms) > 80:
            self.audio_player.setPosition(ms)

    def _audio_play(self):
        """Start audio in sync with current time."""
        self._ensure_audio_loaded()
        if not self._current_audio_path:
            return
        self._audio_seek_to_current_time()
        self.audio_player.play()

    def _audio_pause(self):
        try:
            self.audio_player.pause()
        except Exception:
            pass

    def _audio_stop(self):
        try:
            self.audio_player.stop()
        except Exception:
            pass
    def _update_time_slider_range(self):
        total_frames = int(round(self._animation_duration * self._animation_fps))
        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(total_frames)
        self.time_slider.blockSignals(False)

    def _update_time_slider_position(self):
        current_frame = int(round(self._current_time * self._animation_fps))
        # Ensure the value is within the slider's range
        current_frame = max(0, min(current_frame, self.time_slider.maximum()))
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(current_frame)
        self.time_slider.blockSignals(False)

    def _update_time_label(self):
        current_frame = int(round(self._current_time * self._animation_fps))
        self.time_label.setText(f"Time: {self._current_time:.2f}s (Frame {current_frame})")

    def _on_timer_tick(self):
        if not self._is_playing:
            return

        # If audio exists, let audio drive time
        if self._current_audio_path:
            pos_ms = self.audio_player.position()
            self._current_time = max(0.0, pos_ms / 1000.0)

            # Loop behavior: if we reach the animation end, restart both
            if self._current_time >= self._animation_duration:
                self._current_time = 0.0
                self.audio_player.setPosition(0)
                # keep playing
        else:
            # No audio -> old behavior
            self._current_time += (1.0 / self._animation_fps)
            if self._current_time >= self._animation_duration:
                self._current_time = 0.0

        self._update_time_slider_position()
        self._update_time_label()
        self.update_preview()

    def toggle_play_pause(self):
        self._is_playing = not self._is_playing

        if self._is_playing:
            self.play_pause_button.setText("⏸ Pause")

            # always re-evaluate which audio file is in the pipeline
            self._ensure_audio_loaded()

            # if near end, reset
            if self._current_time >= self._animation_duration - (1.0 / self._animation_fps):
                self._current_time = 0.0

            # start audio synced
            self._audio_play()

            self.timer.start(int(1000 / self._animation_fps))
        else:
            self.play_pause_button.setText("▶ Play")
            self._audio_pause()
            self.update_preview()

    def pause_animation(self):
        if self._is_playing:
            self.toggle_play_pause() # This will pause it and update button text

    def set_duration(self, value: float):
        self._animation_duration = value
        self._update_time_slider_range()
        if self._current_time > self._animation_duration:
            self._current_time = self._animation_duration # Snap to new end
            self._update_time_slider_position()
        self._update_time_label()
        self.update_preview()

    def set_fps(self, value: int):
        self._animation_fps = float(value)
        self.timer.setInterval(int(1000 / self._animation_fps))
        self._update_time_slider_range()
        self._update_time_slider_position() # Update slider for new frame count
        self._update_time_label()
        self.update_preview()

    def set_current_frame_from_slider(self, frame_index: int):
        self._current_time = frame_index / self._animation_fps
        self._update_time_label()

        # keep audio aligned while scrubbing
        self._ensure_audio_loaded()
        self._audio_seek_to_current_time()

    def add_block(self):
        name = self.block_combo.currentText().strip().lower()

        block_cls = (
            getattr(GFX_REGISTRY, "_by_name", {}).get(name)
            or getattr(ANIMATION_REGISTRY, "_by_name", {}).get(name)
        )
        if not block_cls:
            QMessageBox.warning(self, "Block Not Found", f"Error: Class for '{name}' not found in any registry.")
            print(f"Error: Class for '{name}' not found.")
            return

        ctrl = BlockControl(name, block_cls, len(self.blocks))
        ctrl.remove_requested.connect(self.remove_block)
        ctrl.moved_up.connect(self.move_up)
        ctrl.moved_down.connect(self.move_down) # Connect new signal
        ctrl.params_changed.connect(self.update_preview)

        self.blocks.append(ctrl)
        self.stack_layout.addWidget(ctrl)
        self.refresh_ui()
        self.update_preview()

    def remove_block(self, ctrl: BlockControl):
        if ctrl in self.blocks:
            self.blocks.remove(ctrl)
        ctrl.deleteLater()
        QTimer.singleShot(50, self.refresh_ui) # Use singleShot to allow UI to update before re-indexing

    def move_up(self, ctrl: BlockControl):
        try:
            idx = self.blocks.index(ctrl)
        except ValueError:
            return
        if idx > 0:
            self.blocks[idx], self.blocks[idx - 1] = self.blocks[idx - 1], self.blocks[idx]
            self.refresh_ui()
            self.update_preview()

    def move_down(self, ctrl: BlockControl):
        try:
            idx = self.blocks.index(ctrl)
        except ValueError:
            return
        if idx < len(self.blocks) - 1:
            self.blocks[idx], self.blocks[idx + 1] = self.blocks[idx + 1], self.blocks[idx]
            self.refresh_ui()
            self.update_preview()

    def refresh_ui(self):
        # Re-add widgets to layout to reflect new order and update titles
        for b in self.blocks:
            self.stack_layout.removeWidget(b) # Remove without deleting
        for i, b in enumerate(self.blocks):
            self.stack_layout.addWidget(b)
            b.setTitle(f"Stage {i}: {b.block_name.upper()}")
        self.update_preview() # Re-render with new pipeline order

    def update_preview(self):
        if not self.blocks:
            # Display a blank/placeholder image if no blocks are present
            blank_img = Image.new("RGBA", (self.engine.width, self.engine.height), (0, 0, 0, 255))
            self._display_pil_image(blank_img)
            return

        try:
            pipeline = [b.block_name for b in self.blocks]
            extras = [b.get_stage_extras() for b in self.blocks]

            ctx = AnimationContext(
                frame=int(self._current_time * self._animation_fps),
                total_frames=int(self._animation_duration * self._animation_fps),
                time=self._current_time,
                duration=self._animation_duration,
                fps=self._animation_fps,
                width=self.engine.width,
                height=self.engine.height,
            )

            pil_img = self.engine.render_frame(pipeline=pipeline, extras=extras, ctx=ctx)
            self._display_pil_image(pil_img)
            self._ensure_audio_loaded()
        except Exception:
            # print(traceback.format_exc()) # Uncomment for debugging
            # Display an error image if rendering fails
            error_img = Image.new("RGBA", (self.engine.width, self.engine.height), (255, 0, 0, 255))
            draw = ImageDraw.Draw(error_img)
            draw.text((10, 10), "RENDER ERROR", fill=(255, 255, 255, 255))
            self._display_pil_image(error_img)

    def _display_pil_image(self, pil_img: Image.Image):
        """Helper to convert PIL Image to QPixmap and display."""
        if pil_img.mode != "RGBA":
            pil_img = pil_img.convert("RGBA")
        data = pil_img.tobytes("raw", "RGBA")
        qim = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qim)
        # Scale pixmap to fit label if necessary, maintaining aspect ratio
        scaled_pix = pix.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(scaled_pix)

    def resizeEvent(self, event):
        """Handle resize events to scale the preview image."""
        super().resizeEvent(event)
        # Re-display the current image to scale it correctly
        self.update_preview()

    def export_animation_video(self):
        if not self.blocks:
            QMessageBox.warning(self, "Export Failed", "Add some blocks to the pipeline before exporting.")
            return

        # Pause animation during export
        if self._is_playing:
            self.toggle_play_pause()

        # Get output file path from user
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Animation", "animation.mp4", "MP4 Video (*.mp4);;All Files (*)"
        )
        if not file_path:
            return # User cancelled

        # Gather current pipeline and parameters
        pipeline_names = [b.block_name for b in self.blocks]
        extras_list = [b.get_stage_extras() for b in self.blocks]

        audio_path = _find_visualizer_audio_path(pipeline_names, extras_list)
        if audio_path and not os.path.exists(audio_path):
            QMessageBox.warning(self, "Audio Missing",
                                f"Visualizer audio file not found:\n{audio_path}\nExporting video without audio.")
            audio_path = None

        # Initialize progress dialog
        self.progress_dialog = QProgressDialog("Exporting Animation...", "Cancel", 0, 100, self)
        self.progress_dialog.setWindowTitle("Exporting Video")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.canceled.connect(self._cancel_export)
        self.progress_dialog.show()

        # Disable main UI elements during export
        self.setEnabled(False)

        # Start export in a separate thread
        self.export_thread = ExportThread(
            engine=self.engine,
            pipeline=pipeline_names,
            extras=extras_list,
            outfile=file_path,
            width=self.engine.width,
            height=self.engine.height,
            duration=self._animation_duration,
            fps=self._animation_fps,
            audio_path=audio_path,
        )
        self.export_thread.progress_update.connect(self._on_export_progress)
        self.export_thread.export_finished.connect(self._on_export_finished)
        self.export_thread.start()

    def _cancel_export(self):
        if self.export_thread and self.export_thread.isRunning():
            self.export_thread.cancel()
            QMessageBox.information(self, "Export Cancelled", "Video export has been cancelled.")

    def _on_export_progress(self, percent: int, message: str):
        if self.progress_dialog:
            self.progress_dialog.setValue(percent)
            self.progress_dialog.setLabelText(message)
            # Ensure progress dialog stays responsive
            QCoreApplication.processEvents()

    def _on_export_finished(self, success: bool, message: str):
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        self.setEnabled(True) # Re-enable main UI

        if success:
            QMessageBox.information(self, "Export Complete", message)
        else:
            QMessageBox.critical(self, "Export Failed", message)

        self.export_thread = None # Clear thread reference


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorGroup.All, QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)

    gui = GraphicsGUI()
    gui.show()
    sys.exit(app.exec())
