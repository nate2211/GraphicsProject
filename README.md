This project represents a highly modular, high-fidelity generative graphics and animation system. Below is a comprehensive README.md designed for a GitHub repository, incorporating the architecture, features, and setup instructions based on the provided code.
Gemini Graphics Engine

A cinematic, modular, and animation-aware 2D/3D generative graphics suite. The Gemini Graphics Engine combines Python's processing power with FFmpeg's video capabilities to create a high-performance "Generative Lab" for static designs, audio visualizers, and complex animations.
🚀 Key Features
🎞️ Pro Animation Engine

    Animation-Aware Context: Every frame is rendered with a precise AnimationContext (frame index, time, duration, FPS).

    Tweening & Easing: Built-in support for Linear, Quadratic, Sine, and Cubic easing.

    Keyframe Controllers: Orchestrate complex parameter changes over time using string-based keyframes (e.g., 0:0.2, 1:0.8).

    Physics-style Wiggle: Add deterministic, seed-based noise (wobble) to any parameter.

🎥 Cinema-Grade Camera Rig

    Perspective & Lens: Simulated tilt-shift, focal length adjustments, and radial lens distortion (K1/K2).

    Dynamic Modes: Choose between Direct, Target tracking, Orbit around a center, or following a complex Path.

    Auto-Fit Tracking: Computer-vision-style bounding box detection (alpha or luma based) to automatically frame and scale content.

🎨 Modular FX Stack

    Geometric Designs: Draw stars, flowers, n-gons, sunbursts, and Archimedean spirals.

    Mastering-grade Post Optics: Chromatic aberration, film grain, bloom/glow, and directional motion blur.

    Color Grading: Lift/Gamma/Gain color balancing, gradient mapping, and posterization.

🎼 High-Fidelity Audio Visualizers

    FFT Spectrum Bars: Log-spaced frequency bins optimized for music visualization.

    Waveform Rendering: Smooth time-domain visualization with adjustable windowing.

    RMS Energy Meters: Real-time loudness tracking for reactive UI elements.

🛠 Tech Stack

    Language: Python 3.10+

    GUI: PyQt6

    Image Processing: Pillow (PIL), NumPy

    Signal Processing: SciPy

    Video/Audio I/O: FFmpeg

📦 Installation
1. Prerequisites

Ensure you have FFmpeg installed on your system.

    Note: The engine looks for FFmpeg in the path defined in videos.py. For bundled versions, it searches the application root.

2. Python Dependencies
Bash

pip install numpy Pillow PyQt6 requests scipy

3. Running the GUI
Bash

python gui.py

🔧 Block Architecture

The engine uses a Registry Pattern, allowing you to pipe blocks together using the | syntax.
Module	Examples
Designs	star, flower, ngon, sunburst
Detail	bloom, chromaticaberration, motionblur, sharpen
Animation	animateparam, pingpongparam, keyframeparam, wiggleparam
Visualizer	audiobars, audiowaveform, audiorms
Camera	camerapipeline (The master compositing block)
Example Pipeline String

solidcolor | drawstar | bloom | camerapipeline
💾 Exporting Video

The engine includes a dedicated ExportThread that communicates via pipes with FFmpeg. This allows for:

    Frame-Perfect Rendering: No dropped frames regardless of complexity.

    Audio Injection: Automatically muxes source audio into the final MP4.

    High Dims: Supports rendering at any resolution (4K, 1080p, etc.) independent of the preview window.

📜 License

This project is open-source. Please ensure compliance with the FFmpeg license (LGPL/GPL) when distributing binaries.
