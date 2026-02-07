# -*- mode: python ; coding: utf-8 -*-
import os
import sys

block_cipher = None

# --- CONFIGURATION ---
# Path to your local ffmpeg bin folder
# This ensures PyInstaller grabs the exe and bundles it inside the app
ffmpeg_bin_path = r'C:\Users\natem\PycharmProjects\graphicsProject\ffmpeg-8.0-essentials_build\bin'

# List your plugin modules here.
# Because they are loaded via REGISTRY, PyInstaller might not "see" them otherwise.
block_modules = [
    'animations',
    'designs',
    'detail',
    'warp',
    'camera',
    'visualizer',
    'videos',
    'threed',
    'images'
]

added_files = [
    # Include ffmpeg and ffprobe in the root of the bundled app
    (os.path.join(ffmpeg_bin_path, 'ffmpeg.exe'), '.'),
    (os.path.join(ffmpeg_bin_path, 'ffprobe.exe'), '.'),
]

a = Analysis(
    ['gui.py'],  # Your main entry point for the GUI version
    pathex=[],
    binaries=added_files,
    datas=[],
    hiddenimports=block_modules + [
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'numpy',
        'PIL.Image',
        'requests'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GeminiGraphicsEngine',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False, # Set to True if you need to see terminal errors for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['icon.ico'] if os.path.exists('icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GeminiGraphicsEngine',
)