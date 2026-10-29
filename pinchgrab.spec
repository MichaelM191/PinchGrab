# PyInstaller build recipe. Build with:  pyinstaller pinchgrab.spec
# Output: dist/PinchGrab/PinchGrab.exe on Windows, dist/PinchGrab.app on macOS.
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

# MediaPipe loads its native library and data files at runtime, which PyInstaller can't
# discover on its own; pynput picks its OS backend dynamically.
datas, binaries, hiddenimports = collect_all("mediapipe")
datas += [("hand_landmarker.task", ".")]
hiddenimports += collect_submodules("pynput")
if sys.platform == "win32":
    hiddenimports += collect_submodules("pygrabber")

a = Analysis(
    ["pinch_grab.py"],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PinchGrab",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="PinchGrab")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PinchGrab.app",
        bundle_identifier="com.pinchgrab.app",
        info_plist={
            # Without this macOS refuses camera access outright.
            "NSCameraUsageDescription": "PinchGrab watches your hand to move the mouse.",
            "NSHighResolutionCapable": True,
        },
    )
