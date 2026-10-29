# PinchGrab

Control your mouse with your hand through the webcam.

- **Point:** your index knuckle (blue circle in the preview) moves the cursor. Keep it inside the blue box.
- **Pinch** thumb and index finger: grab (hold the left mouse button). Move while pinched to drag.
- **Open your fingers:** let go.
- **Pause/resume:** Ctrl+Alt+P on Windows, Ctrl+Option+P on Mac (works from any app).
- With the preview window focused: `q` quit, `p` pause, `c` switch camera.

## Download

Get `PinchGrab-Windows.zip` or `PinchGrab-Mac.zip` from the latest release (or the latest
"Build apps" run under Actions). No Python needed.

**Windows:** unzip, open the `PinchGrab` folder, run `PinchGrab.exe`. If SmartScreen warns,
click *More info → Run anyway*.

**Mac (Apple Silicon):** unzip, then right-click `PinchGrab.app` → *Open* → *Open* (needed once,
because the app isn't signed). Allow **Camera** when asked, and add PinchGrab under
*System Settings → Privacy & Security → Accessibility* so it can move the mouse.

## Building it yourself

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt pyinstaller   # Mac: .venv/bin/pip
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
pyinstaller --noconfirm pinchgrab.spec
```

Tweak the box position, smoothing and pinch sensitivity at the top of `pinch_grab.py`.
