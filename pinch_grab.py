"""PinchGrab: control the mouse with your hand through the webcam.

- Point with your hand: your index knuckle (blue circle in the preview) moves the cursor.
- Pinch (touch thumb + index): presses and holds the left mouse button, i.e. grabs.
- Move while pinched: drags whatever you grabbed.
- Open your fingers: lets go.

On start you pick which camera to use; the choice is remembered for next time.

Keys:  Ctrl+Alt+P (Mac: Ctrl+Option+P) = pause/resume (works from any app).
With the preview window focused also:  q / Esc = quit,  p = pause/resume,  c = next camera.

Runs on Windows and macOS. On macOS, allow Camera and Accessibility access when asked.
"""
import json
import math
import os
import sys
import threading
import time
import tkinter as tk
import traceback
from tkinter import messagebox

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_tasks
from pynput import keyboard
from mediapipe.tasks.python import vision

IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

# In a packaged app (PyInstaller) bundled files live in sys._MEIPASS, not next to this file.
HERE = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(HERE, "hand_landmarker.task")
# Kept in the home folder: a packaged app's own folder may not be writable.
SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".pinchgrab.json")
CAMERA_BACKEND = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_AVFOUNDATION if IS_MAC else cv2.CAP_ANY
# Only this box of the camera image (fractions of width/height) maps to the screen, so you
# can reach the screen edges without your hand leaving the camera's view. It sits low in the
# image and to the right, so your (right) hand can rest low and to the side.
BOX_LEFT, BOX_RIGHT = 0.52, 0.92
BOX_TOP, BOX_BOTTOM = 0.45, 0.85
# Cursor smoothing: 0 = frozen, 1 = no smoothing (jittery).
SMOOTHING = 0.35
# Pinch thresholds, as a fraction of hand size. Two thresholds (hysteresis) stop the
# grab from flickering on and off when your fingers hover near the limit.
PINCH_ON = 0.28
PINCH_OFF = 0.42
# Seconds the cursor stays put right after a pinch, so the click lands where you aimed.
CLICK_HOLD = 0.2
# System-wide pause/resume shortcut: Ctrl+Alt+P (Option is Alt on a Mac). P is matched by its
# key code, because with Ctrl+Alt held the key may be reported with or without its letter.
PAUSE_VK = 35 if IS_MAC else 0x50  # P
HOTKEY_LABEL = "Ctrl+Option+P" if IS_MAC else "Ctrl+Alt+P"
CTRL_KEYS = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
ALT_KEYS = {keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr}

THUMB_TIP, INDEX_TIP, WRIST, INDEX_MCP, MIDDLE_MCP = 4, 8, 0, 5, 9
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

pyautogui.PAUSE = 0
# The hand can push the cursor into a corner, which would otherwise abort the program.
pyautogui.FAILSAFE = False


def list_cameras():
    """Camera names, in the same order as OpenCV's camera indices."""
    if IS_WINDOWS:
        try:
            from pygrabber.dshow_graph import FilterGraph
            names = FilterGraph().get_input_devices()
            if names:
                return names
        except Exception:
            pass
    # No names available: probe the first few indices instead.
    names = []
    for i in range(5):
        cap = cv2.VideoCapture(i, CAMERA_BACKEND)
        if cap.isOpened():
            names.append(f"Camera {i}")
        cap.release()
    return names


def load_settings():
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_settings(settings):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def pick_camera(names, last_name):
    """Small window listing the cameras. Returns the chosen index, or None if closed."""
    choice = {"index": None}
    root = tk.Tk()
    root.title("PinchGrab - choose camera")
    root.attributes("-topmost", True)
    root.resizable(False, False)

    tk.Label(root, text="Which camera should watch your hand?", font=("Segoe UI", 11)).pack(padx=16, pady=(14, 6))
    box = tk.Listbox(root, height=max(len(names), 3), width=40, font=("Segoe UI", 11), activestyle="none")
    for name in names:
        box.insert(tk.END, name)
    box.pack(padx=16)
    start_at = names.index(last_name) if last_name in names else 0
    box.selection_set(start_at)
    box.activate(start_at)
    box.focus_set()

    def start(_event=None):
        sel = box.curselection()
        if sel:
            choice["index"] = sel[0]
            root.destroy()

    box.bind("<Double-Button-1>", start)
    root.bind("<Return>", start)
    root.bind("<Escape>", lambda _e: root.destroy())
    tk.Button(root, text="Start", width=12, font=("Segoe UI", 10), command=start).pack(pady=12)
    tk.Label(root, text="Tip: press c in the preview to switch cameras later.", fg="gray").pack(pady=(0, 10))

    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 3
    root.geometry(f"+{x}+{y}")
    root.mainloop()
    return choice["index"]


def open_camera(index):
    cap = cv2.VideoCapture(index, CAMERA_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def start_pause_hotkey(on_pause):
    """Listen system-wide for the pause shortcut; calls on_pause (on the listener's thread)."""
    held = set()

    def press(key):
        if key in CTRL_KEYS:
            held.add("ctrl")
        elif key in ALT_KEYS:
            held.add("alt")
        elif getattr(key, "vk", None) == PAUSE_VK and held >= {"ctrl", "alt"}:
            on_pause()

    def release(key):
        if key in CTRL_KEYS:
            held.discard("ctrl")
        elif key in ALT_KEYS:
            held.discard("alt")

    listener = keyboard.Listener(on_press=press, on_release=release)
    listener.start()
    return listener


def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def to_screen(x, y, screen_w, screen_h):
    nx = min(max((x - BOX_LEFT) / (BOX_RIGHT - BOX_LEFT), 0.0), 1.0)
    ny = min(max((y - BOX_TOP) / (BOX_BOTTOM - BOX_TOP), 0.0), 1.0)
    return nx * (screen_w - 1), ny * (screen_h - 1)


def draw_hand(frame, lm, pinching):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 2)
    for p in pts:
        cv2.circle(frame, p, 3, (255, 255, 255), -1)
    color = (0, 0, 255) if pinching else (0, 255, 0)
    cv2.line(frame, pts[THUMB_TIP], pts[INDEX_TIP], color, 3)
    cv2.circle(frame, pts[INDEX_MCP], 7, (255, 180, 0), 2)  # aim point


def make_landmarker():
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
    ))


def selftest():
    """Load the bundled model and run one detection; used to check packaged builds."""
    landmarker = make_landmarker()
    blank = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), np.uint8))
    landmarker.detect_for_video(blank, 0)
    landmarker.close()


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Missing model file: {MODEL_PATH}")
    if "--selftest" in sys.argv:
        selftest()
        return

    names = list_cameras()
    if not names:
        raise SystemExit("No cameras found.")
    settings = load_settings()
    cam_index = pick_camera(names, settings.get("camera"))
    if cam_index is None:
        return
    settings["camera"] = names[cam_index]
    save_settings(settings)

    landmarker = make_landmarker()

    cap = open_camera(cam_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {names[cam_index]}.")

    screen_w, screen_h = pyautogui.size()
    cur_x, cur_y = pyautogui.position()
    pinching = False
    paused = False
    black_frames = 0
    hold_until = 0.0
    start = time.monotonic()

    # The hotkey fires on pynput's thread; the main loop picks it up on the next frame.
    pause_requested = threading.Event()
    hotkeys = start_pause_hotkey(pause_requested.set)

    win = f"PinchGrab  ({HOTKEY_LABEL} = pause, q = quit, c = camera)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 400, 300)
    cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)

    try:
        while True:
            ok, frame = cap.read()
            hands = []
            if ok:
                if frame.shape[1] > 640:  # the hand model doesn't need more; bigger is just slower
                    frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
                frame = cv2.flip(frame, 1)  # mirror, so moving right moves the cursor right
                black_frames = black_frames + 1 if frame.mean() < 3 else 0
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                hands = landmarker.detect_for_video(image, int((time.monotonic() - start) * 1000)).hand_landmarks
                status = "PAUSED" if paused else "no hand"
                if black_frames >= 5:
                    # Another app (e.g. a browser tab) holding the camera, or a privacy switch,
                    # makes it send pure black.
                    status = "camera is black - in use by another app/tab? or press c"
            else:
                frame = np.zeros((480, 640, 3), np.uint8)
                status = "camera not available - press c"

            if hands:
                lm = hands[0]
                hand_size = dist(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6
                ratio = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / hand_size

                if not paused:
                    # Aim with the index knuckle: unlike the fingertips, it barely moves
                    # when you pinch, so clicking doesn't shift the cursor.
                    if time.monotonic() >= hold_until:
                        tx, ty = to_screen(lm[INDEX_MCP].x, lm[INDEX_MCP].y, screen_w, screen_h)
                        cur_x += (tx - cur_x) * SMOOTHING
                        cur_y += (ty - cur_y) * SMOOTHING
                        pyautogui.moveTo(int(cur_x), int(cur_y))

                    if not pinching and ratio < PINCH_ON:
                        pinching = True
                        pyautogui.mouseDown()
                        hold_until = time.monotonic() + CLICK_HOLD
                    elif pinching and ratio > PINCH_OFF:
                        pinching = False
                        pyautogui.mouseUp()
                    status = "GRABBING" if pinching else "pointing"

                draw_hand(frame, lm, pinching)
            elif pinching:
                # Hand left the camera: drop whatever was grabbed.
                pinching = False
                pyautogui.mouseUp()

            h, w = frame.shape[:2]
            cv2.rectangle(frame, (int(w * BOX_LEFT), int(h * BOX_TOP)), (int(w * BOX_RIGHT), int(h * BOX_BOTTOM)),
                          (255, 180, 0), 1)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9 if len(status) < 20 else 0.55,
                        (0, 0, 255) if pinching else (0, 255, 0), 2)
            cv2.putText(frame, names[cam_index], (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p") or pause_requested.is_set():
                pause_requested.clear()
                paused = not paused
                if paused and pinching:
                    pinching = False
                    pyautogui.mouseUp()
            if key == ord("c") and len(names) > 1:
                if pinching:
                    pinching = False
                    pyautogui.mouseUp()
                cap.release()
                cam_index = (cam_index + 1) % len(names)
                cap = open_camera(cam_index)
                black_frames = 0
                settings["camera"] = names[cam_index]
                save_settings(settings)
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        hotkeys.stop()
        if pinching:
            pyautogui.mouseUp()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


def show_error(message):
    # The packaged app has no console, so problems have to be shown in a window.
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror("PinchGrab", message)
    root.destroy()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if isinstance(e.code, str) and "--selftest" not in sys.argv:
            show_error(e.code)
        raise
    except Exception:
        if "--selftest" in sys.argv:
            sys.exit(1)  # no dialog: a build machine has nobody to click it away
        show_error("PinchGrab crashed:\n\n" + traceback.format_exc(limit=3))
        raise
