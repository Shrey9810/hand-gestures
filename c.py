import cv2
import mediapipe as mp
import math
import ctypes
import time
import threading
import subprocess

import pystray
from pystray import MenuItem as item
from PIL import Image, ImageDraw

# ================= WINDOWS VOLUME KEYS =================
VK_VOLUME_UP = 0xAF
VK_VOLUME_DOWN = 0xAE

def volume_up():
    ctypes.windll.user32.keybd_event(VK_VOLUME_UP, 0, 0, 0)

def volume_down():
    ctypes.windll.user32.keybd_event(VK_VOLUME_DOWN, 0, 0, 0)

def set_brightness(level):
    level = int(max(0, min(100, level)))
    try:
        subprocess.run(
            ["powershell",
             f"(Get-WmiObject -Namespace root/WMI "
             f"-Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,{level})"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
    except:
        pass

# ================= GLOBAL STATE =================
running = True
current_mode = "IDLE"    # IDLE | VOLUME | BRIGHTNESS

# ----- Volume Settings (SMOOTH & TIGHT) -----
current_volume = 50
volume_anchor_x = None

# High Sensitivity: detects small movements
VOLUME_SENSITIVITY = 300  
# Tiny Deadzone: reacts instantly
VOLUME_DEADZONE = 0.002   
# 🔒 Safety Cap: Max 1 step per frame (Prevents jumping)
MAX_VOL_STEP = 1          

# ----- Brightness Settings (SMOOTH & TIGHT) -----
current_brightness = 50
brightness_anchor_scale = None

# Very High Sensitivity: small depth change triggers action
BRIGHTNESS_SENSITIVITY = 400 
# Tiny Deadzone
BRIGHTNESS_DEADZONE = 0.005  
BRIGHTNESS_RANGE = 0.12
# 🔒 Safety Cap: Max 1% change per frame (Butter smooth)
MAX_BRIGHTNESS_STEP = 15      

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= UTILS =================
def finger_extended(hand, tip, pip):
    return hand.landmark[tip].y < hand.landmark[pip].y

def is_index_only(hand):
    return (
        finger_extended(hand, 8, 6) and
        not finger_extended(hand, 12, 10) and
        not finger_extended(hand, 16, 14) and
        not finger_extended(hand, 20, 18)
    )

def is_open_hand(hand):
    return all(
        finger_extended(hand, t, p)
        for t, p in [(8,6),(12,10),(16,14),(20,18)]
    )

def hand_scale(hand):
    wrist = hand.landmark[0]
    tips = [4, 8, 12, 16, 20]
    return sum(
        math.hypot(hand.landmark[t].x - wrist.x,
                   hand.landmark[t].y - wrist.y)
        for t in tips
    ) / len(tips)

# ================= CAMERA LOOP (HEADLESS) =================
def gesture_loop():
    global running, current_mode
    global current_volume, volume_anchor_x
    global current_brightness, brightness_anchor_scale

    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            h = res.multi_hand_landmarks[0]

            # ---------- VOLUME ----------
            if is_index_only(h):
                if current_mode != "VOLUME":
                    current_mode = "VOLUME"
                    volume_anchor_x = h.landmark[8].x

                ix = h.landmark[8].x
                dx = ix - volume_anchor_x

                if abs(dx) > VOLUME_DEADZONE:
                    # Calculate raw steps
                    raw_delta = int(dx * VOLUME_SENSITIVITY)
                    
                    # CLAMP: Limit the speed to MAX_VOL_STEP
                    # This ensures smooth 1-by-1 increments
                    delta = max(-MAX_VOL_STEP, min(MAX_VOL_STEP, raw_delta))
                    
                    if delta != 0:
                        for _ in range(abs(delta)):
                            volume_up() if delta > 0 else volume_down()
                        # Update anchor to current pos to require continuous movement
                        volume_anchor_x = ix

            # ---------- BRIGHTNESS ----------
            elif is_open_hand(h):
                if current_mode != "BRIGHTNESS":
                    current_mode = "BRIGHTNESS"
                    brightness_anchor_scale = hand_scale(h)

                scale = hand_scale(h)
                d = brightness_anchor_scale - scale

                # Clamp depth range
                d = max(-BRIGHTNESS_RANGE, min(BRIGHTNESS_RANGE, d))

                if abs(d) > BRIGHTNESS_DEADZONE:
                    raw_delta = int((d / BRIGHTNESS_RANGE) * BRIGHTNESS_SENSITIVITY)
                    
                    # CLAMP: Limit to MAX_BRIGHTNESS_STEP
                    delta = max(-MAX_BRIGHTNESS_STEP, min(MAX_BRIGHTNESS_STEP, raw_delta))

                    if delta != 0:
                        current_brightness = max(0, min(100, current_brightness + delta))
                        set_brightness(current_brightness)
                        brightness_anchor_scale = scale

            # ---------- IDLE ----------
            else:
                current_mode = "IDLE"
                volume_anchor_x = None
                brightness_anchor_scale = None

        time.sleep(0.01)

    cap.release()

# ================= TRAY =================
def icon_img():
    img = Image.new("RGB",(64,64),"black")
    d = ImageDraw.Draw(img)
    d.ellipse((16,16,48,48), fill="cyan")
    return img

def exit_app(icon,_):
    global running
    running = False
    icon.stop()

def tray_menu():
    return pystray.Menu(
        item("Exit", exit_app)
    )

# ================= MAIN =================
if __name__ == "__main__":
    threading.Thread(target=gesture_loop, daemon=True).start()
    pystray.Icon(
        "GestureControl",
        icon_img(),
        "Gesture Volume & Brightness",
        tray_menu()
    ).run()