import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
import math
import pystray
from pystray import MenuItem as item
from PIL import Image, ImageDraw

# ================= ENHANCED SETTINGS =================
BASE_SMOOTHING = 3
FRAME_MARGIN = 80

# Click timings
MIDDLE_CLICK_HOLD_TIME = 0.5      # Hold middle finger up for 0.3s for single click
FIST_HOLD_TIME = 0.5              # Hold fist for 0.7s for double click
OPEN_HAND_HOLD_TIME = 0.5         # Hold open hand for 0.5s for right click

# Scroll settings
SCROLL_SENSITIVITY = 80
SCROLL_DEADZONE = 0.008
SCROLL_COOLDOWN = 0.05

# Movement boundary
VIRTUAL_WIDTH, VIRTUAL_HEIGHT = 1.0, 1.0

pyautogui.MINIMUM_DURATION = 0
pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

# ================= ENHANCED GLOBAL STATE =================
running = True
screen_w, screen_h = pyautogui.size()
plocX, plocY = 0, 0 
prev_norm_x, prev_norm_y = 0.5, 0.5
velocity = 0.0

# Middle finger click state
middle_click_start_time = 0
middle_detected = False
middle_click_done = False

# Gesture State
active_mode = "NONE"
last_mode_switch = time.time()
mode_cooldown = 0.2

# Fist detection
fist_start_time = 0
fist_detected = False
fist_action_done = False

# Open hand detection
open_hand_start_time = 0
open_hand_detected = False
open_hand_action_done = False

# Scroll State
scroll_start_y = 0
last_scroll_time = 0
scroll_accumulator = 0
is_scrolling = False

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= ENHANCED UTILS =================
def get_dist(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)

def is_finger_up(landmark, pip_joint):
    """Check if finger is extended (up)"""
    # More lenient threshold for better detection
    return landmark.y < pip_joint.y - 0.02

def is_finger_down(landmark, pip_joint):
    """Check if finger is curled (down)"""
    # More lenient threshold for better detection
    return landmark.y > pip_joint.y + 0.02

def icon_img():
    img = Image.new("RGB", (64, 64), "black")
    d = ImageDraw.Draw(img)
    d.polygon([(16,16), (48,32), (32,32), (32,48), (16,16)], fill="cyan")
    return img

def calculate_velocity(curr_x, curr_y, prev_x, prev_y, dt=0.01):
    """Calculate movement velocity for dynamic smoothing"""
    dist = math.hypot(curr_x - prev_x, curr_y - prev_y)
    return dist / dt

def is_fist(hand):
    """Check if hand is making a fist (all fingers curled)"""
    # Landmarks for finger tips
    thumb_tip = hand.landmark[4]
    index_tip = hand.landmark[8]
    middle_tip = hand.landmark[12]
    ring_tip = hand.landmark[16]
    pinky_tip = hand.landmark[20]
    
    # Check if fingers are curled (tips below PIP joints)
    thumb_curled = thumb_tip.y > hand.landmark[3].y
    index_curled = index_tip.y > hand.landmark[6].y
    middle_curled = middle_tip.y > hand.landmark[10].y
    ring_curled = ring_tip.y > hand.landmark[14].y
    pinky_curled = pinky_tip.y > hand.landmark[18].y
    
    # At least 4 fingers should be curled for fist detection
    curled_count = sum([thumb_curled, index_curled, middle_curled, ring_curled, pinky_curled])
    
    return curled_count >= 4

def is_open_hand(hand):
    """Check if all fingers are extended (open hand)"""
    # Landmarks for finger tips
    thumb_tip = hand.landmark[4]
    index_tip = hand.landmark[8]
    middle_tip = hand.landmark[12]
    ring_tip = hand.landmark[16]
    pinky_tip = hand.landmark[20]
    
    # Check if fingers are extended
    thumb_extended = thumb_tip.y < hand.landmark[3].y
    index_extended = index_tip.y < hand.landmark[6].y
    middle_extended = middle_tip.y < hand.landmark[10].y
    ring_extended = ring_tip.y < hand.landmark[14].y
    pinky_extended = pinky_tip.y < hand.landmark[18].y
    
    # At least 4 fingers should be extended for open hand
    extended_count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
    
    return extended_count >= 4

def is_middle_only(hand):
    """Check if only middle finger is up (for single click)"""
    index_tip = hand.landmark[8]
    middle_tip = hand.landmark[12]
    ring_tip = hand.landmark[16]
    pinky_tip = hand.landmark[20]
    
    # Middle finger should be up
    middle_up = is_finger_up(middle_tip, hand.landmark[10])
    
    # Other fingers (except index which is for cursor) should be down
    index_down = is_finger_down(index_tip, hand.landmark[6])
    ring_down = is_finger_down(ring_tip, hand.landmark[14])
    pinky_down = is_finger_down(pinky_tip, hand.landmark[18])
    
    return middle_up and index_down and ring_down and pinky_down

def is_index_only(hand):
    """Check if only index finger is up (for cursor movement)"""
    index_tip = hand.landmark[8]
    middle_tip = hand.landmark[12]
    ring_tip = hand.landmark[16]
    pinky_tip = hand.landmark[20]
    
    # Index should be up
    index_up = is_finger_up(index_tip, hand.landmark[6])
    
    # Other fingers should be down
    middle_down = is_finger_down(middle_tip, hand.landmark[10])
    ring_down = is_finger_down(ring_tip, hand.landmark[14])
    pinky_down = is_finger_down(pinky_tip, hand.landmark[18])
    
    return index_up and middle_down and ring_down and pinky_down

def is_peace_sign(hand):
    """Check for peace sign (index and middle up, others down) for scroll"""
    index_tip = hand.landmark[8]
    middle_tip = hand.landmark[12]
    ring_tip = hand.landmark[16]
    pinky_tip = hand.landmark[20]
    
    # Index and middle should be up
    index_up = is_finger_up(index_tip, hand.landmark[6])
    middle_up = is_finger_up(middle_tip, hand.landmark[10])
    
    # Ring and pinky should be down
    ring_down = is_finger_down(ring_tip, hand.landmark[14])
    pinky_down = is_finger_down(pinky_tip, hand.landmark[18])
    
    return index_up and middle_up and ring_down and pinky_down

# ================= DYNAMIC CURSOR CONTROL =================
def move_cursor(landmark, w, h):
    """Move cursor with dynamic smoothing based on movement speed"""
    global plocX, plocY, prev_norm_x, prev_norm_y, velocity
    
    # Normalize coordinates
    norm_x = np.interp(landmark.x, 
                      [FRAME_MARGIN/w, 1 - FRAME_MARGIN/w], 
                      [0, VIRTUAL_WIDTH])
    norm_y = np.interp(landmark.y,
                      [FRAME_MARGIN/h, 1 - FRAME_MARGIN/h],
                      [0, VIRTUAL_HEIGHT])
    
    # Clamp to virtual screen
    norm_x = max(0, min(VIRTUAL_WIDTH, norm_x))
    norm_y = max(0, min(VIRTUAL_HEIGHT, norm_y))
    
    # Calculate velocity
    velocity = calculate_velocity(norm_x, norm_y, prev_norm_x, prev_norm_y)
    
    # Dynamic smoothing
    if velocity > 5.0:  # Fast movement
        smoothing = max(1.5, BASE_SMOOTHING - (velocity / 10))
    elif velocity > 2.0:  # Medium movement
        smoothing = BASE_SMOOTHING
    else:  # Slow movement (precision mode)
        smoothing = BASE_SMOOTHING * 2
    
    # Map to actual screen
    target_x = int(norm_x * screen_w)
    target_y = int(norm_y * screen_h)
    
    # Apply dynamic smoothing
    clocX = plocX + (target_x - plocX) / smoothing
    clocY = plocY + (target_y - plocY) / smoothing
    
    # Move cursor
    pyautogui.moveTo(int(clocX), int(clocY))
    plocX, plocY = clocX, clocY
    
    # Update previous position
    prev_norm_x, prev_norm_y = norm_x, norm_y
    
    return norm_x, norm_y, velocity

def handle_cursor_mode(hand, w, h):
    """Handle cursor movement only (no clicking)"""
    ind_tip = hand.landmark[8]
    
    # Move cursor with dynamic smoothing
    _, _, _ = move_cursor(ind_tip, w, h)

def handle_middle_click_gesture():
    """Handle middle finger gesture for single click"""
    global middle_click_start_time, middle_detected, middle_click_done
    
    current_time = time.time()
    
    if not middle_click_done:
        hold_time = current_time - middle_click_start_time
        
        # Trigger single click after hold time
        if hold_time >= MIDDLE_CLICK_HOLD_TIME:
            pyautogui.click()
            middle_click_done = True
            
            # Small delay to avoid immediate re-trigger
            time.sleep(0.2)
    
    return min(hold_time / MIDDLE_CLICK_HOLD_TIME, 1.0)

def handle_scroll_mode(hand, w, h):
    """Handle smooth scrolling"""
    global scroll_start_y, last_scroll_time, scroll_accumulator, is_scrolling
    
    mid_tip = hand.landmark[12]
    
    if not is_scrolling:
        scroll_start_y = mid_tip.y
        is_scrolling = True
        scroll_accumulator = 0
        return
    
    # Calculate vertical movement
    diff_y = scroll_start_y - mid_tip.y  # Positive when moving hand UP
    
    # Add to accumulator
    if abs(diff_y) > SCROLL_DEADZONE:
        scroll_accumulator += diff_y * 5
        
        # Trigger scroll when accumulated enough
        if abs(scroll_accumulator) > 0.002:
            current_time = time.time()
            if current_time - last_scroll_time > SCROLL_COOLDOWN:
                scroll_amount = int(scroll_accumulator * SCROLL_SENSITIVITY * 1000)
                
                if scroll_amount != 0:
                    pyautogui.scroll(scroll_amount)
                    last_scroll_time = current_time
                
                # Reset accumulator but keep a small fraction for smoothness
                scroll_accumulator *= 0.2

def handle_fist_gesture():
    """Handle fist gesture for double click"""
    global fist_start_time, fist_detected, fist_action_done
    
    current_time = time.time()
    
    if not fist_action_done:
        hold_time = current_time - fist_start_time
        
        # Trigger double click after hold time
        if hold_time >= FIST_HOLD_TIME:
            pyautogui.doubleClick()
            fist_action_done = True
            
            # Small delay to avoid immediate re-trigger
            time.sleep(0.3)
    
    return min(hold_time / FIST_HOLD_TIME, 1.0)

def handle_open_hand_gesture():
    """Handle open hand gesture for right click"""
    global open_hand_start_time, open_hand_detected, open_hand_action_done
    
    current_time = time.time()
    
    if not open_hand_action_done:
        hold_time = current_time - open_hand_start_time
        
        # Trigger right click after hold time
        if hold_time >= OPEN_HAND_HOLD_TIME:
            pyautogui.rightClick()
            open_hand_action_done = True
            
            # Small delay to avoid immediate re-trigger
            time.sleep(0.3)
    
    return min(hold_time / OPEN_HAND_HOLD_TIME, 1.0)

def determine_gesture(hand):
    """Determine current gesture with improved detection"""
    # Priority order: Fist -> Open Hand -> Middle Only -> Peace Sign -> Index Only
    
    # Check for fist (double click) - highest priority
    if is_fist(hand):
        return "FIST"
    
    # Check for open hand (right click)
    if is_open_hand(hand):
        return "OPEN_HAND"
    
    # Check for middle finger only (single click)
    if is_middle_only(hand):
        return "MIDDLE_CLICK"
    
    # Check for peace sign (scroll)
    if is_peace_sign(hand):
        return "SCROLL"
    
    # Check for index only (cursor movement)
    if is_index_only(hand):
        return "CURSOR"
    
    return "NONE"

# ================= ENHANCED MOUSE LOOP =================
def mouse_loop():
    global running, plocX, plocY
    global middle_click_start_time, middle_detected, middle_click_done
    global scroll_start_y, last_scroll_time, is_scrolling
    global active_mode, last_mode_switch, velocity
    global fist_start_time, fist_detected, fist_action_done
    global open_hand_start_time, open_hand_detected, open_hand_action_done
    
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Hand control system started successfully!")
    print("Running in background... Use tray icon to exit.")
    
    while running:
        ret, frame = cap.read()
        if not ret: 
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            hand = res.multi_hand_landmarks[0]
            
            # Determine current gesture
            current_time = time.time()
            new_mode = determine_gesture(hand)
            
            # Apply mode switch cooldown
            if (new_mode != active_mode and 
                current_time - last_mode_switch > mode_cooldown):
                
                # Reset previous mode states
                if active_mode == "FIST":
                    fist_detected = False
                    fist_action_done = False
                elif active_mode == "OPEN_HAND":
                    open_hand_detected = False
                    open_hand_action_done = False
                elif active_mode == "MIDDLE_CLICK":
                    middle_detected = False
                    middle_click_done = False
                elif active_mode == "SCROLL":
                    is_scrolling = False
                    scroll_accumulator = 0
                elif active_mode == "CURSOR":
                    pass  # No state to reset for cursor
                
                # Start timing for new gesture modes
                if new_mode == "FIST":
                    fist_start_time = current_time
                    fist_detected = True
                    fist_action_done = False
                elif new_mode == "OPEN_HAND":
                    open_hand_start_time = current_time
                    open_hand_detected = True
                    open_hand_action_done = False
                elif new_mode == "MIDDLE_CLICK":
                    middle_click_start_time = current_time
                    middle_detected = True
                    middle_click_done = False
                
                active_mode = new_mode
                last_mode_switch = current_time
            
            # Execute mode-specific actions
            if active_mode == "CURSOR":
                handle_cursor_mode(hand, w, h)
                
            elif active_mode == "SCROLL":
                handle_scroll_mode(hand, w, h)
                
            elif active_mode == "MIDDLE_CLICK" and middle_detected:
                progress = handle_middle_click_gesture()
                if middle_click_done:
                    # Reset after action complete
                    active_mode = "NONE"
                    middle_detected = False
                
            elif active_mode == "FIST" and fist_detected:
                progress = handle_fist_gesture()
                if fist_action_done:
                    # Reset after action complete
                    active_mode = "NONE"
                    fist_detected = False
                
            elif active_mode == "OPEN_HAND" and open_hand_detected:
                progress = handle_open_hand_gesture()
                if open_hand_action_done:
                    # Reset after action complete
                    active_mode = "NONE"
                    open_hand_detected = False
        
        time.sleep(0.01)

    cap.release()

# ================= TRAY =================
def exit_app(icon, _):
    global running
    running = False
    icon.stop()

def tray_menu():
    return pystray.Menu(
        item("Exit", exit_app)
    )

# ================= MAIN =================
if __name__ == "__main__":
    print("Starting Enhanced Hand Control v6...")
    print("\n" + "="*50)
    print("BACKGROUND MODE - NO CAMERA WINDOW")
    print("="*50)
    print("\nGESTURE GUIDE:")
    print("1. 👆 Index finger = CURSOR MOVEMENT")
    print("2. 🖕 Middle finger = Single Click (hold 0.3s)")
    print("3. ✌️ Peace sign = Scroll")
    print("4. ✊ Fist = Double Click (hold 0.7s)")
    print("5. 🖐️ Open hand = Right Click (hold 0.5s)")
    print("\nSTATUS: Running in background")
    print("• Look for tray icon in system tray")
    print("• Right-click tray icon to exit")
    print("• System will automatically detect your hand gestures")
    print("="*50)
    
    t = threading.Thread(target=mouse_loop, daemon=True)
    t.start()
    pystray.Icon("SmartMouse", icon_img(), "Smart Hand Control v6", tray_menu()).run()