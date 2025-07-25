# YOLO-based Video Frame Extractor for Conveyor Belt (using Ultralytics)
#
# This script uses an optimized two-stage process with BATCH PROCESSING for maximum speed.
# It employs an "event-driven" logic to ensure the highest quality frames are captured.
# It waits until an object or group has passed, then selects the single best frame from
# that "event", preventing clipped objects and redundant saves.

# --- Prerequisites ---
# You need to install the following Python libraries.
# pip install ultralytics opencv-python numpy

# --- IMPORTANT: MODEL SETUP ---
# The 'ultralytics' library handles everything for you. The first time you run
# this script, it will automatically download the 'yolov8l.pt' model file.
# -----------------------------

import cv2
import os
import numpy as np
from ultralytics import YOLO
from collections import deque

# --- Configuration ---
CONFIG = {
    # Path to your video file. Leave empty to open a file dialog.
    "VIDEO_PATH": "",

    # Directory where the extracted frames will be saved.
    "OUTPUT_DIR": "output_frames_yolo",

    # BATCH_SIZE: How many frames to process on the GPU at once.
    # Lowered to 8 to accommodate the larger YOLOv8l model on GPUs with less VRAM.
    "BATCH_SIZE": 8,
    
    # FRAME_SKIP: How many frames to skip between analyses. 
    # A value of 0 means every frame is analyzed (highest quality).
    # A value of 1 processes every 2nd frame, etc.
    "FRAME_SKIP": 1,

    # Confidence Threshold:
    # Only detections with a confidence score higher than this value will be considered.
    "CONFIDENCE_THRESHOLD": 0.4,

    # Stability Frame Count:
    # An object is considered "stable" if it has been detected away from the edges
    # for this many consecutive frames.
    "STABILITY_FRAMES": 4,

    # Event Cooldown:
    # How many frames a group must be absent before its "event" is considered over.
    # This ensures we pick the best frame after the group has fully passed.
    "EVENT_COOLDOWN_FRAMES": 15,

    # Minimum Motion Area:
    # The minimum size of a moving area to trigger the full YOLO analysis.
    # Helps filter out noise and minor camera jitter.
    "MIN_MOTION_AREA": 500,

    # Blur Threshold:
    # The script will reject frames where the object area is blurrier than this value.
    # A higher value means the script is stricter and will only accept sharper images.
    "BLUR_THRESHOLD": 100.0,

    # Edge Proximity Threshold (percentage):
    # A safety margin from the ROI's edge. Objects must be inside this zone.
    "EDGE_THRESHOLD": 0.05
}

def select_file():
    """Opens a file dialog to select a video file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
        )
        return file_path
    except ImportError:
        print("Tkinter is not installed. Please provide the VIDEO_PATH manually.")
        return None

def select_roi(frame):
    """
    Allows the user to select a Region of Interest (ROI) with an interactive window.
    """
    # --- Dynamic Resizing Logic ---
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        max_display_width = int(screen_width * 0.9)
        max_display_height = int(screen_height * 0.85)
    except ImportError:
        max_display_width = 1280
        max_display_height = 720
        print("Warning: Tkinter not found. Falling back to default display size (1280x720).")

    orig_h, orig_w = frame.shape[:2]
    scale_w = max_display_width / orig_w
    scale_h = max_display_height / orig_h
    scale_factor = min(scale_w, scale_h)

    if scale_factor > 1.0:
        scale_factor = 1.0

    display_w = int(orig_w * scale_factor)
    display_h = int(orig_h * scale_factor)
    display_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
    
    # --- State variables for the custom ROI selector ---
    roi_state = {
        'start_point': None,
        'end_point': None,
        'drawing': False,
        'selection_done': False
    }

    def mouse_callback(event, x, y, flags, param):
        """Handles mouse events for drawing the ROI."""
        # Constrain mouse coordinates to be within the window bounds
        x = max(0, min(x, display_w - 1))
        y = max(0, min(y, display_h - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            roi_state['start_point'] = (x, y)
            roi_state['end_point'] = (x, y)
            roi_state['drawing'] = True
            roi_state['selection_done'] = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if roi_state['drawing']:
                roi_state['end_point'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            roi_state['drawing'] = False
            roi_state['selection_done'] = True
            # Ensure start_point is top-left and end_point is bottom-right
            x1, y1 = roi_state['start_point']
            x2, y2 = roi_state['end_point']
            roi_state['start_point'] = (min(x1, x2), min(y1, y2))
            roi_state['end_point'] = (max(x1, x2), max(y1, y2))
    
    window_name = "Select Conveyor Belt ROI"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n" + "="*60)
    print("      ACTION REQUIRED: Select the conveyor belt area")
    print("="*60)

    while True:
        temp_frame = display_frame.copy()
        
        # --- Draw instructions on the frame ---
        font_scale = max(0.5, display_w / 1500) # Scale font based on window width
        thickness = 1 if font_scale < 0.7 else 2
        instructions1 = "Click and drag to select ROI. Press ENTER to confirm."
        instructions2 = "Press 'c' to cancel and use the full frame."
        cv2.putText(temp_frame, instructions1, (10, int(30 * font_scale * 2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(temp_frame, instructions2, (10, int(60 * font_scale * 2)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        if roi_state['start_point'] and roi_state['end_point']:
            if roi_state['drawing']:
                # Draw blue rectangle and crosshair while dragging
                cv2.rectangle(temp_frame, roi_state['start_point'], roi_state['end_point'], (255, 0, 0), 2)
                
                # --- Midpoint Crosshair Logic ---
                x1, y1 = roi_state['start_point']
                x2, y2 = roi_state['end_point']
                # Calculate the center of the current rectangle
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # Draw crosshair lines within the rectangle
                cv2.line(temp_frame, (center_x, y1), (center_x, y2), (255, 0, 0), 1)
                cv2.line(temp_frame, (x1, center_y), (x2, center_y), (255, 0, 0), 1)

            elif roi_state['selection_done']:
                # Draw green rectangle after selection is made (no crosshair)
                cv2.rectangle(temp_frame, roi_state['start_point'], roi_state['end_point'], (0, 255, 0), 2)

        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 or key == 32: # ENTER or SPACE key
            break
        if key == ord('c'):
            roi_state['start_point'] = None # Cancel selection
            break
            
    cv2.destroyAllWindows()

    if not roi_state['start_point'] or not roi_state['selection_done']:
        print("ROI selection cancelled or invalid. Using the full frame.")
        return 0, 0, orig_w, orig_h

    # --- Scale coordinates back to original frame size ---
    x1, y1 = roi_state['start_point']
    x2, y2 = roi_state['end_point']
    
    x_orig = int(x1 / scale_factor)
    y_orig = int(y1 / scale_factor)
    w_orig = int((x2 - x1) / scale_factor)
    h_orig = int((y2 - y1) / scale_factor)
    
    roi_original = (x_orig, y_orig, w_orig, h_orig)

    print(f"Region of Interest selected (scaled to original video size): {roi_original}")
    return roi_original

def calculate_frame_score(boxes, roi_w, roi_h):
    """
    Calculates a score for how 'good' a frame is.
    - If there is 1 object, score is based on how centered it is (higher is better).
    - If there are >1 objects, score is based on total area (larger is better).
    """
    if boxes.size == 0:
        return 0.0, "N/A"

    # --- HYBRID SCORING LOGIC ---
    if len(boxes) == 1:
        # --- Centeredness Score for Single Objects ---
        box = boxes[0]
        roi_center_x, roi_center_y = roi_w / 2, roi_h / 2
        
        box_center_x = (box[0] + box[2]) / 2
        box_center_y = (box[1] + box[3]) / 2

        distance = np.sqrt((box_center_x - roi_center_x)**2 + (box_center_y - roi_center_y)**2)
        max_possible_distance = np.sqrt(roi_center_x**2 + roi_center_y**2)
        score = 1.0 - (distance / max_possible_distance) # Invert distance so higher is better
        score_type = "Centeredness"
        return score, score_type
    else:
        # --- Area Score for Multiple Objects ---
        total_area = 0
        for box in boxes:
            x1, y1, x2, y2 = box
            total_area += (x2 - x1) * (y2 - y1)
        score_type = "Total Area"
        return total_area, score_type

def is_blurry(image, boxes, threshold):
    """
    Checks if the combined area of the detected objects is blurry.
    """
    if boxes.size == 0:
        return False, 0

    min_x = int(np.min(boxes[:, 0]))
    min_y = int(np.min(boxes[:, 1]))
    max_x = int(np.max(boxes[:, 2]))
    max_y = int(np.max(boxes[:, 3]))

    object_area = image[min_y:max_y, min_x:max_x]
    if object_area.size == 0:
        return False, 0
    
    gray = cv2.cvtColor(object_area, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return laplacian_var < threshold, laplacian_var

def process_batch_results(batch_results, batch_data, object_states, active_events, config):
    """
    Processes the results from a batch of frames using an event-driven logic.
    """
    for i, results in enumerate(batch_results):
        full_frame, roi_frame, frame_num = batch_data[i]
        
        current_boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.array([])
        current_ids = set(results.boxes.id.int().cpu().tolist()) if results.boxes.id is not None else set()

        if not current_ids:
            continue

        id_to_box = {int(obj_id): box for obj_id, box in zip(results.boxes.id.int().cpu().tolist(), current_boxes)} if current_ids else {}
        roi_h, roi_w = roi_frame.shape[:2]

        # --- Simplified Stability Check ---
        stable_ids = set()
        for obj_id in current_ids:
            box = id_to_box[obj_id]
            is_at_edge = (box[0] < roi_w * config["EDGE_THRESHOLD"] or box[1] < roi_h * config["EDGE_THRESHOLD"] or 
                          box[2] > roi_w * (1 - config["EDGE_THRESHOLD"]) or box[3] > roi_h * (1 - config["EDGE_THRESHOLD"]))

            if obj_id not in object_states:
                object_states[obj_id] = {'stability_counter': 0}

            if is_at_edge:
                object_states[obj_id]['stability_counter'] = 0
            else:
                object_states[obj_id]['stability_counter'] += 1
            
            if object_states[obj_id]['stability_counter'] >= config["STABILITY_FRAMES"]:
                stable_ids.add(obj_id)
        
        if not stable_ids:
            continue

        # --- Event Tracking Logic ---
        group_key = frozenset(stable_ids)
        stable_boxes = np.array([id_to_box[obj_id] for obj_id in stable_ids])

        blurry, blur_score = is_blurry(roi_frame, stable_boxes, config["BLUR_THRESHOLD"])
        if blurry:
            continue

        score, score_type = calculate_frame_score(stable_boxes, roi_w, roi_h)

        if group_key not in active_events:
            active_events[group_key] = {'candidates': [], 'last_seen': frame_num}
        
        active_events[group_key]['candidates'].append({
            'score': score,
            'score_type': score_type,
            'frame_data': full_frame.copy(),
            'frame_num': frame_num,
            'blur_score': blur_score
        })
        active_events[group_key]['last_seen'] = frame_num

def analyze_video_yolo(video_path, output_dir, config):
    """
    Analyzes the video using a two-stage, batch-processing approach for speed and accuracy.
    """
    print("Loading YOLOv8l model via Ultralytics...")
    try:
        model = YOLO('yolov8l.pt') # Upgraded model for better accuracy
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video file at {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video has a total of {total_frames} frames.")

    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Could not read the first frame of the video.")
        cap.release()
        return
    
    x, y, w, h = select_roi(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    back_sub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
    frame_count = 0
    object_states = {}
    active_events = {}
    best_frames_from_events = {}
    
    batch_data = [] # Stores (full_frame, roi_frame, frame_count)
    roi_batch = []  # Stores just the roi_frame for the model

    print("\nStarting optimized video analysis...")
    print("Step 1: Analyzing video and collecting event candidates...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % (config["FRAME_SKIP"] + 1) != 0:
            continue

        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"\rProgress: {progress:.1f}% (frame {frame_count}/{total_frames})", end="")

        roi_frame = frame[y:y+h, x:x+w]
        if roi_frame.size == 0:
            continue

        fg_mask = back_sub.apply(roi_frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if any(cv2.contourArea(c) > config["MIN_MOTION_AREA"] for c in contours):
            batch_data.append((frame, roi_frame, frame_count))
            roi_batch.append(roi_frame)

        if len(roi_batch) >= config["BATCH_SIZE"]:
            results = model.track(source=roi_batch, conf=config["CONFIDENCE_THRESHOLD"], persist=True, verbose=False)
            process_batch_results(results, batch_data, object_states, active_events, config)
            batch_data, roi_batch = [], []
        
        # Check for finished events
        finished_keys = []
        for key, event in active_events.items():
            if frame_count - event['last_seen'] > config["EVENT_COOLDOWN_FRAMES"]:
                best_candidate = max(event['candidates'], key=lambda c: c['score'])
                best_frames_from_events[key] = best_candidate
                finished_keys.append(key)
        
        for key in finished_keys:
            del active_events[key]

    if roi_batch:
        results = model.track(source=roi_batch, conf=config["CONFIDENCE_THRESHOLD"], persist=True, verbose=False)
        process_batch_results(results, batch_data, object_states, active_events, config)

    # Process any remaining active events at the end of the video
    for key, event in active_events.items():
        best_candidate = max(event['candidates'], key=lambda c: c['score'])
        best_frames_from_events[key] = best_candidate

    cap.release()
    print()
    print(f"Step 1 complete. Found {len(best_frames_from_events)} candidate object groups.")

    print("\nStep 2: Filtering for the most complete frames...")
    final_frames = {}
    sorted_groups = sorted(best_frames_from_events.items(), key=lambda item: len(item[0]), reverse=True)

    for ids, data in sorted_groups:
        if not any(ids.issubset(final_ids) for final_ids in final_frames.keys()):
            final_frames[ids] = data

    print(f"Step 2 complete. Filtered down to {len(final_frames)} unique frames.")

    print("\nStep 3: Saving final frames...")
    saved_count = 0
    for ids, data in final_frames.items():
        saved_count += 1
        file_name = f"frame_{saved_count:05d}.jpg"
        save_path = os.path.join(output_dir, file_name)
        cv2.imwrite(save_path, data['frame_data'])
        score_val = f"{data['score']:.2f}" if data['score_type'] == "Centeredness" else f"{data['score']:.0f}"
        print(f"  - Saving {file_name} (objects: {set(ids)}, frame: {data['frame_num']}, score type: {data['score_type']}, score: {score_val}, sharpness: {data['blur_score']:.2f})")

    print("\nAnalysis complete.")
    print(f"Total unique frames saved: {len(final_frames)}")
    print(f"Frames saved to: '{os.path.abspath(output_dir)}'")


if __name__ == "__main__":
    video_file = CONFIG["VIDEO_PATH"]
    if not video_file:
        video_file = select_file()

    if video_file and os.path.exists(video_file):
        analyze_video_yolo(video_file, CONFIG["OUTPUT_DIR"], CONFIG)
    elif not video_file:
        print("No video file was selected. Exiting.")
    else:
        print(f"CRITICAL ERROR: The file does not exist at the path '{video_file}'.")
        print("Please check for typos or if the file has been moved.")
