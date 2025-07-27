# ==================================================================================================
# YOLO-Based Multi-Angle/Single-File Frame Extractor
# Description:
#   This script provides a powerful and flexible pipeline for extracting high-quality frames
#   of objects from video footage, specifically designed for creating machine learning datasets.
#
# Features:
#   - Two-Stage AI Pipeline: Uses a fast detector (e.g., yolov8n) for initial filtering and a
#     high-quality tracker (e.g., yolov8x) for detailed analysis, optimizing both speed and accuracy.
#   - Multi-Angle Batch Processing: Can automatically find and process a full set of videos
#     (e.g., top, front, back views) based on a shared category name in the filename.
#   - Single-File Mode: Can be easily switched to process only a single, selected video.
#   - Intelligent Frame Selection: Identifies the "best" frame of an object based on its
#     stability and centeredness within a user-defined Region of Interest (ROI).
#   - Burst Frame Extraction: Optionally saves surrounding frames (bursts) for additional context,
#     neatly organized into a separate subfolder.
#   - Flexible Naming Conventions: Supports both simple sequential numbering (ideal for annotation)
#     and descriptive filenames.
#   - Interactive ROI Selection: A user-friendly GUI to draw the processing area for each video.
# ==================================================================================================

import cv2
import os
import re
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict

# --- Main Configuration Block ---
# Adjust the settings below to control the script's behavior.
# ==================================================================================================
CONFIG = {
    # --- I/O Configuration ---
    # Base directory where all output folders and frames will be saved.
    "OUTPUT_DIR": "final_frames_dataset",

    # --- Mode & Naming Configuration ---
    # Master switch for processing mode.
    # True: Multi-angle batch mode. Finds and processes all videos in ANGLES_TO_PROCESS.
    # False: Single-file mode. Processes only the video selected by the user.
    "PROCESS_MULTIPLE_ANGLES": True,

    # List of camera angle prefixes to search for when in batch mode.
    # The script expects filenames like: [angle_prefix]_[category_name].mp4
    "ANGLES_TO_PROCESS": [
        "top_view",
        "front_view",
        "back_view",
        "side_left_view",
        "side_right_view"
    ],
    
    # Filename format for saved images.
    # True: Descriptive names (e.g., 'video_best_123_frame_125.jpg'). Useful for debugging.
    # False: Simple sequential names (e.g., 'video_0.jpg'). Ideal for annotation software.
    "USE_DESCRIPTIVE_FILENAMES": False,

    # --- Frame Extraction & Burst Settings ---
    # Number of frames to save before and after the "best" (hero) frame.
    # If > 0, these burst frames are saved into a 'burst_frames' subfolder.
    # A value of 5 saves 5 frames before + 1 hero frame + 5 frames after.
    "FRAME_BURST_COUNT": 5,

    # --- AI Model & Performance Configuration ---
    # Fast model for pre-filtering frames. Runs on every frame to detect if anything is present.
    "DETECTOR_MODEL": 'yolov8n.pt',
    # High-quality model for tracking and scoring. Runs only on frames that contain objects.
    "TRACKER_MODEL": 'yolov8x.pt',
    # How many frames with objects to process on the GPU at once. Adjust based on VRAM.
    "BATCH_SIZE": 16,

    # --- Detection & Scoring Logic ---
    # Confidence threshold for the high-quality TRACKER_MODEL. Detections below this are ignored.
    "CONFIDENCE_THRESHOLD": 0.3,
    # The best frame for an object must have a centeredness score above this to be saved. (Range: 0.0 to 1.0)
    "MIN_CENTEREDNESS_SCORE": 0.7,
    # An object is "stable" if it's detected away from the ROI edges for this many consecutive frames.
    "STABILITY_FRAMES": 2,
    # A safety margin from the ROI's edge (as a percentage). Objects near the edge are considered unstable.
    "EDGE_THRESHOLD": 0.02,
    # How many frames an object group must be absent before its "event" is considered over and finalized.
    "EVENT_COOLDOWN_FRAMES": 150,
    # Force full analysis on the first N frames to ensure initial objects aren't missed by the fast detector.
    "FORCE_YOLO_FRAMES_START": 2000,

    # --- Debugging ---
    # If True, saves a debug video showing detections and the ROI for each processed video.
    "DEBUG_MODE": True,
    "DEBUG_VIDEO_FILENAME": "debug_output.mp4",
}
# ==================================================================================================


# This master list defines the canonical order for processing and file numbering.
# This ensures that 'top_view' is always processed first, 'front_view' second, etc.,
# which is critical for consistent sequential numbering across multiple angles.
CANONICAL_ANGLE_ORDER = [
    "top_view", "front_view", "back_view", "side_left_view", "side_right_view"
]

def select_file(title="Select a Video File"):
    """Opens a native file dialog for the user to select a video file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        file_path = filedialog.askopenfilename(
            title=title,
            filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
        )
        return file_path
    except ImportError:
        print("Tkinter is not installed. Cannot show file dialog.")
        return None

def find_video_group(selected_video_path, required_angles):
    """
    Identifies the video's category and finds all other required angle videos.

    This function parses the selected filename to extract a category name based on the
    pattern '[angle]_[category].ext'. It then searches the same directory for other
    files that match the required angles and the same category.

    Args:
        selected_video_path (str): The full path to the video file selected by the user.
        required_angles (list): A list of angle prefixes to search for.

    Returns:
        tuple: A tuple containing (video_group, category, selected_angle) on success,
               or (None, None, None) on failure.
               - video_group (dict): Maps angle prefixes to their full video file paths.
               - category (str): The auto-detected category name.
               - selected_angle (str): The angle prefix of the user-selected video.
    """
    if not selected_video_path:
        return None, None, None

    directory = os.path.dirname(selected_video_path)
    filename = os.path.basename(selected_video_path)

    # Use regex to robustly parse the filename for angle and category.
    match = re.match(r"([a-z_]+)_([a-zA-Z0-9_]+)\.(mp4|avi|mov|mkv)$", filename, re.IGNORECASE)
    if not match:
        print(f"\nERROR: Filename '{filename}' does not match pattern: [angle]_[category].ext")
        return None, None, None
    
    selected_angle = match.group(1).lower()
    category = match.group(2)
    
    if selected_angle not in required_angles:
        print(f"\nERROR: The selected angle '{selected_angle}' is not in the ANGLES_TO_PROCESS list in your config.")
        return None, None, None
    
    print(f"\nDetected Category: '{category}'. Searching for {len(required_angles)} required angle(s)...")

    video_group = {}
    all_files_in_dir = os.listdir(directory)

    for angle in required_angles:
        found = False
        for f in all_files_in_dir:
            # Case-insensitive search for robust matching.
            if f.lower().startswith(f"{angle}_{category}".lower()):
                video_group[angle] = os.path.join(directory, f)
                print(f"  - Found: {f}")
                found = True
                break
        if not found:
            print(f"  - ERROR: Could not find required video for angle '{angle}'.")

    if len(video_group) != len(required_angles):
        print("\nCRITICAL ERROR: Did not find all required videos specified in the config. Please check filenames.")
        return None, None, None
    
    print(f"\nSuccessfully found all {len(required_angles)} required videos.")
    return video_group, category, selected_angle

def select_roi(frame, video_name_for_title):
    """
    Displays a window allowing the user to interactively select a Region of Interest (ROI).

    The user can click and drag to draw a box. Pressing ENTER confirms the selection,
    and 'c' cancels and uses the full frame.

    Args:
        frame (np.array): The first frame of the video to display for ROI selection.
        video_name_for_title (str): The name of the video, used for the window title.

    Returns:
        tuple: A tuple (x, y, w, h) representing the ROI in the original video's resolution.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        max_display_width = int(root.winfo_screenwidth() * 0.9)
        max_display_height = int(root.winfo_screenheight() * 0.85)
    except ImportError:
        max_display_width, max_display_height = 1280, 720

    orig_h, orig_w = frame.shape[:2]
    scale = min(max_display_width / orig_w, max_display_height / orig_h, 1.0)
    display_w, display_h = int(orig_w * scale), int(orig_h * scale)
    display_frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
    
    roi_state = {'start_point': None, 'end_point': None, 'drawing': False, 'selection_done': False}

    def mouse_callback(event, x, y, flags, param):
        """Handles mouse events for drawing the ROI."""
        x, y = max(0, min(x, display_w - 1)), max(0, min(y, display_h - 1))
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_state.update({'start_point': (x, y), 'end_point': (x, y), 'drawing': True, 'selection_done': False})
        elif event == cv2.EVENT_MOUSEMOVE and roi_state['drawing']:
            roi_state['end_point'] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            roi_state['drawing'], roi_state['selection_done'] = False, True
            x1, y1 = roi_state['start_point']; x2, y2 = roi_state['end_point']
            roi_state.update({'start_point': (min(x1, x2), min(y1, y2)), 'end_point': (max(x1, x2), max(y1, y2))})
    
    window_name = f"ROI for: {video_name_for_title}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp_frame = display_frame.copy()
        # Display instructions on the window.
        cv2.putText(temp_frame, "Click-drag to select ROI. ENTER to confirm.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(temp_frame, "Press 'c' to use the full frame.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        if roi_state['start_point'] and roi_state['end_point']:
            x1, y1 = roi_state['start_point']
            x2, y2 = roi_state['end_point']
            
            # Draw visual feedback for the user.
            if roi_state['drawing']:
                # Red box and crosshair while drawing.
                cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.line(temp_frame, (cx, y1), (cx, y2), (0, 0, 255), 1)
                cv2.line(temp_frame, (x1, cy), (x2, cy), (0, 0, 255), 1)
            elif roi_state['selection_done']:
                # Green box when selection is made.
                cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Yellow inner box to visualize the edge stability threshold.
                inset_x, inset_y = int((x2-x1) * CONFIG['EDGE_THRESHOLD']), int((y2-y1) * CONFIG['EDGE_THRESHOLD'])
                cv2.rectangle(temp_frame, (x1 + inset_x, y1 + inset_y), (x2 - inset_x, y2 - inset_y), (0, 255, 255), 1)
        
        cv2.imshow(window_name, temp_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [13, 32] and roi_state['selection_done']: break # Confirm with Enter or Space
        if key == ord('c'): roi_state['start_point'] = None; break # Cancel with 'c'
    cv2.destroyAllWindows()

    if not roi_state.get('start_point'):
        print("ROI selection cancelled. Using the full frame.")
        return 0, 0, orig_w, orig_h
    
    # Scale the display ROI back to the original video's dimensions.
    x1, y1 = roi_state['start_point']
    x2, y2 = roi_state['end_point']
    return tuple(int(c / scale) for c in (min(x1,x2), min(y1,y2), abs(x2 - x1), abs(y2 - y1)))

def calculate_frame_score(boxes, roi_w, roi_h):
    """
    Calculates a quality score for a frame based on its detections.
    The primary score is the number of objects, and the secondary score is
    how centered the group of objects is within the ROI.

    Returns:
        tuple: A tuple ((num_objects, centeredness_score), "score_type_string").
    """
    if not isinstance(boxes, np.ndarray) or boxes.size == 0: return (0, 0.0), "N/A"
    
    # Calculate the bounding box of the entire group of objects.
    group_x1, group_y1 = np.min(boxes[:, 0]), np.min(boxes[:, 1])
    group_x2, group_y2 = np.max(boxes[:, 2]), np.max(boxes[:, 3])
    
    roi_cx, roi_cy = roi_w / 2, roi_h / 2
    group_cx, group_cy = (group_x1 + group_x2) / 2, (group_y1 + group_y2) / 2

    # Calculate centeredness: 1.0 is perfect center, 0.0 is at a corner.
    distance = np.sqrt((group_cx - roi_cx)**2 + (group_cy - roi_cy)**2)
    max_dist = np.sqrt(roi_cx**2 + roi_cy**2)
    centeredness = 1.0 - (distance / max_dist if max_dist > 0 else 0)
    
    return (len(boxes), centeredness), "Group Centeredness"

def process_batch_results(batch_data, object_states, active_events, config):
    """
    Processes detection results for a batch of frames, updating object states and events.
    This function tracks objects, determines their stability, and identifies the best
    frame for each group of objects.
    """
    for results, frame_num in batch_data:
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else np.array([])
        ids = set(results.boxes.id.int().cpu().tolist()) if hasattr(results.boxes, 'id') and results.boxes.id is not None else set()
        id_to_box = {int(i): b for i, b in zip(results.boxes.id.int().cpu().tolist(), boxes)} if ids else {}
        
        roi_h, roi_w = results.orig_shape
        stable_ids = set()
        for obj_id in ids:
            box = id_to_box[obj_id]
            # Check if the object is too close to the edge of the ROI.
            at_edge = (box[0] < roi_w * config["EDGE_THRESHOLD"] or box[1] < roi_h * config["EDGE_THRESHOLD"] or 
                       box[2] > roi_w * (1 - config["EDGE_THRESHOLD"]) or box[3] > roi_h * (1 - config["EDGE_THRESHOLD"]))
            
            # Update stability counter: reset if at edge, otherwise increment.
            object_states[obj_id]['stability_counter'] = 0 if at_edge else object_states[obj_id].get('stability_counter', 0) + 1
            if object_states[obj_id]['stability_counter'] >= config["STABILITY_FRAMES"]:
                stable_ids.add(obj_id)
        
        # If there are stable objects, process them as a group event.
        if stable_ids:
            group_key = frozenset(stable_ids)
            active_events.setdefault(group_key, {})['last_seen'] = frame_num
            
            # Calculate the score for the current frame.
            score, _ = calculate_frame_score(np.array([id_to_box[i] for i in stable_ids]), roi_w, roi_h)
            
            # If this frame is better than the previous best, update it.
            if 'best_candidate' not in active_events[group_key] or score > active_events[group_key]['best_candidate']['score']:
                active_events[group_key]['best_candidate'] = {'score': score, 'frame_num': frame_num}

def analyze_single_video(video_path, roi, output_dir, config, models, start_frame_index):
    """
    Performs the full analysis pipeline on a single video file.

    Args:
        video_path (str): Path to the video file.
        roi (tuple): The (x, y, w, h) of the region to analyze.
        output_dir (str): The directory to save output frames and debug video.
        config (dict): The main configuration dictionary.
        models (tuple): A tuple containing the loaded (detector_model, tracker_model).
        start_frame_index (int): The starting number for sequential file naming.

    Returns:
        int: The next available frame index for the subsequent video.
    """
    detector_model, tracker_model = models
    x, y, w, h = roi
    video_name = os.path.basename(video_path)
    print(f"\n--- Starting analysis for: {video_name} ---")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"ERROR: Could not open {video_path}"); return start_frame_index

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Setup debug video writer if enabled.
    debug_writer = None
    if config["DEBUG_MODE"]:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        debug_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_debug.mp4")
        fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (fw, fh))

    # Initialize state-tracking variables for this video.
    object_states, active_events, best_frames = defaultdict(dict), {}, {}
    roi_batch, data_batch = [], []

    # --- Main Processing Loop ---
    for frame_count in range(total_frames):
        ret, frame = cap.read()
        if not ret: break
        
        # Update progress bar.
        if frame_count % 50 == 0:
            print(f"\r  Progress: |{'█' * int(30 * (frame_count+1)/total_frames)}{'-' * (30 - int(30 * (frame_count+1)/total_frames))}| {((frame_count+1)/total_frames):.1%} ({video_name})", end="")
        
        # Crop the frame to the selected ROI.
        roi_frame = frame[y:y+h, x:x+w]
        if roi_frame.size == 0: continue

        # --- Two-Stage AI Pipeline ---
        # 1. Fast detector checks if any object is present.
        # 2. If an object is found (or we are in the forced-start phase), add to batch for the high-quality tracker.
        if frame_count <= config["FORCE_YOLO_FRAMES_START"] or len(detector_model(roi_frame, conf=0.25, verbose=False)[0].boxes) > 0:
            roi_batch.append(roi_frame)
            data_batch.append({'frame_num': frame_count, 'full_frame': frame if config["DEBUG_MODE"] else None})
        elif debug_writer:
            debug_writer.write(frame) # Write empty frames to debug video for context.

        # Process the batch when it's full or the video ends.
        if len(roi_batch) >= config["BATCH_SIZE"] or (not ret and roi_batch):
            results_batch = tracker_model.track(source=roi_batch, conf=config["CONFIDENCE_THRESHOLD"], persist=True, verbose=False)
            if debug_writer:
                for i, results in enumerate(results_batch):
                    df = data_batch[i]['full_frame'].copy(); cv2.rectangle(df, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    if hasattr(results.boxes, 'id') and results.boxes.id is not None:
                        for box in results.boxes.xyxy.cpu().numpy():
                            x1r, y1r, x2r, y2r = map(int, box); cv2.rectangle(df, (x1r+x, y1r+y), (x2r+x, y2r+y), (255, 0, 0), 2)
                    debug_writer.write(df)
            process_batch_results([(r, d['frame_num']) for r, d in zip(results_batch, data_batch)], object_states, active_events, config)
            roi_batch, data_batch = [], []

        # Periodically check for and finalize "finished" events.
        if frame_count % 50 == 0:
            for k in [k for k, e in active_events.items() if frame_count - e.get('last_seen', frame_count) > config["EVENT_COOLDOWN_FRAMES"]]:
                if k in active_events and 'best_candidate' in active_events[k] and active_events[k]['best_candidate']['score'][1] >= config["MIN_CENTEREDNESS_SCORE"]:
                    best_frames[k] = active_events[k]['best_candidate']
                if k in active_events: del active_events[k]

    # Finalize any remaining active events at the end of the video.
    for k, e in active_events.items():
        if 'best_candidate' in e and e['best_candidate']['score'][1] >= config["MIN_CENTEREDNESS_SCORE"]:
            best_frames[k] = e['best_candidate']
    
    print(f"\r  Analysis for {video_name} complete. Found {len(best_frames)} candidate groups.          ")

    # --- De-duplication and Frame Saving ---
    # Filter out overlapping object groups, keeping the one with the most objects.
    final_bursts, master_saved_ids = {}, set()
    for ids, data in sorted(best_frames.items(), key=lambda item: item[1]['score'][0], reverse=True):
        if not ids.isdisjoint(master_saved_ids): continue
        final_bursts[data['frame_num']] = ids; master_saved_ids.update(ids)

    if not final_bursts: print("  No valid frames to save."); cap.release(); return start_frame_index

    # Efficiently grab all needed frames in one pass.
    all_frames_to_grab = {fn for best_fn in final_bursts for fn in range(max(0, best_fn - config["FRAME_BURST_COUNT"]), min(total_frames, best_fn + config["FRAME_BURST_COUNT"] + 1))}
    frames_grabbed = {}
    for frame_num in sorted(list(all_frames_to_grab)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames_grabbed[frame_num] = frame
            
    # --- Frame Saving Logic ---
    hero_saved_count = 0
    burst_saved_count = 0
    output_base_name = os.path.splitext(video_name)[0]
    burst_output_dir = os.path.join(output_dir, "burst_frames")
    use_burst_folder = config.get("FRAME_BURST_COUNT", 0) > 0
    
    if use_burst_folder:
        os.makedirs(burst_output_dir, exist_ok=True)
        print(f"  Saving burst frames to: {burst_output_dir}")

    # Branch logic based on chosen filename convention.
    if config.get("USE_DESCRIPTIVE_FILENAMES", False):
        print("  Using descriptive filenames.")
        for best_frame_num, ids in sorted(final_bursts.items()):
            start_burst = max(0, best_frame_num - config["FRAME_BURST_COUNT"])
            end_burst = min(total_frames, best_frame_num + config["FRAME_BURST_COUNT"] + 1)
            for frame_to_save_num in range(start_burst, end_burst):
                if frame_to_save_num in frames_grabbed and frames_grabbed[frame_to_save_num] is not None:
                    file_name = f"{output_base_name}_best_{best_frame_num:06d}_frame_{frame_to_save_num:06d}.jpg"
                    is_hero_frame = (frame_to_save_num == best_frame_num)
                    save_dir = output_dir if is_hero_frame or not use_burst_folder else burst_output_dir
                    cv2.imwrite(os.path.join(save_dir, file_name), frames_grabbed[frame_to_save_num])
                    if is_hero_frame: hero_saved_count += 1 
                    else: burst_saved_count += 1
    else:
        print("  Using sequential filenames.")
        current_hero_index = start_frame_index
        for best_frame_num in sorted(final_bursts.keys()):
            hero_index_for_this_burst = current_hero_index
            burst_sub_index = 0
            
            # Save the hero frame first to establish its main index.
            if best_frame_num in frames_grabbed and frames_grabbed[best_frame_num] is not None:
                file_name = f"{output_base_name}_{hero_index_for_this_burst}.jpg"
                cv2.imwrite(os.path.join(output_dir, file_name), frames_grabbed[best_frame_num])
                hero_saved_count += 1
            
            # Save burst frames with a sub-index if applicable.
            if use_burst_folder:
                start_burst = max(0, best_frame_num - config["FRAME_BURST_COUNT"])
                end_burst = min(total_frames, best_frame_num + config["FRAME_BURST_COUNT"] + 1)
                for frame_to_save_num in range(start_burst, end_burst):
                    if frame_to_save_num != best_frame_num and frame_to_save_num in frames_grabbed and frames_grabbed[frame_to_save_num] is not None:
                        file_name = f"{output_base_name}_{hero_index_for_this_burst}_{burst_sub_index}.jpg"
                        cv2.imwrite(os.path.join(burst_output_dir, file_name), frames_grabbed[frame_to_save_num])
                        burst_sub_index += 1
                        burst_saved_count += 1
            
            current_hero_index += 1 # Increment main hero counter for the next group.

    print(f"  Saved {hero_saved_count} hero frames and {burst_saved_count} burst frames.")
    cap.release()
    if debug_writer: debug_writer.release()
    
    # Return the next starting index for sequential numbering.
    if config.get("USE_DESCRIPTIVE_FILENAMES", False):
        return hero_saved_count + burst_saved_count
    else:
        return start_frame_index + hero_saved_count

def main():
    """Main function to orchestrate the entire processing workflow."""
    print("--- Multi-Angle/Single-File Video Processor ---")

    # --- 1. Select Mode and Load Models ---
    is_batch_mode = CONFIG.get("PROCESS_MULTIPLE_ANGLES", True)
    
    selected_video = select_file()
    if not selected_video:
        print("No video selected. Exiting.")
        return

    print("\n--- Step 1: Loading AI Models ---")
    try:
        models = (YOLO(CONFIG['DETECTOR_MODEL']), YOLO(CONFIG['TRACKER_MODEL']))
        print("AI models loaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to load YOLO models: {e}"); return

    # --- 2. Branch Workflow Based on Mode ---
    if is_batch_mode:
        # --- BATCH MODE WORKFLOW ---
        print("\nRunning in Multi-Angle Batch Mode.")
        angles_to_process = CONFIG.get("ANGLES_TO_PROCESS", [])
        if not angles_to_process:
            print("ERROR: In Batch Mode, but 'ANGLES_TO_PROCESS' in CONFIG is empty."); return
        print(f"Configured to process {len(angles_to_process)} angle(s): {', '.join(angles_to_process)}")

        video_group, category, selected_angle = find_video_group(selected_video, angles_to_process)
        if not video_group: return

        print("\n--- Step 2: Select Region of Interest (ROI) for each video ---")
        # Start ROI selection with the user's chosen video for better UX.
        roi_selection_order = [selected_angle] + [a for a in CANONICAL_ANGLE_ORDER if a != selected_angle and a in angles_to_process]
        rois = {}
        for angle in roi_selection_order:
            video_path = video_group[angle]
            cap = cv2.VideoCapture(video_path); ret, frame = cap.read(); cap.release()
            if not ret: print(f"ERROR: Could not read frame from {video_path}."); return
            print(f"\nOpening window for: {os.path.basename(video_path)}")
            rois[angle] = select_roi(frame, os.path.basename(video_path))

        print("\n--- Step 3: Processing all videos (in canonical order) ---")
        total_hero_frames_saved = 0
        category_output_dir = os.path.join(CONFIG['OUTPUT_DIR'], category)
        os.makedirs(category_output_dir, exist_ok=True)

        # Process videos in the fixed canonical order for consistent numbering.
        for angle in CANONICAL_ANGLE_ORDER:
            if angle in angles_to_process:
                video_path = video_group[angle]
                roi = rois[angle]
                angle_output_dir = os.path.join(category_output_dir, os.path.splitext(os.path.basename(video_path))[0])
                os.makedirs(angle_output_dir, exist_ok=True)
                
                # Pass the running total to maintain a continuous sequence.
                next_start_index = analyze_single_video(video_path, roi, angle_output_dir, CONFIG, models, total_hero_frames_saved)
                total_hero_frames_saved = next_start_index
        
        print(f"\n\n--- Batch Workflow Complete ---\nTotal HERO frames saved for category '{category}': {total_hero_frames_saved}")
        print(f"All data saved in: {os.path.abspath(category_output_dir)}")

    else: 
        # --- SINGLE-FILE MODE WORKFLOW ---
        print("\nRunning in Single-File Mode.")
        video_name = os.path.basename(selected_video)
        output_base_name = os.path.splitext(video_name)[0]
        
        # Attempt to create a category folder, otherwise use the video name.
        match = re.match(r"([a-z_]+)_([a-zA-Z0-9_]+)\.", video_name, re.IGNORECASE)
        category = match.group(2) if match else output_base_name
        
        category_output_dir = os.path.join(CONFIG['OUTPUT_DIR'], category)
        final_output_dir = os.path.join(category_output_dir, output_base_name)
        os.makedirs(final_output_dir, exist_ok=True)

        print("\n--- Step 2: Select Region of Interest (ROI) ---")
        cap = cv2.VideoCapture(selected_video); ret, frame = cap.read(); cap.release()
        if not ret: print(f"ERROR: Could not read frame from {video_name}."); return
        roi = select_roi(frame, video_name)

        print("\n--- Step 3: Processing video ---")
        total_frames = analyze_single_video(selected_video, roi, final_output_dir, CONFIG, models, 0)
        
        print(f"\n\n--- Single-File Workflow Complete ---\nTotal HERO frames saved for '{video_name}': {total_frames}")
        print(f"All data saved in: {os.path.abspath(final_output_dir)}")


if __name__ == "__main__":
    main()
