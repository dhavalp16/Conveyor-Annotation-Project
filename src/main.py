# main.py
#
# This is the main entry point for the multi-angle video processing project.
# It orchestrates the frame extraction process by scanning for video files
# in the structured 'input_videos' directory and calling the processing
# function for each one.

import os
import sys
from glob import glob

# --- Main Logic ---

def main():
    """
    Main function to orchestrate the video processing for all camera angles.
    """
    # This gets the absolute path of the directory containing this script (src/)
    # then goes up one level to get the project's root directory.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    input_base_path = os.path.join(project_root, 'input_videos')
    output_base_path = os.path.join(project_root, 'output_frames')

    # Ensure the base input and output directories exist
    if not os.path.isdir(input_base_path):
        print(f"ERROR: Input directory not found at '{input_base_path}'")
        print("Please create it and place your angle folders (e.g., 'top', 'front') inside.")
        return
        
    if not os.path.isdir(output_base_path):
        print(f"INFO: Output directory not found at '{output_base_path}'. Creating it now.")
        os.makedirs(output_base_path)

    # Dynamically find all the angle directories in the input folder
    try:
        angle_folders = [d for d in os.listdir(input_base_path) if os.path.isdir(os.path.join(input_base_path, d))]
    except FileNotFoundError:
        print(f"ERROR: Could not read directories from '{input_base_path}'.")
        return

    if not angle_folders:
        print(f"WARNING: No angle folders found inside '{input_base_path}'. Nothing to process.")
        return

    print(f"Found {len(angle_folders)} angle(s) to process: {', '.join(angle_folders)}")
    print("="*60)

    # --- Import the frame extractor logic and config ---
    # We do this here to ensure the path is set up correctly
    try:
        from frame_extractor import analyze_video_yolo, CONFIG
    except ImportError:
        print("ERROR: Could not import from 'frame_extractor.py'.")
        print("Please ensure both 'main.py' and 'frame_extractor.py' are in the 'src/' directory.")
        return

    # Loop through each angle folder and process its videos
    for angle in angle_folders:
        print(f"\nProcessing angle: '{angle}'")
        print("-" * 30)
        
        input_angle_dir = os.path.join(input_base_path, angle)
        output_angle_dir = os.path.join(output_base_path, angle)
        
        # Create the corresponding output directory if it doesn't exist
        if not os.path.exists(output_angle_dir):
            os.makedirs(output_angle_dir)
            
        # Find all video files in the current angle directory
        # This searches for .mp4 and .avi files. Add other extensions if needed.
        video_files = glob(os.path.join(input_angle_dir, '*.mp4')) + \
                      glob(os.path.join(input_angle_dir, '*.avi'))

        if not video_files:
            print(f"  - No video files (.mp4, .avi) found in '{input_angle_dir}'. Skipping.")
            continue

        # Process each video found for the current angle
        for video_path in video_files:
            video_name = os.path.basename(video_path)
            print(f"  -> Starting analysis for video: '{video_name}'")
            
            # The output directory for this specific video's frames will be named
            # after the video file itself (without the extension) for clarity.
            video_output_folder_name = os.path.splitext(video_name)[0]
            final_output_path = os.path.join(output_angle_dir, video_output_folder_name)

            # Override the default VIDEO_PATH and OUTPUT_DIR in a copy of the config
            current_config = CONFIG.copy()
            current_config["VIDEO_PATH"] = video_path
            current_config["OUTPUT_DIR"] = final_output_path

            try:
                # Call the main analysis function from the other script, passing the config
                analyze_video_yolo(video_path, final_output_path, current_config)
                print(f"  -> Finished analysis for '{video_name}'.")
            except Exception as e:
                print(f"  -> An unexpected error occurred while processing '{video_name}': {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "="*60)
    print("All processing complete.")


if __name__ == "__main__":
    main()
