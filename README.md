# Conveyor Belt Frame Extractor for Annotation

## Project Overview

This project was developed to streamline the process of creating image datasets for object annotation. To capture objects in all possible orientations for machine learning models, they were recorded on a moving conveyor belt from five different angles (top, front, back, left, and right).

This script automates the process of extracting relevant frames from video files. It uses a two-stage AI pipeline to identify the best shots of objects, preparing a dataset that is ready for annotation software like AnyLabeling.

## Key Features

* **Dual AI Pipeline:** Utilizes a fast YOLO detector for initial screening and a high-accuracy YOLO tracker for precise object tracking and scoring, ensuring both speed and quality.

* **Multi-Angle Batch Mode:** Select one video, and the script will automatically find and process the other corresponding angle videos based on a shared category name.

* **Flexible Single-File Mode:** Easily switch to process only a single video file when needed.

* **Intelligent Frame Selection:** Identifies the single best "hero" frame for each object pass based on its stability and centeredness within a defined area.

* **Burst Frame Separation:** Optionally saves the frames immediately before and after the hero shot into a separate `burst_frames` subfolder for additional context without cluttering the main dataset.

* **Customizable Naming Conventions:** Choose between simple sequential numbering (e.g., `top_view_fmcg_0.jpg`, perfect for annotation software) or descriptive filenames for easier debugging.

* **Interactive ROI Selection:** A user-friendly GUI allows you to draw the precise area of the conveyor belt to analyze for each video.

## Setup and Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/dhavalp16/Conveyor-Belt-Frame-Extractor.git
   cd Conveyor-Belt-Frame-Extractor
   ```

2. **Create a Python Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO Models:** The script uses YOLOv8 models. They will be downloaded automatically by the `ultralytics` library on the first run.

## Directory Structure

Before running, make sure your project has the following folder structure. The `input_videos` folder is where you should place all your raw video files.

```
CONVEYOR_ANNOTATION_PROJECT/
|
|-- input_videos/
|   |-- top_view_fmcg.mp4
|   |-- front_view_fmcg.mp4
|   |-- back_view_fmcg.mp4
|   |-- side_left_view_fmcg.mp4
|   |-- side_right_view_fmcg.mp4
|   |-- ... (other categories)
|
|-- .gitignore
|-- app.py
|-- requirements.txt
```

## How to Use

1. **Place Videos:** Add your video files to the `input_videos` folder. Ensure they follow the naming convention: `angle_category.mp4` (e.g., `top_view_biscuit.mp4`).

2. **Configure `app.py`:** Open `app.py` and adjust the settings in the `CONFIG` dictionary at the top of the file to match your needs.

   **Primary Settings:**

   * `PROCESS_MULTIPLE_ANGLES`: `True` for batch mode, `False` for single-file mode.

   * `ANGLES_TO_PROCESS`: A list of the angle prefixes you want to process in batch mode.

   * `USE_DESCRIPTIVE_FILENAMES`: `False` for sequential numbering, `True` for detailed names.

   * `FRAME_BURST_COUNT`: Set to `0` to disable burst saving, or to a number like `5` to save 5 frames before and after the hero shot.

   **AI & Detection Logic Settings:**

   * `BATCH_SIZE`: The number of frames to process on the GPU at once. Lower this value if you run into VRAM issues.

   * `CONFIDENCE_THRESHOLD`: The minimum confidence score for the AI to consider an object as a valid detection.

   * `MIN_CENTEREDNESS_SCORE`: A value from 0.0 to 1.0. The "best" frame for an object must have a centeredness score above this threshold to be saved.

   * `STABILITY_FRAMES`: An object is considered "stable" only after it has been detected away from the edges of the ROI for this many consecutive frames.

   * `EDGE_THRESHOLD`: A percentage (e.g., 0.02 for 2%) that defines a "safe zone" away from the ROI border. Detections outside this zone are considered unstable.

   * `EVENT_COOLDOWN_FRAMES`: The number of frames an object must be gone before the script considers its pass complete and finalizes the best frame.

3. **Run the Script:**

   ```bash
   python app.py
   ```

4. **Follow On-Screen Instructions:**

   * A file dialog will prompt you to select one of your videos.

   * An interactive window will appear for each video, allowing you to draw the Region of Interest (ROI).

## Output

The script will save the processed frames in the `final_frames_dataset` directory, organized by category and then by angle.

```
final_frames_dataset/
|
|-- fmcg/
|   |
|   |-- top_view_fmcg/
|   |   |-- top_view_fmcg_0.jpg
|   |   |-- top_view_fmcg_1.jpg
|   |   |-- burst_frames/
|   |       |-- top_view_fmcg_0_0.jpg
|   |       |-- top_view_fmcg_0_1.jpg
|   |
|   |-- front_view_fmcg/
|       |-- front_view_fmcg_2.jpg
|       |-- ...
