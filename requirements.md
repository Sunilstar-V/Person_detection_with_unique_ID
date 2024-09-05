# Person Detection and Tracking Using YOLOv8 and DeepSORT

## Overview

This project uses YOLOv8 for person detection and DeepSORT for tracking multiple people in a video. The system assigns unique IDs to each individual and tracks them throughout the video, even if they leave the frame and re-enter or become partially occluded. The goal is to distinguish between different individuals (such as children and therapists) and continuously track them in long-duration videos.

## Features
- **Person Detection**: Detects all persons in the video using YOLOv8.
- **Unique ID Assignment**: Assigns a unique ID to each person detected.
- **Re-Entry Tracking**: Tracks individuals if they leave the frame and re-enter.
- **Post-Occlusion Tracking**: Continues to track individuals after occlusion or partial visibility.
- **Video Output**: Saves the results with bounding boxes and unique IDs overlaid on the video.

## Setup and Installation

## Prerequisites
- Python 3.8 or higher
- Pycharm (or any IDE)

### Install Required Libraries
To set up the environment, you will need to install the following dependencies:

```bash
pip install ultralytics
pip install depp_sort_realtime
pip install opencv-python
```

## Model and Video
- Download the YOLOv8 model from the [Ultralytics](https://docs.ultralytics.com/usage/python/#predict)
- Use any video file where you want to detect and track people.

## Folder Structure
Ensure that the following files are available in your project directory:

```bash
project/
│
├── detect_and_track.py  # Main script for detection and tracking
├── Group_Therapy.mp4    # Input video
├── yolov8m.pt           # Pre-trained YOLOv8 model
└── README.md            # This file
```

## How to Run the Code

1. Clone this repository or download the necessary files.
2. Open the project directory in Pycharm.
3. Run the detection and tracking script using the following command:

```bash
python detect_and_track.py
```

4. After the script finishes, the output video with bounding boxes and unique IDs will be saved in the project folder as bash``` final_vid.mp4```

## Code Explanation
The detection and tracking pipeline is built using the following steps:
- **YOLOv8 for Person Detection**: The YOLOv8 model is used to detect all objects in the video. We filter the detections to keep only those related to persons.
- **DeepSORT for Tracking**: DeepSORT tracks individuals across frames by assigning unique IDs. It handles tracking even if people go out of the frame and return or if they become occluded.
- **Video Processing**: OpenCV is used to read the input video frame by frame, overlay the bounding boxes and IDs, and save the final video output.

### Key Parts of the Code:
```bash
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Load YOLOv8 model
model = YOLO("yolov8m.pt")
result = model("Group_Therapy.mp4")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Process each frame of the video
for res in result:
    detections = []
    for obj in res.boxes.data.tolist():
        if obj[5] == 0:  # Class ID 0 corresponds to persons
            x1, y1, x2, y2 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            detections.append(([x1, y1, x2, y2], confidence))

    # Update tracker with detections
    tracked_objects = tracker.update_tracks(detections, frame=res.orig_img)

    # Overlay bounding boxes and IDs on the frame
    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(res.orig_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(res.orig_img, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('final_vid.mp4', fourcc, 30.0, (res.orig_img.shape[1], res.orig_img.shape[0]))
for frame in result:
    out.write(frame.orig_img)
out.release()
```

### Results
The output will be a video (final_vid.mp4) with bounding boxes drawn around detected persons and a unique ID displayed above each person.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

```csharp

This is now properly formatted for a `README.md` file in Markdown format. You can copy this code directly into your `README.md` file in GitHub.
```