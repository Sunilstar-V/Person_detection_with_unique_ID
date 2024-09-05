from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Load the YOLOv8 model
model = YOLO("yolov8m.pt")

# Process the video file
input_video = "C:\\Users\\sunil\\Downloads\\Group_Therapy.mp4"  # Change this to the path of your input video
result = model(input_video)

person_detection = []

# Extract person detections (class ID 0 represents a person in YOLOv8)
for res in result:
    for obj in res.boxes.data.tolist():
        if obj[5] == 0:  # YOLO class ID for 'person'
            person_detection.append(obj)

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path = "final_vid.mp4"
out = None

for res in result:
    # Initialize output video writer if not done yet
    if out is None:
        out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (res.orig_img.shape[1], res.orig_img.shape[0]))

    detections = []
    for obj in res.boxes.data.tolist():
        if obj[5] == 0:
            # Get bounding box coordinates and confidence score
            x1, y1, x2, y2 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            detections.append(([x1, y1, x2, y2], confidence))

    # Update the tracker with the current frame's detections
    tracked_objects = tracker.update_tracks(detections, frame=res.orig_img)

    # Draw bounding boxes and track IDs
    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)

        # Draw the bounding box and the ID on the frame
        cv2.rectangle(res.orig_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(res.orig_img, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the updated frame to the output video
    out.write(res.orig_img)

# Release the video writer
if out is not None:
    out.release()

print(f"Video saved to {output_video_path}")
