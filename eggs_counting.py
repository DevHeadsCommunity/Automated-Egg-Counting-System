import torch
import cv2
import numpy as np
import pathlib
from pathlib import Path
import warnings

# Suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

pathlib.PosixPath = pathlib.WindowsPath

# Path to your YOLOv5 model (best.pt)
model_path = r"C:/Users/CLIENT/Downloads/content/yolov5/runs/train/yolov5s_results/weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Initialize webcam (0 for default camera, use 1 for secondary camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define checkpoint zone coordinates (adjust as needed for your setup)
checkpoint_x_start, checkpoint_y_start = 100, 200  # top-left corner of checkpoint
checkpoint_x_end, checkpoint_y_end = 400, 300      # bottom-right corner of checkpoint

# Initialize egg counter
egg_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run the model on the captured frame
    results = model(frame)
    result_frame = np.squeeze(results.render())  # Convert from tensor to numpy array

    # Draw the checkpoint area on the frame
    cv2.rectangle(result_frame, (checkpoint_x_start, checkpoint_y_start), (checkpoint_x_end, checkpoint_y_end), (0, 255, 0), 2)

    # Process detections and check if they fall within the checkpoint area
    detected_eggs = results.pandas().xyxy[0]  # Get bounding box results as pandas DataFrame

    for index, egg in detected_eggs.iterrows():
        # Get the coordinates of the detected egg
        x_min, y_min, x_max, y_max = int(egg['xmin']), int(egg['ymin']), int(egg['xmax']), int(egg['ymax'])
        
        # Check if the center of the detected egg is within the checkpoint zone
        egg_center_x, egg_center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        if (checkpoint_x_start <= egg_center_x <= checkpoint_x_end) and (checkpoint_y_start <= egg_center_y <= checkpoint_y_end):
            # Increment egg counter for each egg that passes the checkpoint
            egg_count += 1
            print(f"Eggs passed through checkpoint: {egg_count}")

    # Display the resulting frame
    cv2.imshow('Real-Time Detection', result_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
