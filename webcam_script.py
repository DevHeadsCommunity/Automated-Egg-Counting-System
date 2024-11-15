# webcam_capture.py
import cv2
import torch
from src.utils import preprocess_frame, count_eggs, load_model

def capture_from_webcam(model_path="model/my_model.pth"):
    # Load the model
    model = load_model(model_path)
    print("Model loaded. Starting webcam...")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame")
            break
        
        # Preprocess the frame
        processed_frame = preprocess_frame(frame)
        
        # Perform inference
        egg_count = count_eggs(model, processed_frame)
        
        # Display results
        cv2.putText(frame, f"Eggs: {egg_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Egg Counter", frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_from_webcam()
