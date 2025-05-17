from ultralytics import YOLO
import cv2
import numpy as np

# Load the custom trained model
model = YOLO('fod50.pt')

# Initialize the video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run inference on the frame
        results = model(frame, conf=0.8)  # confidence threshold set to 0.8

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("FOD Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
