from ultralytics import YOLO
import cv2
import numpy as np

def predict(image: np.ndarray) -> list:
        model = YOLO("../../../Charcter-LP.pt")
        results = model(image)[0]
        boxes = results.boxes.cpu().numpy()  # Get bb
        detections = []

        if boxes.shape[0] == 0:
            return []

        for box in boxes:
            if box.cls is None:
                continue

            label = results.names[int(box.cls[0])]

            detections.append(label)

        return detections   

# open image
image = cv2.imread("frames_gold/frame_14_processed.jpg_plate_processed.jpg")

print(predict(image))