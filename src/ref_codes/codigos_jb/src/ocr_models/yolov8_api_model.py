from ultralytics import YOLO
import cv2
import numpy as np

class OCRYOLO:
    def __init__(self, endpoint: str = "../../../Charcter-LP.pt"):
        self.__model = YOLO(endpoint)

    def predict(self, image: np.ndarray) -> list:
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        results = self.__model(image)[0]
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
