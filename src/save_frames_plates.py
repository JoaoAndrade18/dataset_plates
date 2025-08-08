import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from typing import List, Dict

class VehiclePlateProcessor:
    def __init__(self, video_path: str, output_dir: str, vehicle_model_path: str, frame_interval: int = 5):
        self.video_path = video_path
        self.output_dir = output_dir
        self.vehicle_model = YOLO(vehicle_model_path)
        self.frame_interval = frame_interval
        self.frame_count = 1
        self.cut_frame_count = 1
        self.data = []

        self.frames_dir = os.path.join(output_dir, "frames")
        self.cuts_dir = os.path.join(output_dir, "cut_frame")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.cuts_dir, exist_ok=True)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % self.frame_interval != 0:
                frame_id += 1
                continue

            detections = self.detect_vehicles(frame)

            if detections:
                frame_name = f"frame_{self.frame_count}.jpg"
                cv2.imwrite(os.path.join(self.frames_dir, frame_name), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    label = det['type']

                    cut_crop = frame[y1:y2, x1:x2]
                    cut_name = f"cut_frame_{self.cut_frame_count}.jpg"
                    cv2.imwrite(os.path.join(self.cuts_dir, cut_name), cut_crop, [cv2.IMWRITE_JPEG_QUALITY, 100])

                    self.data.append({
                        "ID_frame": self.frame_name,
                        "ID_cut_frame": self.cut_name,
                        "Type": label
                    })

                    self.cut_frame_count += 1

                self.frame_count += 1

            frame_id += 1

        cap.release()
        self.save_csv()

    def detect_vehicles(self, frame: np.ndarray) -> List[Dict]:
        results = self.vehicle_model(frame)
        detections = []

        h, w, _ = frame.shape
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls)
                label = self.vehicle_model.names[cls_id]

                if label not in ['car', 'motorcycle', 'truck', 'bus']:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_height = y2 - y1
                box_area = (x2 - x1) * box_height
                frame_area = w * h

                if box_height > 0.2 * h or box_area > 0.05 * frame_area:
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'type': label
                    })

        return detections

    def save_csv(self):
        df = pd.DataFrame(self.data)
        df.to_csv(os.path.join(self.output_dir, "vehicles_info.csv"), index=False)

processor = VehiclePlateProcessor(
    video_path="/content/drive/MyDrive/MESTRADO/dataset_plates/SimPlay-Grand Village-Portaria-CAM6.mp4",
    output_dir="/content/drive/MyDrive/MESTRADO/dataset_plates/results",
    vehicle_model_path="yolov8n.pt",
    frame_interval=5
)
processor.process_video()