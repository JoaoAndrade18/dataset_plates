import pandas as pd
import cv2
import time
import os
from preprocessors import preprocess_pipeline
from ocr_models.tesseract_model import OCRTesseract
from ocr_models.yolov8_api_model import OCRYOLO
from utils import parse_bbox, calculate_accuracy

class InferPlate:
    def __init__(self, csv_path: str, image_dir: str, models: str, device: str, batch_size=16):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.output_dir = "original_plate_images/" # temp
        self.device = device
        self.batch_size = batch_size

        self.models = {
            "tesseract": OCRTesseract(),
            "yolo": OCRYOLO()
        }

        self.selected_models = {name: self.models[name] for name in models}

    def run_evaluation(self):
        results = {model: [] for model in self.selected_models}
        times = {model: [] for model in self.selected_models}

        aux = 0
        for _, row in self.df.iterrows():
            if aux != 0:
                break
            aux += 1
            img_path = os.path.join(self.image_dir, row["ID_image"])

            print("[INFO] Processing image:", img_path)

            img = cv2.imread(img_path)
            if img is None:
                continue

            bbox = parse_bbox(row["bbox_plate"])

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            plate_crop = img[y1:y2, x1:x2]

            processed_plate = preprocess_pipeline(plate_crop)

            cv2.imshow("[IMAGE] after preprocessing", processed_plate)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            for name, model in self.selected_models.items():
                start = time.time()
                ocr_text = model.predict(processed_plate)
                end = time.time()

                print(f"[{name}] OCR Result: {ocr_text} | Expected: {row['plate_car']}")

                times[name].append(end - start)
                #print("[INFO]", ocr_text, row["plate_car"])
                results[name].append((ocr_text, row["plate_car"]))

        # Metrics
        #for name in self.selected_models:
        #    accuracy = calculate_accuracy(results[name])
        #    avg_time = sum(times[name]) / len(times[name])
        #    print(f"[{name}] Acurácia: {accuracy:.2%} | Tempo médio: {avg_time:.3f}s")
