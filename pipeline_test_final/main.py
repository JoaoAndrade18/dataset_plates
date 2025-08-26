import os
import io
import cv2
import time
import csv
import json
import glob
import base64
import typing as T


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -- load datasframes --
def open_datasets(path_alpr: str, gold_csv: str,  image_ext=".png", txt_ext=".txt"):
    df = pd.read_csv(gold_csv)
    df_gold = df[['ID_image', 'plate_car']]

    data = []
    for img_path in glob.glob(os.path.join(path_alpr, "**", f"*{image_ext}"), recursive=True):
        base, _ = os.path.splitext(img_path)
        txt_path = base + txt_ext
        plate_value = None
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().lower().startswith("plate:"):
                        plate_value = line.split(":", 1)[1].strip()
                        break
        data.append({"image": img_path, "plate": plate_value})

    df_alpr = pd.DataFrame(data)

    return df_gold, df_alpr

def load_models(models: list[str]):
    if 'paddleOCR' in models:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            use_textline_orientation=False,
            use_doc_unwarping=False,
            device='cpu',
            lang='en',
            text_detection_model_dir=None,
            text_det_box_thresh=0.2,
            text_recognition_model_dir=None
        )

    return ocr

def execute_ocr(models):
    # -- paddle --
    result = ocr.predict(image)
    textos = result[0]['rec_texts']