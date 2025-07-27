from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image
import torch
import cv2
import numpy as np
import pytesseract
import os

feat = YolosFeatureExtractor.from_pretrained(
    "nickmuchi/yolos-small-finetuned-license-plate-detection"
)
model = YolosForObjectDetection.from_pretrained(
    "nickmuchi/yolos-small-finetuned-license-plate-detection"
)


def detectar_recortar_ocr(imagem_path):
    img = Image.open(imagem_path).convert("RGB")
    inputs = feat(images=img, return_tensors="pt")
    outputs = model(**inputs)

    # Filtrar deteções com confiança > 0.5 (ajuste conforme necessário)
    probs = outputs.logits.softmax(-1)[0, :, 1]  # classe placa índice 1
    boxes = outputs.pred_boxes[0][probs > 0.5]
    scores = probs[probs > 0.5]

    if len(boxes) == 0:
        return None, None

    # Converter bbox para coordenadas na imagem original
    W, H = img.size
    box = boxes[0].detach().cpu().numpy()
    cx, cy, w, h = box
    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)

    # Recorte
    recorte = np.array(img.crop((x1, y1, x2, y2)))

    # OCR via Tesseract
    gray = cv2.cvtColor(recorte, cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    texto = pytesseract.image_to_string(bin_img,
        config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ).strip()

    return recorte, texto