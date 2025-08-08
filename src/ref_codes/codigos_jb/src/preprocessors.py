import cv2
import numpy as np
import os
from utils import deskew  # Certifique-se que essa função esteja corretamente implementada

# Carregamento do modelo de super-resolução
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr_model_path = os.path.join("../../ESPCN_x3.pb")  
sr.readModel(sr_model_path)
sr.setModel("espcn", 3)

def remove_borders(img, border_threshold=10):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    if w > border_threshold and h > border_threshold:
        return img[y:y+h, x:x+w]
    return img

def preprocess_pipeline(image):
    try:
        image = sr.upsample(image)
    except Exception as e:
        print(f"[ERRO] Super-resolução falhou: {e}")

    # Remoção de bordas (antes da binarização)
    image = remove_borders(image)

    # Cinza + equalização
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Ajuste de rotação (deskew)
    gray = deskew(gray)

    # Binarização adaptativa
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )

    # Dilatação para engrossar traços
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thick = cv2.dilate(binary, kernel, iterations=1)

    return thick
