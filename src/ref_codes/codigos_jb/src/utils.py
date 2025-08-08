import ast
import numpy as np
import cv2

def parse_bbox(bbox_str):
    return tuple(map(int, ast.literal_eval(bbox_str)))

import cv2
import numpy as np
import math

def rotate_image(image, angle):
    """
    Rotaciona a imagem em torno de seu centro pelo ângulo especificado.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def compute_skew(image):
    """
    Calcula o ângulo de inclinação (skew) da imagem com base em detecção de bordas e linhas.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Filtro para reduzir ruído
    blur = cv2.medianBlur(gray, 3)

    # Detecção de bordas
    edges = cv2.Canny(blur, threshold1=50, threshold2=150, apertureSize=3, L2gradient=True)

    # Detecção de linhas
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=gray.shape[1] // 2, maxLineGap=20)

    if lines is None:
        return 0.0  # Nenhuma linha encontrada, assume sem inclinação

    angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = angle_rad * 180.0 / np.pi

        # Filtra ângulos muito inclinados
        if -30 < angle_deg < 30:
            angles.append(angle_deg)

    if len(angles) == 0:
        return 0.0

    # Retorna a média dos ângulos encontrados
    return np.mean(angles)

def deskew(image):
    """
    Corrige a inclinação da imagem automaticamente.
    """
    angle = compute_skew(image)
    return rotate_image(image, angle)


def calculate_accuracy(predictions):
    correct = 0
    for pred, true in predictions:
        if pred == true:
            correct += 1
    return correct / len(predictions)
