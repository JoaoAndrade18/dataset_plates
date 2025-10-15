import time
import pytesseract
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

yolo_model = YOLO('best_placa.pt').to(device='cuda')

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def compute_skew(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blur, threshold1=100, threshold2=150, apertureSize=5, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=gray.shape[1] // 2, maxLineGap=30)
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = angle_rad * 180.0 / np.pi
        if -30 < angle_deg < 30:
            angles.append(angle_deg)
    if len(angles) == 0:
        return 0.0
    return np.mean(angles)

def deskew(image):
    angle = compute_skew(image)
    return rotate_image(image, angle)

def clean_plate_text(text):
    return text.upper().replace('-', '').replace(' ', '')

def corrige_placa(text):
    text = ''.join(c for c in text.upper() if c.isalnum())
    if len(text) < 7:
        return [text, text]
    def is_letra(ch): return ch.isalpha()
    def is_num(ch): return ch.isdigit()
    padrao_antigo = [is_letra, is_letra, is_letra, is_num, is_num, is_num, is_num]
    padrao_novo = [is_letra, is_letra, is_letra, is_num, is_letra, is_num, is_num]
    def aplica_padrao(padrao, chars):
        corr = []
        for i, (ch, f) in enumerate(zip(chars, padrao)):
            if f(ch):
                corr.append(ch)
            else:
                if f == is_letra:
                    mapa = {'0':'O', '1':'I', '2':'Z', '4':'A', '5':'S', '6':'G', '7':'T', '8':'B', 'Q':'O'}
                    corr.append(mapa.get(ch, ch))
                else:
                    mapa = {'O':'0', 'Q':'0', 'D':'0', 'I':'1', 'Z':'2', 'A':'4', 'S':'5', 'G':'6', 'T':'7', 'B':'8'}
                    corr.append(mapa.get(ch, ch))
        return "".join(corr)
    placa_antiga = aplica_padrao(padrao_antigo, text)
    placa_nova = aplica_padrao(padrao_novo, text)
    return [placa_antiga, placa_nova]

image_list = os.listdir('PODI-LPR-01')
correct_predictions = 0
total_images = len(image_list)

for image_name in image_list:
    ground_truth = image_name.split('_')[0]
    image_path = f'PODI-LPR-01/{image_name}'
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Could not read image at {image_path}. Skipping.")
        continue

    results = yolo_model(image)
    print(f'\nProcessing {image_name}...')
    
    recognized_text = ""
    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = image[y1:y2, x1:x2]
        roi = cv2.resize(roi, (240, 78))
        h, w = roi.shape[:2]
        roi = roi[h - 55:h, w - 225:w - 20]
        roi = deskew(roi)
        
        # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # gray = cv2.bilateralFilter(gray, 11, 17, 17)
        # processed_image = cv2.adaptiveThreshold(
        #     gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #     cv2.THRESH_BINARY, 19, 9
        # )

        imgGray = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(imgGray.copy(), (3,3), cv2.BORDER_DEFAULT)
        thresh = cv2.adaptiveThreshold(blurred.copy(), 70, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 11)
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) 
        erode = cv2.erode(thresh.copy(),kernel_rect,iterations=1)
        processed_image = cv2.dilate(erode.copy(),kernel_rect,iterations=1)

        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        
        start = time.time()
        text = pytesseract.image_to_string(processed_image, config="--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        end = time.time()
        
        recognized_text = text.strip()
        
        print(f"Resultados do OCR: {recognized_text}")
        tempo_inferencia = end - start
        print(f"Tempo de inferência: {tempo_inferencia:.4f} segundos")
    else:
        print("No license plate detected by YOLO. OCR skipped.")

    opcoes = corrige_placa(recognized_text)
    ground_truth_clean = clean_plate_text(ground_truth)
    
    if opcoes[0] == ground_truth_clean or opcoes[1] == ground_truth_clean:
        correct_predictions += 1
        print("Match! Correct prediction.")
    else:
        print(f"Mismatch. Recognized: '{recognized_text}', Opção 1: '{opcoes[0]}', Opção 2: '{opcoes[1]}', Correct: '{ground_truth}'")

print("\n" + "="*30)
print("--- RELATÓRIO DE ACURÁCIA")
print("="*30)
print(f"Total de imagens processadas: {total_images}")
print(f"Previsões corretas: {correct_predictions}")
accuracy_percentage = (correct_predictions / total_images) * 100 if total_images > 0 else 0
print(f"Acurácia do modelo OCR: {accuracy_percentage:.2f}%")
print("="*30)