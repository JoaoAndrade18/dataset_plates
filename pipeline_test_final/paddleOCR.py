from paddleocr import PaddleOCR
import time
from ultralytics import YOLO
import cv2
import os
import numpy as np

def clean_plate_text(text):
    return text.upper().replace('-', '').replace(' ', '')

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

ocr = PaddleOCR(
    use_textline_orientation=False,
    use_doc_unwarping=False,
    device='cpu',
    lang='en',
    # text_detection_model_dir=None,
    # text_det_box_thresh=0.2,
    # text_recognition_model_dir=None
)
image_list = os.listdir('PODI-LPR-01')
yolo_model = YOLO('best_placa.pt')
correct_predictions = 0
total_images = len(image_list)
acertos_6 = 0
total_6 = 0
acertos_7 = 0
total_7 = 0

def preprocess_plate(img, use_erode=True, use_dilate=True):
    img = deskew(img)
   
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if use_erode:
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.erode(img, kernel_rect, iterations=1)
    if use_dilate:
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.dilate(img, kernel_rect, iterations=1)
  
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def count_correct_chars(gt, pred):
    return sum(a == b for a, b in zip(gt, pred))

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
   
        roi = preprocess_plate(roi, use_erode=True, use_dilate=True)
       
        start = time.time()
        result = ocr.predict(roi)
        end = time.time()
        
        if result and 'rec_texts' in result[0] and result[0]['rec_texts']:
            recognized_text = result[0]['rec_texts'][0]
        
        print(f"Resultados do OCR: {recognized_text}")
        tempo_inferencia = end - start
        print(f"Tempo de inferência: {tempo_inferencia:.4f} segundos")
    else:
        print("No license plate detected by YOLO. OCR skipped.")

    opcoes = corrige_placa(recognized_text)
    ground_truth_clean = clean_plate_text(ground_truth)

    # Contagem de acertos para 7 caracteres (match total)
    if len(ground_truth_clean) == 7:
        total_7 += 1
        if opcoes[0] == ground_truth_clean or opcoes[1] == ground_truth_clean:
            acertos_7 += 1
        else:
            # Contagem de acertos de 6 caracteres
            if count_correct_chars(ground_truth_clean, opcoes[0]) == 6 or count_correct_chars(ground_truth_clean, opcoes[1]) == 6:
                acertos_6 += 1

print("\n" + "="*30)
print("--- RELATÓRIO DE ACURÁCIA")
print("="*30)
print(f"Total de imagens processadas: {total_images}")
accuracy_percentage = (correct_predictions / total_images) * 100 if total_images > 0 else 0
print(f"Acurácia para placas de 6 caracteres corretos: {(acertos_6/total_7)*100 if total_7 else 0:.2f}% ({acertos_6}/{total_7})")
print(f"Acurácia para placas de 7 caracteres corretos: {(acertos_7/total_7)*100 if total_7 else 0:.2f}% ({acertos_7}/{total_7})")
print("="*30)