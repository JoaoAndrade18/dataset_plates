import torch
import cv2
import os
import numpy as np
from ultralytics import YOLO

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
        angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -30 < angle_deg < 30:
            angles.append(angle_deg)
    if not angles:
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
                    mapa = {'0':'O','1':'I','2':'Z','4':'A','5':'S','6':'G','7':'T','8':'B','Q':'O'}
                    corr.append(mapa.get(ch, ch))
                else:
                    mapa = {'O':'0','Q':'0','D':'0','I':'1','Z':'2','A':'4','S':'5','G':'6','T':'7','B':'8'}
                    corr.append(mapa.get(ch, ch))
        return "".join(corr)

    return [aplica_padrao(padrao_antigo, text), aplica_padrao(padrao_novo, text)]

def preprocess_plate(img):
    imgGray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(imgGray.copy(), (3, 3), cv2.BORDER_DEFAULT)
    thresh = cv2.adaptiveThreshold(blurred.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 11)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erode = cv2.erode(thresh.copy(), kernel_rect, iterations=1)
    processed_image = cv2.dilate(erode.copy(), kernel_rect, iterations=1)
    processed_image = cv2.resize(processed_image, (240, 78), interpolation=cv2.INTER_NEAREST)
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    return processed_image

def load_ocr_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="OCR_novo.pt", device=0, force_reload=True)
    return model

def ocr_detect(frame, model):
    text_plate = None
    try:
        rets = []
        text_plate = ""
        plate_list = []                
        res = model(frame)
        for i in res.xyxy[0]:
            x1 = int(i[0])
            y1 = int(i[1])
            x2 = int(i[2])
            y2 = int(i[3])
            rets.append([[x1,y1,x2-x1,y2-y1], int(i[5])])
        for boxL, classId in rets:
            if boxL[2]*boxL[3] > 700:
                continue
            stop = False
            for b in plate_list:
                if abs(boxL[0]-b[0]) < 5:
                    stop = True
                    break
            if stop:
                continue
            char = class_names[classId]
            cv2.rectangle(frame, boxL, (255,0,0), 1)
            cv2.putText(frame, char, (boxL[0]+2,boxL[1]-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)   
            cv2.imshow("OCR", frame)
            cv2.waitKey(0)     
            
            if len(plate_list)==0:
                plate_list.append((boxL[0],char))
            else:
                tamList = len(plate_list)
                inserido = False
                for i in range(tamList):
                    if plate_list[i][0] > boxL[0]:
                        plate_list.insert(i,(boxL[0],char))
                        inserido = True
                        break
                if not inserido:
                    plate_list.append((boxL[0],char))
        plate_list = [x2 for x1,x2 in plate_list]
        text_plate = "".join(plate_list)
        
        return corrige_placa(text_plate)[0], frame
    
    except Exception as e:
        print("Erro OCR: ", e)
        return "", frame

class_names = []
with open("ocr-net.names", "r") as f:
	class_names = [cname.strip() for cname in f.readlines()]

ocr_model = load_ocr_model()
yolo_model = YOLO('best_placa.pt')

images = ["PODI-LPR-01/"+x for x in os.listdir("PODI-LPR-01/")]

for image_path in images:
    image = cv2.imread(image_path)
    cv2.imshow("Teste", image)
    cv2.waitKey(0)
    if image is None:
        continue

    image_name = os.path.basename(image_path)
    print(f'\nProcessing {image_name}...')

    results = yolo_model(image)
    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = image[y1:y2, x1:x2]
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)

        roi = cv2.resize(roi, (240, 78))
        h, w = roi.shape[:2]
        roi = roi[h-55:h, w-225:w-20]
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)

        roi = deskew(roi)
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)

        processed_image = preprocess_plate(roi)
        cv2.imshow("Processed", processed_image)
        cv2.waitKey(0)

        text_plate, _ = ocr_detect(roi, ocr_model)
        print(f"Espected: {image_path[12:19]}, Detected Plate (corrigida): {text_plate}")
    else:
        print(f"Nenhuma placa detectada em {image_path}")

# --- Avaliação ---
def count_correct_chars(gt, pred):
    return sum(a == b for a, b in zip(gt, pred))

acertos_7, acertos_6, total_7 = 0, 0, 0

for image_path in images:
    image = cv2.imread(image_path)
    if image is None:
        continue

    ground_truth = os.path.basename(image_path)[:7].upper().replace('-', '').replace(' ', '')
    text_plate, _ = ocr_detect(image, ocr_model)
    if not text_plate:
        continue

    opcoes = corrige_placa(text_plate)
    total_7 += 1
    if opcoes[0] == ground_truth or opcoes[1] == ground_truth:
        acertos_7 += 1
    elif count_correct_chars(ground_truth, opcoes[0]) == 6 or count_correct_chars(ground_truth, opcoes[1]) == 6:
        acertos_6 += 1

print(f"Acurácia placas 7 caracteres: {(acertos_7/total_7)*100 if total_7 else 0:.2f}% ({acertos_7}/{total_7})")
print(f"Acurácia placas 6 caracteres: {(acertos_6/total_7)*100 if total_7 else 0:.2f}% ({acertos_6}/{total_7})")
