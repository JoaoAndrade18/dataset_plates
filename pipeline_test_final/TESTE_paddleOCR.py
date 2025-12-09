import cv2
import os
import time
import csv
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

RESULTS_FILE = "resultPaddleOcRODOSOL_motorcycles.csv" 
IMAGE_PATH_DIR = "RodoSol-ALPR/images/motorcycles-br/"
YOLO_MODEL_PATH = 'best_placa.pt'

def clean_plate_text(text):
    """Removes spaces and hyphens, and converts to uppercase."""
    return ''.join(c for c in text.upper() if c.isalnum())

def read_ground_truth(txt_path):
    """Reads the .txt file and extracts the ground truth plate."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('plate:'):
                    plate = line.split('plate:')[1].strip()
                    return clean_plate_text(plate)
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
    return None

def rotate_image(image, angle):
    """Rotates an image by a given angle."""
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def compute_skew(image):
    """Computes the skew angle of the license plate in the image."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    blur = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blur, threshold1=100, threshold2=150, apertureSize=5, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=gray.shape[1] // 2, maxLineGap=30)
    
    if lines is None:
        return 0.0
    
    angles = [np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi for line in lines for x1, y1, x2, y2 in line if -30 < (np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi) < 30]
    
    return np.mean(angles) if angles else 0.0

def deskew(image):
    """Corrects the skew of an image."""
    angle = compute_skew(image)
    return rotate_image(image, angle)

def preprocess_plate(img, use_erode=True, use_dilate=True):
    roi = cv2.resize(img, (240, 78))
    h, w = roi.shape[:2]
    roi = roi[h - 55:h, w - 225:w - 20]
    roi = deskew(roi)
    
    imgGray = cv2.cvtColor(roi.copy(), cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(imgGray.copy(), (3,3), cv2.BORDER_DEFAULT)
    thresh = cv2.adaptiveThreshold(blurred.copy(), 70, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 11)
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)) 
    erode = cv2.erode(thresh.copy(),kernel_rect,iterations=1)
    processed_image = cv2.dilate(erode.copy(),kernel_rect,iterations=1)
 
    return cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

def corrige_placa(text):
    text = clean_plate_text(text)
    if len(text) < 7:
        return text, text
    def is_letra(ch): return ch.isalpha()
    def is_num(ch): return ch.isdigit()
    padrao_antigo = [is_letra, is_letra, is_letra, is_num, is_num, is_num, is_num]
    padrao_novo =   [is_letra, is_letra, is_letra, is_num, is_letra, is_num, is_num]
    def aplica_padrao(padrao, chars):
        corr = []
        for ch, f in zip(chars, padrao):
            if f(ch):
                corr.append(ch)
            else:
                if f == is_letra:
                    mapa = {'0':'O', '1':'I', '2':'Z', '3':'E', '4':'A', '5':'S', '6':'G', '7':'T', '8':'B', '9':'B'}
                    corr.append(mapa.get(ch, ch))
                else:
                    mapa = {'O':'0', 'Q':'0', 'D':'0', 'I':'1', 'Z':'2', 'A':'4', 'S':'5', 'G':'6', 'T':'7', 'B':'8', 'E':'3', 'C':'6', 'Y':'4',
                            'L':'1', 'F':'5', 'H':'4', 'X':'8', 'J':'1', 'K':'6', 'M':'1', 'N':'1', 'P':'9', 'R':'2', 'U':'0', 'V':'8', 'W':'3'}
                    corr.append(mapa.get(ch, ch))
        return "".join(corr)
    
    placa_antiga = aplica_padrao(padrao_antigo, text[:7])
    placa_nova = aplica_padrao(padrao_novo, text[:7])

    return placa_antiga, placa_nova

def check_match(pred, expected):
    """Returns the number of correct characters."""
    return sum(p == t for p, t in zip(str(pred).upper(), str(expected).upper()))

ocr = PaddleOCR(
    use_textline_orientation=False,
    use_doc_unwarping=False,
    use_doc_orientation_classify=False,
    device='cpu',
    lang='en',
)
yolo_model = YOLO(YOLO_MODEL_PATH).to(device=0)

all_files = os.listdir(IMAGE_PATH_DIR)
# Filtra apenas arquivos de imagem
image_list = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

total = 0
mediumTime = 0
sevenCorrected = 0    
sevenCorrectedFixed = 0 
sixCorrected = 0      
sixCorrectedFixed = 0   
totalImagesInference = 0 

with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['image', 'true_plate', 'predicted_plate', 'q_char_corrected']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for image_name in image_list:
        # Obtém o nome base sem extensão
        base_name = os.path.splitext(image_name)[0]
        txt_name = base_name + '.txt'
        
        image_path = os.path.join(IMAGE_PATH_DIR, image_name)
        txt_path = os.path.join(IMAGE_PATH_DIR, txt_name)
        
        # Lê o ground truth do arquivo .txt
        ground_truth_plate = read_ground_truth(txt_path)
        
        if ground_truth_plate is None:
            print(f"Ground truth not found for {image_name}, skipping...")
            continue
        
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Could not read image at {image_path}. Skipping.")
            continue
        
        start_time = time.time()

        yolo_results = yolo_model(image, verbose=False)
        
        recognized_text_raw = ""
        if yolo_results and len(yolo_results[0].boxes) > 0:
            totalImagesInference += 1
            box = yolo_results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = image[y1:y2, x1:x2]

            processed_roi = preprocess_plate(roi, use_erode=True, use_dilate=True)
            
            ocr_result = ocr.predict(processed_roi)
            if ocr_result and ocr_result[0]['rec_texts']:
                recognized_text_raw = ocr_result[0]['rec_texts'][0]
            else:
                print(f"Warning: OCR returned no text for {image_name}. Skipping.")
                continue

        end_time = time.time()
        mediumTime += (end_time - start_time)
        
        recognized_text_clean = clean_plate_text(recognized_text_raw)
        corrected_old, corrected_new = corrige_placa(recognized_text_clean)

        acertos_bruto = check_match(recognized_text_clean, ground_truth_plate)
        acertos_corrigido_old = check_match(corrected_old, ground_truth_plate)
        acertos_corrigido_new = check_match(corrected_new, ground_truth_plate)

        if acertos_corrigido_new > acertos_corrigido_old:
            best_corrected_plate = corrected_new
            acertos_corrigido = acertos_corrigido_new
        else:
            best_corrected_plate = corrected_old
            acertos_corrigido = acertos_corrigido_old

        plate = ""
        acertos = 0

        if acertos_bruto == 7:
            sevenCorrected += 1
            plate = recognized_text_clean
            acertos = acertos_bruto
        elif acertos_corrigido == 7:
            sevenCorrectedFixed += 1
            plate = best_corrected_plate
            acertos = acertos_corrigido
        elif acertos_bruto == 6:
            sixCorrected += 1
            plate = recognized_text_clean
            acertos = acertos_bruto
        elif acertos_corrigido == 6:
            sixCorrectedFixed += 1
            plate = best_corrected_plate
            acertos = acertos_corrigido
        else:
            plate = recognized_text_clean
            acertos = acertos_bruto

        total += 1
        
        writer.writerow({
            'image': image_name,
            'true_plate': ground_truth_plate,
            'predicted_plate': plate,
            'q_char_corrected': acertos,
        })
        
        print(f"CORRETO: {ground_truth_plate} | PREDICT: {recognized_text_clean} | [ADJUSTED] PREDICT: {best_corrected_plate} | Acertos: {acertos}/{7}")

print("\n" + "-"*20 + " Final Results " + "-"*20)
print(f"Total Images Processed: {total}")
print(f"Total Images with Detections: {totalImagesInference}")
print("-" * 55)
print("Results based on images WITH detections:")
print(f"  Raw 7/7 Correct: {sevenCorrected} ({round(sevenCorrected / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
print(f"  Raw 6/7 Correct: {sixCorrected} ({round(sixCorrected / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
print(f"  Adjusted 7/7 Correct: {sevenCorrectedFixed} ({round(sevenCorrectedFixed / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
print(f"  Adjusted 6/7 Correct: {sixCorrectedFixed} ({round(sixCorrectedFixed / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
total_hits_inference = sevenCorrected + sixCorrected + sevenCorrectedFixed + sixCorrectedFixed
print(f"  TOTAL HITS (6 or 7): {total_hits_inference} ({round(total_hits_inference / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
print("-" * 55)
print("Results based on ALL images processed:")
print(f"  Raw 7/7 Correct: {sevenCorrected} ({round(sevenCorrected / total * 100, 2) if total else 0}%)")
print(f"  Raw 6/7 Correct: {sixCorrected} ({round(sixCorrected / total * 100, 2) if total else 0}%)")
print(f"  Adjusted 7/7 Correct: {sevenCorrectedFixed} ({round(sevenCorrectedFixed / total * 100, 2) if total else 0}%)")
print(f"  Adjusted 6/7 Correct: {sixCorrectedFixed} ({round(sixCorrectedFixed / total * 100, 2) if total else 0}%)")
total_hits_all = sevenCorrected + sixCorrected + sevenCorrectedFixed + sixCorrectedFixed
print(f"  TOTAL HITS (6 or 7): {total_hits_all} ({round(total_hits_all / total * 100, 2) if total else 0}%)")
print("-" * 55)
if total > 0:
    print(f"Average Time per Image: {round(mediumTime / total, 4)}s")
print("-" * 55)