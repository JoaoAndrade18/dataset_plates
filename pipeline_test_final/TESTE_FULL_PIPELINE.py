import matplotlib.pyplot as plt
import pytesseract
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import os
import glob
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# -- modelo YOLO --
yolo_model = YOLO('best_placa.pt').to(device='cuda') 
yolo_model.info()

# -- PaddleOCR --
paddle_ocr = PaddleOCR(
    use_textline_orientation=False,
    use_doc_unwarping=False,
    device='cpu',
    lang='en',
    # text_detection_model_dir=None, 
    # text_det_box_thresh=0.2,
    # text_recognition_model_dir=None
)

def plt_show(image, title='', gray=False):
    temp = image.copy()
    if not gray:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    plt.title(title)
    plt.imshow(temp, cmap='gray' if gray else None)
    plt.show()

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
    """
    Corrige a inclinação da imagem automaticamente.
    """
    angle = compute_skew(image)
    return rotate_image(image, angle)

def corrige_placa(text):
    # Limpa o texto OCR, removendo caracteres inválidos e mantendo apenas alfanuméricos
    text = ''.join(c for c in text.upper() if c.isalnum() and c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    
    # Se mais de 7 caracteres, pega os primeiros 7
    if len(text) > 7:
        text = text[:7]
    # Se menos de 7, retorna como está para ambos os padrões
    if len(text) < 7:
        return [text, text]

    # Definir padrões
    def is_letra(ch): return ch.isalpha()
    def is_num(ch): return ch.isdigit()

    padrao_antigo = [is_letra, is_letra, is_letra, is_num, is_num, is_num, is_num]  # LLLNNNN
    padrao_novo   = [is_letra, is_letra, is_letra, is_num, is_letra, is_num, is_num]  # LLLNLNN

    def aplica_padrao(padrao, chars):
        corr = []
        for i, (ch, f) in enumerate(zip(chars, padrao)):
            if f(ch):
                corr.append(ch)
            else:
                if f == is_letra:
                    corr.append(corrige_para_letra(ch))
                else:
                    corr.append(corrige_para_num(ch))
        return "".join(corr)

    def corrige_para_letra(ch):
        mapa = {'0':'O', '1':'I', '2':'Z', '4':'A', '5':'S', '6':'G', '7':'T', '8':'B', 'Q':'O'}
        return mapa.get(ch, ch)

    def corrige_para_num(ch):
        mapa = {'O':'0', 'Q':'0', 'D':'0', 'I':'1', 'Z':'2', 'A':'4', 'S':'5', 'G':'6', 'T':'7', 'B':'8'}
        return mapa.get(ch, ch)

    placa_antiga = aplica_padrao(padrao_antigo, text)
    placa_nova = aplica_padrao(padrao_novo, text)

    return [placa_antiga, placa_nova]

def detect_and_extract_plate(img, upscale_factor=2):
    # Upscale the entire image first to improve ROI quality
    height, width = img.shape[:2]
    new_width = int(width * upscale_factor)
    new_height = int(height * upscale_factor)
    img_upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Run YOLO on the upscaled image
    results = yolo_model(img_upscaled)
    print('YOLO Number plate detected:', len(results[0].boxes))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # -- recorta ROI diretamente no frame upscaled --
            roi = img_upscaled[y1:y2, x1:x2]
            roi = cv2.resize(roi, (240, 78))
            h, w = roi.shape[:2]
            roi = roi[h - 55:h, w - 225:w-20]

            # -- corrige inclinação --
            roi = deskew(roi)

            # -- pré-processamento para OCR --
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 19, 9
            )

            # ROI sem preprocessamento adicional (apenas gray para OCR)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Converter imagens para 3 canais (RGB) para PaddleOCR
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            gray_roi_rgb = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2RGB)

            # plt_show(img_upscaled, "Detecção YOLO (upscaled)")
            # plt_show(roi, "ROI corrigida (placa)")
            # plt_show(processed, "ROI binarizada", gray=True)

            return roi, processed_rgb, gray_roi_rgb

    return None, None, None

def process_image_with_ocr(image_path, expected_plate):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao carregar imagem: {image_path}")
        return None

    roi, processed, gray_roi = detect_and_extract_plate(img)
    if roi is None:
        print(f"Nenhuma placa encontrada em: {image_path}")
        return {
            'image_name': os.path.basename(image_path),
            'expected': expected_plate,
            # Com preprocessamento
            'tesseract_pre_raw': '',
            'tesseract_pre_antiga': '',
            'tesseract_pre_nova': '',
            'paddle_pre_raw': '',
            'paddle_pre_antiga': '',
            'paddle_pre_nova': '',
            # Sem preprocessamento
            'tesseract_no_pre_raw': '',
            'tesseract_no_pre_antiga': '',
            'tesseract_no_pre_nova': '',
            'paddle_no_pre_raw': '',
            'paddle_no_pre_antiga': '',
            'paddle_no_pre_nova': ''
        }

    # Tesseract com preprocessamento
    tesseract_pre_text = pytesseract.image_to_string(
        processed, lang='eng',
        config='--psm 11 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    ).strip()
    tesseract_pre_options = corrige_placa(tesseract_pre_text)

    # PaddleOCR com preprocessamento
    try:
        paddle_pre_result = paddle_ocr.predict(processed)
        paddle_pre_text = ''
        if paddle_pre_result and paddle_pre_result[0]:
            paddle_pre_text = ' '.join([line[1][0] for line in paddle_pre_result[0]]).strip().replace(' ', '')
    except Exception as e:
        print(f"Erro no PaddleOCR (com preprocessamento) para {image_path}: {e}")
        paddle_pre_text = ''
    paddle_pre_options = corrige_placa(paddle_pre_text)

    # Tesseract sem preprocessamento (direto no gray_roi)
    tesseract_no_pre_text = pytesseract.image_to_string(
        gray_roi, lang='eng',
        config='--psm 11 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    ).strip()
    tesseract_no_pre_options = corrige_placa(tesseract_no_pre_text)

    print(f"Tesseract (pre): {tesseract_pre_text} | Tesseract (no pre): {tesseract_no_pre_text}")

    # PaddleOCR sem preprocessamento (direto no gray_roi)
    try:
        paddle_no_pre_result = paddle_ocr.predict(gray_roi)
        paddle_no_pre_text = ''
        if paddle_no_pre_result and paddle_no_pre_result[0]:
            paddle_no_pre_text = ' '.join([line[1][0] for line in paddle_no_pre_result[0]]).strip().replace(' ', '')
    except Exception as e:
        print(f"Erro no PaddleOCR (sem preprocessamento) para {image_path}: {e}")
        paddle_no_pre_text = ''
    paddle_no_pre_options = corrige_placa(paddle_no_pre_text)

    print(f"PaddleOCR (pre): {paddle_pre_text} | PaddleOCR (no pre): {paddle_no_pre_text}")

    return {
        'image_name': os.path.basename(image_path),
        'expected': expected_plate,
        # Com preprocessamento
        'tesseract_pre_raw': tesseract_pre_text,
        'tesseract_pre_antiga': tesseract_pre_options[0],
        'tesseract_pre_nova': tesseract_pre_options[1],
        'paddle_pre_raw': paddle_pre_text,
        'paddle_pre_antiga': paddle_pre_options[0],
        'paddle_pre_nova': paddle_pre_options[1],
        # Sem preprocessamento
        'tesseract_no_pre_raw': tesseract_no_pre_text,
        'tesseract_no_pre_antiga': tesseract_no_pre_options[0],
        'tesseract_no_pre_nova': tesseract_no_pre_options[1],
        'paddle_no_pre_raw': paddle_no_pre_text,
        'paddle_no_pre_antiga': paddle_no_pre_options[0],
        'paddle_no_pre_nova': paddle_no_pre_options[1]
    }

# Configurações
folder_path = 'PODI-LPR-01'  # Substitua pelo caminho da pasta com as imagens
output_csv = 'ocr_results.csv'

# Lista de imagens (assumindo extensões .jpg, .png, etc.)
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(folder_path, ext)))

results = []
for image_path in image_files:
    filename = os.path.basename(image_path)
    expected_plate = filename[:7].upper()  # Primeiros 7 caracteres como placa esperada
    print(f"Processando: {filename} (esperado: {expected_plate})")
    
    result = process_image_with_ocr(image_path, expected_plate)
    if result:
        results.append(result)

# Salvar em CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Resultados salvos em: {output_csv}")

# Exemplo de visualização (opcional)
print(df.head())