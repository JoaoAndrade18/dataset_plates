import matplotlib.pyplot as plt
import pytesseract
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

ocr = PaddleOCR(
    use_textline_orientation=False,
    use_doc_unwarping=False,
    device='gpu',
    lang='en',
    text_detection_model_dir=None, 
    text_det_box_thresh=0.2,
    text_recognition_model_dir=None
)

# -- modelo YOLO --
yolo_model = YOLO('best_placa.pt').to(device='cuda') 

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
            image = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 19, 9
            )

            plt_show(img_upscaled, "Detecção YOLO (upscaled)")
            plt_show(roi, "ROI corrigida (placa)")
            plt_show(image, "ROI binarizada", gray=True)

            return roi, image

    return None, None

# img = cv2.imread('frames_ufpr\\track0149\\track0149[01].png')

img = cv2.imread('PODI-LPR-01/QPW1617_2019-03-29_09-01-17_91.90.jpg')
roi, processed = detect_and_extract_plate(img)

if roi is not None:
    # OCR usando image_to_string
    # text = pytesseract.image_to_string(
    #     processed, lang='eng',
    #     config='--psm 11 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    # )
    result = ocr.predict(processed)
    text = result[0]['rec_texts'] 

    print("Texto bruto OCR:", text.strip())

    # Aplicar correções para ambos os padrões
    opcoes = corrige_placa(text)
    print("Opção placa antiga (LLLNNNN):", opcoes[0])
    print("Opção placa nova (LLLNLNN):", opcoes[1])
else:
    print("Nenhuma placa encontrada.")