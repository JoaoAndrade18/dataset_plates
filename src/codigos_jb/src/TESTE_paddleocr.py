from paddleocr import PaddleOCR
import time
import cv2
import os

ocr = PaddleOCR(
    use_textline_orientation=False,
    use_doc_unwarping=False,
    device='cpu',
    lang='en',
    text_detection_model_dir=None, 
    text_det_box_thresh=0.2,
    text_recognition_model_dir=None
)

image_path = os.listdir('amostras_ufpr/')
images = []

for image in image_path:
    # if image.lower().endswith(('.jpg_plate.png', 'jpg_plate.jpg', 'jpg_plate.jpeg')) and image.startswith("processed_"):
    if image.lower().endswith(('.jpg', '.png', '.jpeg')) and image.startswith("processed_"):
        images.append(image)

for image in images:
    image = f'amostras_ufpr/{image}'

    start = time.time()
    result = ocr.predict(image)
    end = time.time()

    textos = result[0]['rec_texts'] 
    print(f"Resultados do OCR para {image}:", textos)

    tempo_inferencia = end - start

    print(f"Tempo de inferência: {tempo_inferencia:.4f} segundos")
        
# txt_path = 'amostras_ufpr/track0095[01].txt'
# image_path = 'amostras_ufpr/track0095[01].png'

# annotations = {'chars': []}
# with open(txt_path, 'r') as f:
#     for line in f:
#         line = line.strip()
#         if line.startswith('plate:'):
#             annotations['plate'] = line.split(':', 1)[1].strip()
#         elif line.startswith('position_plate:'):
#             parts = line.split(':', 1)[1].strip().split()
#             annotations['position_plate'] = tuple(map(int, parts))
#         elif line.startswith('char'):
#             parts = line.split(':', 1)[1].strip().split()
#             annotations['chars'].append(tuple(map(int, parts)))

# image = cv2.imread(image_path)

# print("-" * 30)
# print(f"Analisando: {image_path}")
# print(f"Placa Correta (Ground Truth): {annotations.get('plate', 'N/A')}")
# print("-" * 30)

# x, y, w, h = annotations['position_plate']
# plate_img = image[y:y+h, x:x+w]
# cv2.imshow("Placa", plate_img)
# cv2.waitKey(0)

# plate_text = ocr.predict(plate_img)
# print(f"Resultado (Placa Inteira):   {plate_text[0]['rec_texts']}")

# recognized_chars = []
# for (x_char, y_char, w_char, h_char) in annotations['chars']:
#     char_img = image[y_char:y_char+h_char, x_char:x_char+w_char]
#     char_text = predict(char_img, config_char)
#     recognized_chars.append(char_text)

# final_plate_from_chars = "".join(recognized_chars)
# print(f"Resultado (Caracteres Ind.): {final_plate_from_chars}")
# print("\n")
