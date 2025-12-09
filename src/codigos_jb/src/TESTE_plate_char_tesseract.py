import time
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\PC\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

txt_path = 'amostras_ufpr/track0092[01].txt'
image_path = 'amostras_ufpr/track0092[01].png'


def predict(image, config) -> str:
        """  """
        # config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(image, config=config)

        return text

annotations = {'chars': []}
with open(txt_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('plate:'):
            annotations['plate'] = line.split(':', 1)[1].strip()
        elif line.startswith('position_plate:'):
            parts = line.split(':', 1)[1].strip().split()
            annotations['position_plate'] = tuple(map(int, parts))
        elif line.startswith('char'):
            parts = line.split(':', 1)[1].strip().split()
            annotations['chars'].append(tuple(map(int, parts)))

image = cv2.imread(image_path)

config_plate = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
config_char = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

print("-" * 30)
print(f"Analisando: {image_path}")
print(f"Placa Correta (Ground Truth): {annotations.get('plate', 'N/A')}")
print("-" * 30)

x, y, w, h = annotations['position_plate']
plate_img = image[y:y+h, x:x+w]
cv2.imshow("Placa", plate_img)
cv2.waitKey(0)

plate_text = predict(plate_img, config_plate)
print(f"Resultado (Placa Inteira):   {plate_text}")

# recognized_chars = []
# for (x_char, y_char, w_char, h_char) in annotations['chars']:
#     char_img = image[y_char:y_char+h_char, x_char:x_char+w_char]
#     char_text = predict(char_img, config_char)
#     recognized_chars.append(char_text)

# final_plate_from_chars = "".join(recognized_chars)
# print(f"Resultado (Caracteres Ind.): {final_plate_from_chars}")
# print("\n")