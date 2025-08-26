import time
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\PC\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def predict(image) -> str:
        """  """
        config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(image, config=config)

        return text

n_images = [2,10,14,18,19]

for i in n_images:
    image_path = f'frames_gold/frame_{i}_processed.jpg_plate_processed.jpg'

    start = time.time()
    result = predict(image_path)
    end = time.time()

    print(f"\nResultados do OCR para {image_path}:", result)

    tempo_inferencia = end - start

    print(f"Tempo de inferência: {tempo_inferencia:.4f} segundos")

