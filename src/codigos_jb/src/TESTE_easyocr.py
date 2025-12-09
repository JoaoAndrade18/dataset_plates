import easyocr
import cv2

reader = easyocr.Reader(['pt'], gpu=False) 

image_path = 'frames_gold/frame_19_processed.jpg_plate_processed.jpg'
results = reader.readtext(image_path)

print("Resultados do OCR:", results)
