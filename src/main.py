import cv2
import os

from detect_plate import DetectPlate
# from preprocess_plate import PreprocessPlate
# from ocr_plate import OCRPlate

detector = DetectPlate()
# preprocessor = PreprocessPlate()
# ocr = OCRPlate()

path = "C:/Users/PC/Documents/PROJETO-WISECONTROL/dataset_plates/src/codigos_jb/amostras_ufpr"

images = os.listdir(path)

for image in images:
    if image.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(path, image)
        placa, bounding_box = detector.detect_plate(image_path)

        # cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 255, 0), 2)

        cv2.imwrite(f"{path}/crop_{image}", placa)

        # cv2.imshow("Plate", placa)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()