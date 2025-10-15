
from fast_alpr import ALPR
from typing import List
import cv2
from ultralytics import YOLO # --- Optional ---

class FastALPR:
    def __init__(self):
        """ Initialize the ALPR system with specified models """
        self.alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v1-global-model",
        )
        self.auxLabelPlateDetector = YOLO('best_placa.pt')

    def fix_plate(self, text):
        def L(ch): return ch.isalpha()
        def N(ch): return ch.isdigit()

        default = [L, L, L, N, None, N, N] 
        
        corr = []
        for i, ch in enumerate(text[:7]): 
            f = default[i] 
            
            if f is None:
                corr.append(ch) 
            elif f == L:  
                if L(ch): 
                    corr.append(ch)
                else:  
                    mapa = {'0':'O', '1':'I', '2':'Z', '3':'E', '4':'A', '5':'S', '6':'G', '7':'T', '8':'B', '9':'B'}
                    corr.append(mapa.get(ch, ch))
            else:  
                if N(ch):  
                    corr.append(ch)
                else: 
                    mapa = {'O':'0', 'Q':'0', 'D':'0', 'I':'1', 'Z':'2', 'A':'4', 'S':'5', 'G':'6', 'T':'7', 'B':'8', 
                            'E':'3', 'C':'6', 'Y':'4', 'L':'1', 'F':'5', 'H':'4', 'X':'8', 'J':'1', 'K':'6', 
                            'M':'1', 'N':'1', 'P':'9', 'R':'2', 'U':'0', 'V':'8', 'W':'3'}
                    corr.append(mapa.get(ch, ch))
        
        return "".join(corr)

    def process_images(self, image_paths: List[str]) -> List[str]:
        """ Return a ordered list of detected texts """
        results = []
        for image_path in image_paths:
            alpr_results = self.alpr.predict(image_path)

            if not alpr_results:
                print(f"[INFO] No results for {image_path}, loading YOLO detector...")
                yolo_results = self.auxLabelPlateDetector(image_path)
                image = cv2.imread(image_path)
                if yolo_results and len(yolo_results[0].boxes) > 0:
                    box = yolo_results[0].boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    imageRoi = image[y1:y2, x1:x2]
                    alpr_results = self.alpr.predict(imageRoi)

            results.append(self.fix_plate(alpr_results[0].ocr.text))

        return results