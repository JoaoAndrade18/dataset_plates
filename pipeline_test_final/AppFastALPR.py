
from time import sleep
from fast_alpr import ALPR
from typing import List
import cv2
from ultralytics import YOLO # --- Optional ---

class ALPR_ATA:
    def __init__(self):
        """ Initialize the ALPR system with specified models """
        self.alpr = ALPR(
            detector_model="yolo-v9-s-608-license-plate-end2end",
            # ocr_model="cct-s-v1-global-model",
            ocr_model_path="cct_s_v1_global_plate_ft_br_me_plate_cars_motorcycles.onnx",
            ocr_config_path="cct_s_v1_global_plate_config.yaml"
        )
        try:
            self.auxLabelPlateDetector = YOLO('best_placa.pt').to(device="cuda")
        except Exception as e:
            self.auxLabelPlateDetector = YOLO('best_placa.pt').to(device="cpu")

    def fix_plate(self, text: str, cls: int = None, conf: float = None) -> str:
        def L(ch): return ch.isalpha()
        def N(ch): return ch.isdigit()

        old = [L, L, L, N, N, N, N] 
        new = [L, L, L, N, L, N, N]
        less = [L, L, L, N, None, N, N]
        
        corr = []
        for i, ch in enumerate(text[:7]): 
            if cls == 4 and conf > 0.6:
                f = old[i]
            elif cls == 5 and conf > 0.6:
                f = new[i]
            else:
                f = less[i]

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

    def predict(self, image_paths: List[str]) -> List[List[str]]:
        """ 
        

        Args:
            image_paths (List[str]): List of image paths 

        Returns:
            List of detected texts
        """
        results = []
        for image_path in image_paths:
            alpr_results = self.alpr.predict(image_path)
            yolo_results = self.auxLabelPlateDetector(image_path)

            if yolo_results and len(yolo_results[0].boxes) > 0:
                box = yolo_results[0].boxes[0]
                cls_ = int(box.cls[0])     
                conf = float(box.conf[0]) 
            else:
                cls_ = None
                conf = None

            if not alpr_results and len(yolo_results[0].boxes) > 0:
                image = cv2.imread(image_path)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                imageRoi = image[y1:y2, x1:x2]
                alpr_results = self.alpr.predict(imageRoi)

            if alpr_results and yolo_results and len(yolo_results[0].boxes) > 0:
                text_plate_corr = self.fix_plate(alpr_results[0].ocr.text, cls_, conf)
            elif alpr_results:
                text_plate_corr = self.fix_plate(alpr_results[0].ocr.text)
            else:
                text_plate_corr = ""

            results.append(text_plate_corr)

        return results