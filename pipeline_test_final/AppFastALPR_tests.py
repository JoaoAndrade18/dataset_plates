import cv2
from typing import List
from fast_alpr import ALPR
from ultralytics import YOLO
import os
import torch
import easyocr
import numpy as np

#  Remove the CPU forcing variables
if 'ORT_TENSORRT_DISABLED' in os.environ:
    del os.environ['ORT_TENSORRT_DISABLED']
if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
    del os.environ['CUDA_VISIBLE_DEVICES']
if 'ORT_DISABLE_CUDA' in os.environ:
    del os.environ['ORT_DISABLE_CUDA']

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class ALPR_ATA:
    def __init__(self):
        """ Initialize the ALPR system with specified models """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")
        
        if self.device == "cuda":
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        try:
            print("[INFO] Loading ALPR engine with GPU...")
            self.alpr = ALPR(
                detector_model="yolo-v9-s-608-license-plate-end2end",
                ocr_model_path="cct_s_v1_global_plate_ft_br_me_plate_cars_motorcycles.onnx",
                ocr_config_path="cct_s_v1_global_plate_config.yaml"
            )
            self.auxiliary_alpr = ALPR(
                detector_model="yolo-v9-s-608-license-plate-end2end",
                ocr_model="global-plates-mobile-vit-v2-model"
            ) 
            self.easy = easyocr.Reader(['en'])
            self.alpr_loaded = True
            print("[INFO] ALPR engine loaded successfully")
        except Exception as e:
            print(f"[ERROR] ALPR engine failed to load: {e}")
            self.alpr_loaded = False
            self.alpr = None

        try:
            print("[INFO] Loading YOLO plate detector with GPU...")
            self.auxLabelPlateDetector = YOLO('best_placa.pt')
            
            if self.device == "cuda":
                self.auxLabelPlateDetector = self.auxLabelPlateDetector.to(self.device)
                dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
                _ = self.auxLabelPlateDetector(dummy_input)
                # _ = self.alpr.predict(dummy_input)
                # _ = self.auxiliary_alpr.predict(dummy_input)
                
            self.yolo_loaded = True
            print("[INFO] YOLO plate detector loaded successfully on {self.device}")
        except Exception as e:
            print(f"[ERROR] YOLO plate detector failed to load: {e}")
            self.yolo_loaded = False
            self.auxLabelPlateDetector = None

    def fix_plate(self, text: str, cls: int = None, conf: float = None) -> str:
        def L(ch): return ch.isalpha()
        def N(ch): return ch.isdigit()

        old = [L, L, L, N, N, N, N] 
        new = [L, L, L, N, L, N, N]
        less = [L, L, L, N, None, N, N]

        text = "".join(ch for ch in text if ch.isalnum())
        text = text.upper()
        
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

    def predict(self, image_paths: List[np.ndarray]) -> List[dict]:
        """ 
        Predict automated license plates text in the given images.
        Args:
            image_paths (List[np.ndarray]): List of image paths 
        Returns:
            List[dict]: Each dict contains 'text', 'roi', and 'label'
        """
        results = []
        for image_path in image_paths:
            roi = None
            label = None
            text_plate_corr = ""

            try:
                alpr_results = self.alpr.predict(image_path)
                
                # YOLO plate detection
                yolo_results = None
                if self.yolo_loaded and self.auxLabelPlateDetector:
                    yolo_results = self.auxLabelPlateDetector(image_path)

                if yolo_results and len(yolo_results[0].boxes) > 0:
                    box = yolo_results[0].boxes[0]
                    cls_ = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    roi = [x1, y1, x2, y2]
                    label = self.auxLabelPlateDetector.model.names[cls_]

                if not alpr_results and roi != None:
                    image = image_path
                    if image is not None:
                        x1, y1, x2, y2 = roi
                        imageRoi = image[y1:y2, x1:x2]
                        alpr_results = self.alpr.predict(imageRoi)
                        # x1, y1 = max(0, x1), max(0, y1)
                        # x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                        
                        # if x2 > x1 and y2 > y1:
                        #     imageRoi = image[y1:y2, x1:x2]
                        #     with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        #         cv2.imwrite(temp_file.name, imageRoi)
                        #         alpr_results = self.alpr.predict(temp_file.name)
                        #         os.unlink(temp_file.name)

                if not alpr_results or alpr_results[0].ocr.text == "" or len(alpr_results[0].ocr.text) < 6:
                    # Use auxiliary model
                    print("[INFO] Using auxiliary model for OCR or text is less than 6 characters")   
                    alpr_results = self.auxiliary_alpr.predict(image_path)

                    if not alpr_results or alpr_results[0].ocr.text == "" or len(alpr_results[0].ocr.text) < 6 and roi != None:
                        image = image_path
                        if image is not None:
                            x1, y1, x2, y2 = roi
                            imageRoi = image[y1:y2, x1:x2]
                        else:
                            print("[INFO] Image is None")
                            continue

                        # use new model OCR -- easyocr --
                        print("[INFO] Using easyocr for OCR")
                        result = self.easy.readtext(imageRoi)
                        alpr_results = result[0][1]

                if alpr_results:
                    try:
                        if hasattr(alpr_results, "__getitem__"): # 1
                            ocr_text = alpr_results[0].ocr.text
                        else:
                            ocr_text = getattr(alpr_results.ocr, "text", "") # Never enter here
                    except:
                        ocr_text = alpr_results
                        print("esteve aqui ###################################################################################")

                    if yolo_results and len(yolo_results[0].boxes) > 0:
                        text_plate_corr = self.fix_plate(ocr_text, cls_, conf)
                    else:
                        text_plate_corr = self.fix_plate(ocr_text)
                else:
                    text_plate_corr = ""

            except Exception as e:
                print(f"[INFO] ALPR prediction error: {e}")
                text_plate_corr = ""

            results.append({
                "text": text_plate_corr,
                "roi": roi,
                "label": label
            })

        return results

alpr_service = ALPR_ATA()
