from detect_plate import DetectPlate
from pipeline_preprocess import preprocess_pipeline
from paddleocr import PaddleOCR
import cv2

class PipelineOCR:
    def __init__(self):
        self.plate_detector = DetectPlate()
        self.preprocess = preprocess_pipeline

        self.ocr = PaddleOCR(
            use_textline_orientation=False,
            use_doc_unwarping=False,
            device='cpu',
            lang='en',
            text_detection_model_dir=None, 
            text_det_box_thresh=0.2,
            text_recognition_model_dir=None
        )

    def ocr_predict(self, preprocessed_image):
        preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2BGR)
        result = self.ocr.predict(preprocessed_image)
        result_ocr = result[0]['rec_texts'] 

        return result_ocr

    def postprocess(self, ocr_result):
        # Aplicar postprocessing steps
        return None

    def run(self, image):
        plate, detected_plate_box = self.plate_detector.detect_plate(image)

        print(f"Detected plate box: {detected_plate_box}")
        preprocessed = self.preprocess(plate, [
            'upsampling',      # Aumentar resolução 4x
            'deblurring',      # Remover blur
            'contrast',        # CLAHE para melhor contraste
            'noise_reduction', # Filtro bilateral
            'binarization',    # Threshold adaptativo
            'padding'          # Bordas brancas
        ])

        cv2.imshow("Preprocessed Image", preprocessed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ocr_result = self.ocr_predict(preprocessed)
        # final_output = self.postprocess(ocr_result)
        return ocr_result