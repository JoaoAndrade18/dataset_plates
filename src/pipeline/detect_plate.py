import cv2
import torch
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from PIL import Image

class DetectPlate:
    def __init__(self):
        self.feat = YolosFeatureExtractor.from_pretrained(
            "nickmuchi/yolos-small-finetuned-license-plate-detection"
        )
        self.model_plate = YolosForObjectDetection.from_pretrained(
            "nickmuchi/yolos-small-finetuned-license-plate-detection"
        )
        self.model_plate.eval()

    def detect_plate(self, image):
        image = cv2.imread(image)
        imagem_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        inputs = self.feat(images=imagem_pil, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model_plate(**inputs)

        probs = outputs.logits.softmax(-1)[0, :, 1]
        boxes = outputs.pred_boxes[0][probs > 0.5]

        if len(boxes) == 0:
            return None, None

        box = boxes[0].detach().cpu().numpy()
        W, H = imagem_pil.size
        cx, cy, w, h = box
        x1 = int((cx - w / 2) * W)
        y1 = int((cy - h / 2) * H)
        x2 = int((cx + w / 2) * W)
        y2 = int((cy + h / 2) * H)

        placa = image[y1:y2, x1:x2]
        
        return placa, (x1, y1, x2, y2)