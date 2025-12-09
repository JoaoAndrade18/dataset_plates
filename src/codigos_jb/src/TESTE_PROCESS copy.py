import os
import cv2
import time
import glob
import typing as T
from PIL import Image
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
import pytesseract
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import re

# -- função para carregar modelo YOLOv5 --
def load_infer_model1(inference_model_path: str, gpu: bool = True):
    """
    Carrega o modelo YOLOv5 a partir do arquivo de pesos.
    """
    device = torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=inference_model_path, device=device)
        model.eval()
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo YOLOv5: {e}")
        return None

# -- classe para detecção de placa --
class DetectPlate:
    def __init__(self):
        self.feat = YolosFeatureExtractor.from_pretrained(
            "nickmuchi/yolos-small-finetuned-license-plate-detection"
        )
        self.model_plate = YolosForObjectDetection.from_pretrained(
            "nickmuchi/yolos-small-finetuned-license-plate-detection"
        )
        self.model_plate.eval()

    def detect_plate(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro: Não foi possível carregar a imagem {image_path}")
            return None, None
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
        return Image.fromarray(cv2.cvtColor(placa, cv2.COLOR_BGR2RGB)), (x1, y1, x2, y2)

# -- funções de deskew --
def rotate_image(image, angle):
    """
    Rotaciona a imagem em torno de seu centro pelo ângulo especificado.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def compute_skew(image):
    """
    Calcula o ângulo de inclinação (skew) da imagem com base em detecção de bordas e linhas.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    blur = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(blur, threshold1=100, threshold2=150, apertureSize=5, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=gray.shape[1] // 2, maxLineGap=30)

    if lines is None:
        return 0.0 

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = angle_rad * 180.0 / np.pi
        if -30 < angle_deg < 30:
            angles.append(angle_deg)

    if len(angles) == 0:
        return 0.0
    
    return np.mean(angles)

def deskew(image):
    """
    Corrige a inclinação da imagem automaticamente.
    """
    angle = compute_skew(image)
    return rotate_image(image, angle)

# -- função de upscaling --
def upscale_image(image):
    """
    Aumenta a resolução da imagem de forma mais suave.
    """
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel("ESPCN_x3.pb")
        sr.setModel("espcn", 3)
        result = sr.upsample(image)
        return result
    except Exception as e:
        print(f"Erro no upscaling: {e}")
        return image

# -- preprocessamento configurável --
def preprocess_image(
    image: T.Union[str, Image.Image],
    use_grayscale=True,
    use_blur=True,
    use_threshold=True,
    use_erode=True,
    use_dilate=True,
    use_deskew=False,
    use_upscale=False
):
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Não foi possível carregar a imagem: {image}")
    else:
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if use_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if use_blur:
        img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    
    if use_threshold and len(img.shape) == 2:
        img = cv2.adaptiveThreshold(img, 70, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 11)
    
    if use_erode and len(img.shape) == 2:
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.erode(img, kernel_rect, iterations=1)
    
    if use_dilate and len(img.shape) == 2:
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.dilate(img, kernel_rect, iterations=1)
    
    if use_deskew:
        img = deskew(img)
    
    if use_upscale:
        img = upscale_image(img)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img)

# -- função para YOLOv5 (detecção de caracteres) --
def ocr_detect(frame, model, class_names):
    text_plate = ""
    try:
        rets = []
        plate_list = []
        res = model(frame)
        for i in res.xyxy[0]:
            x1 = int(i[0])
            y1 = int(i[1])
            x2 = int(i[2])
            y2 = int(i[3])
            rets.append([[x1, y1, x2-x1, y2-y1], int(i[5])])
        for boxL, classId in rets:
            if boxL[2] * boxL[3] > 700:
                continue
            stop = False
            for b in plate_list:
                if abs(boxL[0] - b[0]) < 5:
                    stop = True
                    break
            if stop:
                continue
            char = class_names[classId]
            if len(plate_list) == 0:
                plate_list.append((boxL[0], char))
            else:
                tamList = len(plate_list)
                inserido = False
                for i in range(tamList):
                    if plate_list[i][0] > boxL[0]:
                        plate_list.insert(i, (boxL[0], char))
                        inserido = True
                        break
                if not inserido:
                    plate_list.append((boxL[0], char))
        plate_list = [x2 for x1, x2 in plate_list]
        text_plate = "".join(plate_list)
    except Exception as e:
        print(f"Erro no YOLOv5 OCR: {e}")
    return text_plate

# -- load datasets --
def open_datasets(path_alpr: str, gold_csv: str, gold_path: str = "frames_gold", image_ext_alpr: str = ".png", image_ext_gold: str = ".jpg", txt_ext: str = ".txt"):
    # Carregar CSV do gold
    try:
        df = pd.read_csv(gold_csv)
        if 'ID_image' not in df.columns or 'plate_car' not in df.columns:
            raise ValueError("Colunas 'ID_image' ou 'plate_car' não encontradas no CSV")
        df_gold = df[['ID_image', 'plate_car']].copy()
        df_gold['image'] = df_gold['ID_image'].apply(lambda x: os.path.normpath(os.path.join(gold_path, x)))
        df_gold = df_gold.rename(columns={'plate_car': 'gold_plate'})
        print(f"df_gold carregado com {len(df_gold)} entradas")
        print("Primeiras entradas de df_gold:")
        print(df_gold[['image', 'gold_plate']].head())
    except Exception as e:
        print(f"Erro ao carregar gold_csv: {e}")
        return None, None

    # Carregar imagens do UFPR-ALPR
    data = []
    for img_path in glob.glob(os.path.join(path_alpr, "**", f"*{image_ext_alpr}"), recursive=True):
        base, _ = os.path.splitext(img_path)
        txt_path = base + txt_ext
        plate_value = None
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().lower().startswith("plate:"):
                            plate_value = line.split(":", 1)[1].strip()
                            break
            except Exception as e:
                print(f"Erro ao ler {txt_path}: {e}")
        data.append({"image": os.path.normpath(img_path), "gold_plate": plate_value})
    
    df_alpr = pd.DataFrame(data)
    print(f"df_alpr carregado com {len(df_alpr)} imagens")
    print("Primeiras entradas de df_alpr:")
    print(df_alpr[['image', 'gold_plate']].head())
    
    return df_gold, df_alpr

# -- load models --
def load_models(models: list[str], inference_model_path: str, class_names_path: str, gpu: bool = True, use_plate_detection: bool = True):
    paddleocr = None
    pytesseract_model = None
    config = None
    qwen_model = None
    processor = None
    messages = None
    yolo_model = None
    class_names = []
    detect_plate = None

    if use_plate_detection and ('paddleOCR' in models or 'tesseract' in models or 'qwen' in models):
        detect_plate = DetectPlate()

    if 'paddleOCR' in models:
        try:
            paddleocr = PaddleOCR(
                use_textline_orientation=False,
                use_doc_unwarping=False,
                device='gpu' if gpu and torch.cuda.is_available() else 'cpu',
                lang='en',
                text_detection_model_dir=None,
                text_det_box_thresh=0.2,
                text_recognition_model_dir=None
            )
        except Exception as e:
            print(f"Erro ao carregar PaddleOCR: {e}")

    if 'tesseract' in models:
        pytesseract_model = pytesseract
        pytesseract_model.pytesseract.tesseract_cmd = r'C:\Users\PC\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
        config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    if 'qwen' in models:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"
        try:
            qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype="auto"
            ).to(device)
            processor = AutoProcessor.from_pretrained(model_id)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Transcreva o texto que aparece nesta imagem."}
                    ]
                }
            ]
        except Exception as e:
            print(f"Erro ao carregar Qwen: {e}")

    if 'yolov5' in models:
        yolo_model = load_infer_model1(inference_model_path, gpu)
        try:
            with open(class_names_path, "r") as f:
                class_names = [cname.strip() for cname in f.readlines()]
        except Exception as e:
            print(f"Erro ao carregar class_names: {e}")

    return {
        'paddleocr': paddleocr,
        'pytesseract': pytesseract_model,
        'config': config,
        'qwen_model': qwen_model,
        'processor': processor,
        'messages': messages,
        'yolo_model': yolo_model,
        'class_names': class_names,
        'detect_plate': detect_plate
    }

# -- contar letras corretas --
def count_correct_letters(pred: str, gold: str):
    pred = re.sub(r'[^A-Z0-9]', '', pred.upper()) if pred else ''
    gold = re.sub(r'[^A-Z0-9]', '', gold.upper()) if gold else ''
    if len(pred) != len(gold):
        return 0
    return sum(1 for p, g in zip(pred, gold) if p == g)

# -- executar OCR para uma imagem --
def execute_ocr_for_image(
    model_dict,
    image_path,
    gold_plate,
    use_preprocess=True,
    use_grayscale=True,
    use_blur=True,
    use_threshold=True,
    use_erode=True,
    use_dilate=True,
    use_deskew=False,
    use_upscale=False,
    use_plate_detection=True,
    save_debug_images=False,
    debug_dir="debug_images"
):
    results = {'image': image_path, 'gold_plate': gold_plate if gold_plate else ''}
    
    # Carregar imagem original
    original_image = Image.open(image_path).convert("RGB")
    print(f"Imagem original: {image_path}, tamanho: {original_image.size}")
    
    # Criar diretório de debug, se necessário
    if save_debug_images:
        os.makedirs(debug_dir, exist_ok=True)
        original_image.save(os.path.join(debug_dir, f"{os.path.basename(image_path)}_original.png"))
    
    plate_image = None
    if use_plate_detection and model_dict['detect_plate']:
        plate_image, _ = model_dict['detect_plate'].detect_plate(image_path)
        if plate_image is not None:
            print(f"Placa detectada: tamanho {plate_image.size}")
            if save_debug_images:
                plate_image.save(os.path.join(debug_dir, f"{os.path.basename(image_path)}_plate.png"))
        else:
            print(f"Placa não detectada para {image_path}")
    
    # Selecionar imagem para OCR
    if plate_image is not None:
        input_image = plate_image
        if use_preprocess:
            input_image = preprocess_image(
                plate_image,
                use_grayscale=use_grayscale,
                use_blur=use_blur,
                use_threshold=use_threshold,
                use_erode=use_erode,
                use_dilate=use_dilate,
                use_deskew=use_deskew,
                use_upscale=use_upscale
            )
            print(f"Imagem após pré-processamento: tamanho {input_image.size}")
            if save_debug_images:
                input_image.save(os.path.join(debug_dir, f"{os.path.basename(image_path)}_preprocessed.png"))
    else:
        print(f"Usando imagem inteira para {image_path}")
        input_image = preprocess_image(
            image_path,
            use_grayscale=use_grayscale,
            use_blur=use_blur,
            use_threshold=use_threshold,
            use_erode=use_erode,
            use_dilate=use_dilate,
            use_deskew=use_deskew,
            use_upscale=use_upscale
        ) if use_preprocess else original_image
        print(f"Imagem após pré-processamento (ou original): tamanho {input_image.size}")
        if save_debug_images:
            input_image.save(os.path.join(debug_dir, f"{os.path.basename(image_path)}_preprocessed.png"))
    
    image_cv = cv2.imread(image_path)
    
    if model_dict['paddleocr']:
        start_time = time.time()
        temp_path = "temp_plate.png"
        input_image.save(temp_path)
        try:
            result = model_dict['paddleocr'].predict(temp_path)
            textos = result[0]['rec_texts'] if result and result[0] and 'rec_texts' in result[0] else []
            print(f"Resultados do OCR bruto para {image_path}: {textos}")
            texto = textos[0] if textos else ''  # Usar o primeiro texto, se disponível
        except Exception as e:
            print(f"Erro no PaddleOCR para {image_path}: {e}")
            textos = []
            texto = ''
        time_ms = (time.time() - start_time) * 1000
        correct = count_correct_letters(texto, gold_plate)
        results['paddle_text'] = texto
        results['paddle_correct'] = correct
        results['paddle_time_ms'] = time_ms
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    if model_dict['pytesseract']:
        start_time = time.time()
        try:
            text = model_dict['pytesseract'].image_to_string(input_image, config=model_dict['config'])
            text = text.strip()
            print(f"Tesseract resultado para {image_path}: {text}")
        except Exception as e:
            print(f"Erro no Tesseract para {image_path}: {e}")
            text = ''
        time_ms = (time.time() - start_time) * 1000
        correct = count_correct_letters(text, gold_plate)
        results['tesseract_text'] = text
        results['tesseract_correct'] = correct
        results['tesseract_time_ms'] = time_ms
    
    if model_dict['qwen_model']:
        start_time = time.time()
        try:
            text_prompt = model_dict['processor'].apply_chat_template(model_dict['messages'], tokenize=False, add_generation_prompt=True)
            inputs = model_dict['processor'](text=text_prompt, images=input_image, return_tensors="pt").to(model_dict['qwen_model'].device)
            generated_ids = model_dict['qwen_model'].generate(**inputs, max_new_tokens=7)
            generated_texts = model_dict['processor'].batch_decode(generated_ids, skip_special_tokens=True)
            resultado_final = generated_texts[0].strip()
            pattern = r"<\|im_start\|>assistant\s*(.*?)\s*<\|im_end\|>"
            match = re.search(pattern, resultado_final, re.DOTALL)
            extracted_text = match.group(1).strip() if match else resultado_final
            print(f"Qwen resultado para {image_path}: {extracted_text}")
        except Exception as e:
            print(f"Erro no Qwen para {image_path}: {e}")
            extracted_text = ''
        time_ms = (time.time() - start_time) * 1000
        correct = count_correct_letters(extracted_text, gold_plate)
        results['qwen_text'] = extracted_text
        results['qwen_correct'] = correct
        results['qwen_time_ms'] = time_ms
    
    if model_dict['yolo_model']:
        start_time = time.time()
        try:
            text = ocr_detect(image_cv, model_dict['yolo_model'], model_dict['class_names'])
            print(f"YOLOv5 resultado para {image_path}: {text}")
        except Exception as e:
            print(f"Erro no YOLOv5 para {image_path}: {e}")
            text = ''
        time_ms = (time.time() - start_time) * 1000
        correct = count_correct_letters(text, gold_plate)
        results['yolov5_text'] = text
        results['yolov5_correct'] = correct
        results['yolov5_time_ms'] = time_ms
    
    return results

# -- executar testes --
def run_tests(
    path_alpr: str,
    gold_csv: str,
    models_list: list[str],
    inference_model_path: str,
    class_names_path: str,
    gpu: bool = True,
    output_csv: str = "ocr_results.csv",
    use_preprocess: bool = True,
    use_grayscale: bool = True,
    use_blur: bool = True,
    use_threshold: bool = True,
    use_erode: bool = True,
    use_dilate: bool = True,
    use_deskew: bool = False,
    use_upscale: bool = False,
    use_plate_detection: bool = True,
    save_debug_images: bool = False,
    debug_dir: str = "debug_images",
    num_images: int = None
):
    # Carregar datasets
    df_gold, df_alpr = open_datasets(path_alpr, gold_csv, image_ext_alpr=".png", image_ext_gold=".jpg")
    if df_gold is None or df_alpr is None:
        print("Erro ao carregar datasets. Encerrando.")
        return
    
    # Combinar os DataFrames
    df_gold = df_gold[['image', 'gold_plate']]
    df_alpr = df_alpr[['image', 'gold_plate']]
    df_combined = pd.concat([df_gold, df_alpr], ignore_index=True)
    print(f"df_combined contém {len(df_combined)} entradas")
    print("Primeiras entradas de df_combined:")
    print(df_combined.head())
    
    # Limitar o número de imagens para testes rápidos
    if num_images is not None:
        df_combined = df_combined.head(num_images)
    
    model_dict = load_models(models_list, inference_model_path, class_names_path, gpu, use_plate_detection)
    
    all_results = []
    for _, row in df_combined.iterrows():
        image_path = row['image']
        gold_plate = row['gold_plate']
        if pd.isna(gold_plate):
            gold_plate = ""  # Tratar casos onde a placa não está disponível
        print(f"\nProcessando imagem: {image_path}")
        result = execute_ocr_for_image(
            model_dict,
            image_path,
            gold_plate,
            use_preprocess=use_preprocess,
            use_grayscale=use_grayscale,
            use_blur=use_blur,
            use_threshold=use_threshold,
            use_erode=use_erode,
            use_dilate=use_dilate,
            use_deskew=use_deskew,
            use_upscale=use_upscale,
            use_plate_detection=use_plate_detection,
            save_debug_images=save_debug_images,
            debug_dir=debug_dir
        )
        all_results.append(result)
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(output_csv, index=False)
    print(f"Resultados salvos em {output_csv}")

# Exemplo de uso
run_tests(
    path_alpr="frames_ufpr",
    gold_csv="gold_images.csv",
    models_list=["paddleOCR"],
    inference_model_path="OCR_novo.pt",
    class_names_path="ocr-net.names",
    gpu=True,
    output_csv="ocr_results.csv",
    use_preprocess=True,
    use_grayscale=True,
    use_blur=True,
    use_threshold=True,
    use_erode=True,
    use_dilate=True,
    use_deskew=False,
    use_upscale=False,
    use_plate_detection=True,
    save_debug_images=True,
    debug_dir="debug_images",
    num_images=10
)