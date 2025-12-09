# -*- coding: utf-8 -*-
from time import sleep
from typing import List, Dict, Any, Tuple, Optional
import os, re, time
import cv2
import numpy as np
from fast_alpr import ALPR
from ultralytics import YOLO  # para veículo (opcional) e placa (fallback)

# ==========================
# Configs de debug/saída
# ==========================
DEBUG_VISUALIZE   = False                 # Exibir janelas cv2.imshow (bloqueante)
SAVE_DEBUG        = True                  # Salvar imagens de depuração
SAVE_ONLY_ERRORS  = True                  # Salvar só erros; False = salva tudo
OUT_DIR           = "debugs"              # Raiz de saída

# ==========================
# Regex de tipos de placa
# ==========================
OLD_PLATE_RE      = re.compile(r"^[A-Z]{3}\d{4}$")        # AAA1234
MERCOSUL_PLATE_RE = re.compile(r"^[A-Z]{3}\d[A-Z]\d{2}$") # AAA1A23

def classify_plate_type_gt(plate_text: str) -> str:
    """Classifica o tipo do GROUND TRUTH: 'velha' | 'nova' | 'desconhecida'."""
    if OLD_PLATE_RE.match(plate_text):
        return "velha"
    if MERCOSUL_PLATE_RE.match(plate_text):
        return "nova"
    return "desconhecida"

def clean_plate_text(text: Any) -> str:
    """Remove não-alfanuméricos e coloca em maiúsculas."""
    return ''.join(c for c in str(text).upper() if c.isalnum())

def pattern_score(text7: str, pattern: List[Optional[str]]) -> float:
    """
    Mede aderência do texto (até 7 chars) a um 'pattern' de L/N/None.
    Retorna um score em [0,1].
    """
    def L(ch): return ch.isalpha()
    def N(ch): return ch.isdigit()
    text7 = (text7 or "")[:7]
    if len(text7) == 0:
        return 0.0
    total = min(7, len(text7))
    ok = 0
    for i, ch in enumerate(text7):
        f = pattern[i]
        if f is None:
            ok += 1
        elif f == 'L' and L(ch):
            ok += 1
        elif f == 'N' and N(ch):
            ok += 1
    return ok / total

# Padrões para score heurístico
PATTERN_OLD  = ['L','L','L','N','N','N','N']     # AAA1234
PATTERN_NEW  = ['L','L','L','N','L','N','N']     # AAA1A23
PATTERN_LESS = ['L','L','L','N', None,'N','N']   # relaxado

def infer_type_from_text(text: str) -> Tuple[Optional[str], float]:
    """
    Infere 'velha'/'nova' a partir do texto (7 chars), com score heurístico ∈ [0,1].
    Retorna (tipo|None, score).
    """
    t = clean_plate_text(text)[:7]
    s_old = pattern_score(t, PATTERN_OLD)
    s_new = pattern_score(t, PATTERN_NEW)
    if s_old == 0 and s_new == 0:
        return None, 0.0
    if s_old >= s_new:
        return "velha", s_old
    else:
        return "nova", s_new

# ==========================
# Classe principal (ALPR)
# ==========================
class ALPR_ATA:
    def __init__(self, type_conf_threshold: float = 0.60):
        """
        Inicializa FAST-ALPR e YOLO auxiliar de PLACA.
        - type_conf_threshold: limiar usado para decidir o 'cls' (velha/nova).
        """
        self.type_conf_threshold = float(type_conf_threshold)

        self.alpr = ALPR(
            detector_model="yolo-v9-s-608-license-plate-end2end",
            ocr_model="cct-s-v1-global-model",
            # ocr_model_path="cct_s_v1_global_plate_ft_brasilian_plate.onnx",
            # ocr_config_path="cct_s_v1_global_plate_config.yaml"
        )
        try:
            self.auxLabelPlateDetector = YOLO('best_placa.pt').to(device="cuda")
        except Exception:
            self.auxLabelPlateDetector = YOLO('best_placa.pt').to(device="cpu")

        # IDs do seu modelo auxiliar de placa (ajuste conforme necessário)
        self.cls_map = {4: "velha", 5: "nova"}

    def fix_plate(self, text: str, cls: int = None, conf: float = None) -> str:
        """Correção leve por padrão esperado (sua lógica original preservada)."""
        def L(ch): return ch.isalpha()
        def N(ch): return ch.isdigit()

        old  = [L, L, L, N, N, N, N]
        new  = [L, L, L, N, L, N, N]
        less = [L, L, L, N, None, N, N]

        corr = []
        for i, ch in enumerate(text[:7]):
            if cls == 4 and conf and conf > 0.6:
                f = old[i]
            elif cls == 5 and conf and conf > 0.6:
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
                    mapa = {'O':'0','Q':'0','D':'0','I':'1','Z':'2','A':'4','S':'5','G':'6','T':'7','B':'8',
                            'E':'3','C':'6','Y':'4','L':'1','F':'5','H':'4','X':'8','J':'1','K':'6',
                            'M':'1','N':'1','P':'9','R':'2','U':'0','V':'8','W':'3'}
                    corr.append(mapa.get(ch, ch))
        return "".join(corr)

    def _extract_bbox_from_fast(self, det) -> Optional[list]:
        """Extrai bbox [x1, y1, x2, y2] do fast_alpr."""
        try:
            bb = det.detection.bounding_box
            x1 = int(bb.x1); y1 = int(bb.y1); x2 = int(bb.x2); y2 = int(bb.y2)
            if x2 > x1 and y2 > y1:
                return [x1, y1, x2, y2]
        except Exception:
            pass
        return None

    def _normalize_input(self, item: Any) -> Tuple[Any, bool]:
        """Aceita caminho (str) ou ndarray. Retorna (obj, is_array)."""
        if isinstance(item, np.ndarray):
            return item, True
        if isinstance(item, str):
            return item, False
        return str(item), False

    def predict(self, images: List[Any]) -> List[Dict[str, Any]]:
        """
        Retorno por item:
        {
            "text": <str>,
            "bbox": <[x1,y1,x2,y2] | None>,
            "conf": <float | None>,   # confiança do detector usado
            "source": <'fast'|'yolo'|'none'>,
            "cls": <'velha'|'nova'|None>    # tipo com threshold aplicado
        }
        """
        outputs = []
        for item in images:
            text_plate_corr = ""
            bbox = None
            conf = None
            source = 'none'
            cls_out: Optional[str] = None

            obj, is_array = self._normalize_input(item)

            # 1) FAST-ALPR primeiro
            try:
                fast_results = self.alpr.predict(obj)  # aceita path OU ndarray
            except Exception:
                fast_results = None

            if fast_results and len(fast_results) > 0:
                det = fast_results[0]
                try:
                    raw_text = det.ocr.text
                except Exception:
                    raw_text = str(det)

                bbox = self._extract_bbox_from_fast(det)
                try:
                    conf = float(det.detection.confidence)
                except Exception:
                    conf = None

                text_plate_corr = self.fix_plate(raw_text)

                # Tipo inferido do texto com score heurístico
                inferred_type, type_score = infer_type_from_text(text_plate_corr)
                if inferred_type is not None and type_score >= self.type_conf_threshold:
                    cls_out = inferred_type
                else:
                    cls_out = None

                source = 'fast'

            else:
                # 2) Fallback: YOLO da PLACA + OCR FAST no ROI
                try:
                    yolo_results = self.auxLabelPlateDetector(obj)
                except Exception:
                    yolo_results = None

                if yolo_results and len(yolo_results[0].boxes) > 0:
                    boxes = yolo_results[0].boxes
                    best_i = int(boxes.conf.argmax())
                    box = boxes[best_i]
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox = [x1, y1, x2, y2]

                    if is_array:
                        image = obj
                    else:
                        image = cv2.imread(obj)
                    if image is not None:
                        roi = image[y1:y2, x1:x2]
                        try:
                            roi_results = self.alpr.predict(roi)
                        except Exception:
                            roi_results = None
                        if roi_results and len(roi_results) > 0:
                            try:
                                raw_text = roi_results[0].ocr.text
                            except Exception:
                                raw_text = str(roi_results[0])
                            text_plate_corr = self.fix_plate(raw_text, cls_id, conf)
                        else:
                            text_plate_corr = ""
                    else:
                        text_plate_corr = ""

                    mapped = self.cls_map.get(cls_id, None)
                    if mapped is not None and conf is not None and conf >= self.type_conf_threshold:
                        cls_out = mapped
                    else:
                        cls_out = None

                    source = 'yolo'
                else:
                    text_plate_corr = ""
                    source = 'none'

            outputs.append({
                "text": text_plate_corr,
                "bbox": bbox,
                "conf": conf,
                "source": source,
                "cls": cls_out
            })

        return outputs

# ==========================
# (Opcional) ROI de VEÍCULO
# ==========================
USE_VEHICLE_ROI = True           # já ativado
YOLOV11_WEIGHTS = "yolo11m.pt"
YOLOV11_DEVICE  = "cuda"
YOLOV11_CONF    = 0.25
YOLOV11_IOU     = 0.5
VEH_EXPAND      = 0.08
COCO_VEHICLE_IDS = {2, 3, 5, 7}   # car, motorcycle, bus, truck

yolo_vehicle = None
if USE_VEHICLE_ROI:
    yolo_vehicle = YOLO(YOLOV11_WEIGHTS)

def _clamp(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(round(x1)), w-1))
    x2 = max(0, min(int(round(x2)), w-1))
    y1 = max(0, min(int(round(y1)), h-1))
    y2 = max(0, min(int(round(y2)), h-1))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return x1,y1,x2,y2

def _expand(x1, y1, x2, y2, w, h, r=VEH_EXPAND):
    bw = x2-x1; bh = y2-y1
    dx = int(round(bw*r)); dy = int(round(bh*r))
    return _clamp(x1-dx, y1-dy, x2+dx, y2+dy, w, h)

def _vehicle_roi(img_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int,int,int,int]]]:
    """
    Retorna (ROI do veículo, bbox_xyxy ou None). Sem resize.
    Se não detectar veículo, devolve (imagem inteira, None).
    """
    if yolo_vehicle is None:
        return img_bgr, None
    h, w = img_bgr.shape[:2]
    res = yolo_vehicle.predict(source=img_bgr, device=YOLOV11_DEVICE,
                               conf=YOLOV11_CONF, iou=YOLOV11_IOU, verbose=False)
    best = None; best_area = -1
    for r in res:
        if r.boxes is None:
            continue
        for b in r.boxes:
            cid = int(b.cls.item())
            if cid not in COCO_VEHICLE_IDS:
                continue
            x1,y1,x2,y2 = b.xyxy[0].tolist()
            x1,y1,x2,y2 = _clamp(x1,y1,x2,y2,w,h)
            x1,y1,x2,y2 = _expand(x1,y1,x2,y2,w,h,VEH_EXPAND)
            area = (x2-x1)*(y2-y1)
            if area > best_area:
                best = (x1,y1,x2,y2); best_area = area
    if best is None:
        roi = img_bgr
        bbox = None
    else:
        x1,y1,x2,y2 = best
        roi = img_bgr[y1:y2, x1:x2].copy()
        bbox = (x1,y1,x2,y2)

    if DEBUG_VISUALIZE:
        print(f"[DEBUG] ROI veículo -> shape={roi.shape}, dtype={roi.dtype}, range=({roi.min()},{roi.max()})")
        cv2.imshow("vehicle_roi", roi)
        cv2.waitKey(300)

    return roi, bbox

# ==========================
# Utilidades de salvamento
# ==========================
def _ensure_dirs():
    if not SAVE_DEBUG:
        return
    subdirs = [
        "annotated",              # originais anotadas (apenas bbox veículo)
        "annotated_roi",          # ROI do veículo com bbox da placa
        "vehicle_rois",           # recortes de veículo (sem anotação)
        "errors_text",
        "errors_type",
        "errors_none_pred",
        "ok"
    ]
    for sd in subdirs:
        os.makedirs(os.path.join(OUT_DIR, sd), exist_ok=True)

def _draw_info(
    img,
    bbox_vehicle: Optional[Tuple[int,int,int,int]],
    bbox_plate:   Optional[Tuple[int,int,int,int]],
    gt_text,
    gt_type,
    pred_text,
    pred_cls,
    source,
    conf,
    acertos
):
    """
    Desenha:
      - bbox do veículo (verde) se fornecido
      - bbox da placa (vermelho) se fornecido
      + textos com GT, predição etc.
    """
    vis = img.copy()

    # bbox veículo (verde)
    if bbox_vehicle is not None:
        x1,y1,x2,y2 = bbox_vehicle
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 2)

    # bbox placa (vermelho)
    if bbox_plate is not None:
        px1,py1,px2,py2 = bbox_plate
        cv2.rectangle(vis, (px1,py1), (px2,py2), (0,0,255), 2)

    # legendas
    h = 24
    lines = [
        f"GT: {gt_text} [{gt_type}]",
        f"PRED: {pred_text} [{pred_cls}]",
        f"SRC: {source}  conf:{conf if conf is not None else 'None'}",
        f"ACERTOS: {acertos}/7"
    ]
    y0 = 25
    for i, line in enumerate(lines):
        y = y0 + i*h
        cv2.putText(vis, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(vis, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return vis

def _save_debug_images(
    base_name,
    img_bgr,
    vehicle_roi,
    bbox_vehicle,
    bbox_plate_full,
    bbox_plate_roi,
    gt_text,
    gt_type,
    pred_text,
    pred_cls,
    source,
    conf,
    acertos
):
    if not SAVE_DEBUG:
        return

    # 1) Imagem COMPLETA: só bbox do veículo (como você pediu)
    annotated_full = _draw_info(
        img_bgr,
        bbox_vehicle=bbox_vehicle,
        bbox_plate=None,   # NÃO desenha bbox da placa aqui
        gt_text=gt_text,
        gt_type=gt_type,
        pred_text=pred_text,
        pred_cls=pred_cls,
        source=source,
        conf=conf,
        acertos=acertos
    )
    path_annot = os.path.join(OUT_DIR, "annotated", f"{base_name}.jpg")
    cv2.imwrite(path_annot, annotated_full)

    # 2) ROI bruto do veículo (sem anotação, caso queira usar depois)
    path_roi_raw = os.path.join(OUT_DIR, "vehicle_rois", f"{base_name}.png")
    cv2.imwrite(path_roi_raw, vehicle_roi)

    # 3) ROI anotado com bbox da placa (coordenadas relativas ao ROI)
    if (bbox_vehicle is not None) and (bbox_plate_roi is not None) and (vehicle_roi is not None):
        annotated_roi = _draw_info(
            vehicle_roi,
            bbox_vehicle=None,          # aqui já é só o veículo
            bbox_plate=bbox_plate_roi,  # bbox da placa relativo ao ROI
            gt_text=gt_text,
            gt_type=gt_type,
            pred_text=pred_text,
            pred_cls=pred_cls,
            source=source,
            conf=conf,
            acertos=acertos
        )
        path_annot_roi = os.path.join(OUT_DIR, "annotated_roi", f"{base_name}.jpg")
        cv2.imwrite(path_annot_roi, annotated_roi)

    # classificação para pastas de erro/ok usa a imagem completa anotada
    is_text_ok = (acertos == 7)
    is_type_ok = (pred_cls is None) or (gt_type not in ("velha","nova")) or (pred_cls == gt_type)

    if SAVE_ONLY_ERRORS:
        if not is_text_ok:
            cv2.imwrite(os.path.join(OUT_DIR, "errors_text", f"{base_name}.jpg"), annotated_full)
        if (gt_type in ("velha","nova")) and (pred_cls is None):
            cv2.imwrite(os.path.join(OUT_DIR, "errors_none_pred", f"{base_name}.jpg"), annotated_full)
        if (gt_type in ("velha","nova")) and (pred_cls is not None) and (pred_cls != gt_type):
            cv2.imwrite(os.path.join(OUT_DIR, "errors_type", f"{base_name}.jpg"), annotated_full)
    else:
        if is_text_ok and is_type_ok:
            cv2.imwrite(os.path.join(OUT_DIR, "ok", f"{base_name}.jpg"), annotated_full)
        else:
            if not is_text_ok:
                cv2.imwrite(os.path.join(OUT_DIR, "errors_text", f"{base_name}.jpg"), annotated_full)
            if (gt_type in ("velha","nova")) and (pred_cls is None):
                cv2.imwrite(os.path.join(OUT_DIR, "errors_none_pred", f"{base_name}.jpg"), annotated_full)
            if (gt_type in ("velha","nova")) and (pred_cls is not None) and (pred_cls != gt_type):
                cv2.imwrite(os.path.join(OUT_DIR, "errors_type", f"{base_name}.jpg"), annotated_full)

# ==========================
# Leitura de GT e métricas
# ==========================

def read_ground_truth(image_path: str) -> Optional[str]:
    """
    GT (PODI-LPR):
    - GT = 7 primeiros caracteres do nome do arquivo de imagem.
      Ex.: 'AFV7185_2019-03-28_10-11-23_93....jpg' -> 'AFV7185'
    """
    try:
        base = os.path.basename(image_path)
        name, _ = os.path.splitext(base)
        if len(name) < 7:
            print(f"Nome de arquivo muito curto para GT (esperado >= 7 chars): {base}")
            return None
        gt = clean_plate_text(name[:7])
        return gt if gt else None
    except Exception as e:
        print(f"Erro ao extrair GT do nome '{image_path}': {e}")
        return None

def check_match(pred: str, expected: str) -> int:
    """Retorna o número de caracteres corretos (posicional)."""
    pred = str(pred).upper()
    expected = str(expected).upper()
    return sum(p == t for p, t in zip(pred, expected))

# ==========================
# Avaliação principal
# ==========================
def evaluate_all(images_dir: str, model: ALPR_ATA) -> None:
    _ensure_dirs()

    total = 0
    mediumTime = 0.0
    sevenCorrected = 0
    sixCorrected = 0
    totalImagesInference = 0

    # Por tipo (com base no GT) para texto
    by_type_text = {
        "velha":       {"total": 0, "infer": 0, "acc7": 0, "acc6": 0},
        "nova":        {"total": 0, "infer": 0, "acc7": 0, "acc6": 0},
        "desconhecida":{"total": 0, "infer": 0, "acc7": 0, "acc6": 0},
    }

    # Acurácia de classificação de tipo (predição vs GT)
    type_cls = {"total": 0, "hits": 0, "miss": 0, "none": 0}

    files = sorted(os.listdir(images_dir))
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        image_path = os.path.join(images_dir, img_name)

        gt_text = read_ground_truth(image_path)
        if gt_text is None:
            print(f"Ground truth não encontrado para {img_name}, pulando...")
            continue

        gt_type = classify_plate_type_gt(gt_text)
        by_type_text[gt_type]["total"] += 1

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Falha ao ler {image_path}, pulando...")
            continue

        if USE_VEHICLE_ROI:
            vehicle_roi, vehicle_bbox = _vehicle_roi(img)
            inp = vehicle_roi
        else:
            vehicle_roi = img.copy()
            vehicle_bbox = None
            inp = image_path  # caminho

        start_time = time.time()
        pred = model.predict([inp])[0]
        end_time = time.time()

        mediumTime += (end_time - start_time)
        total += 1

        raw_text = pred.get("text", "")
        pred_text = clean_plate_text(raw_text)
        pred_cls  = pred.get("cls", None)
        source    = pred.get("source")
        conf      = pred.get("conf")
        plate_bbox = pred.get("bbox")  # bbox da placa no MESMO espaço de coord. que 'inp'

        if pred_text:
            totalImagesInference += 1
            by_type_text[gt_type]["infer"] += 1

        acertos = check_match(pred_text, gt_text)
        if pred_text:
            if acertos == 7:
                sevenCorrected += 1
                by_type_text[gt_type]["acc7"] += 1
            elif acertos == 6:
                sixCorrected += 1
                by_type_text[gt_type]["acc6"] += 1

        # Métrica de tipo (apenas quando GT é velha/nova)
        if gt_type in ("velha", "nova"):
            if pred_cls is None:
                type_cls["none"] += 1
            else:
                type_cls["total"] += 1
                if pred_cls == gt_type:
                    type_cls["hits"] += 1
                else:
                    type_cls["miss"] += 1

        print(f"CORRETO(GT): {gt_text} [{gt_type}] | "
              f"PREDICT: '{raw_text}' (clean='{pred_text}', cls={pred_cls}) | "
              f"Acertos: {acertos}/7 | "
              f"source={source} conf={conf} bbox_inp={plate_bbox}")

        # Define quais bboxes vão em cada imagem:
        # - imagem completa: só bbox_vehicle
        # - ROI: bbox_plate (coordenadas do ROI)
        if vehicle_bbox is None:
            # caso sem ROI: se quiser, poderia usar plate_bbox_full aqui
            bbox_plate_full = plate_bbox
            bbox_plate_roi  = None
        else:
            bbox_plate_full = None
            bbox_plate_roi  = plate_bbox

        _save_debug_images(
            base_name=base_name,
            img_bgr=img,
            vehicle_roi=vehicle_roi,
            bbox_vehicle=vehicle_bbox,
            bbox_plate_full=bbox_plate_full,
            bbox_plate_roi=bbox_plate_roi,
            gt_text=gt_text,
            gt_type=gt_type,
            pred_text=pred_text,
            pred_cls=pred_cls,
            source=source,
            conf=conf,
            acertos=acertos
        )

        if DEBUG_VISUALIZE and not USE_VEHICLE_ROI:
            cv2.imshow("original", img)
            cv2.waitKey(200)

    # ------------------- Relatório final -------------------
    print("\n------------------- Final Results ------------------")
    print("Total Images: ", total)
    print("Total Images Inference: ", totalImagesInference)
    print("--------------------------------------------------")
    print("Resumo (TEXTO):")
    p7 = round(sevenCorrected / totalImagesInference * 100, 2) if totalImagesInference else 0
    p6 = round(sixCorrected   / totalImagesInference * 100, 2) if totalImagesInference else 0
    print(f"  7 Acertos: {sevenCorrected} ({p7}%)")
    print(f"  6 Acertos: {sixCorrected} ({p6}%)")
    if totalImagesInference > 0:
        print(f"Média geral das imagens inferidas [{totalImagesInference}]: "
              f"{(sevenCorrected+sixCorrected)/(totalImagesInference)*100:.2f}%")
    if total > 0:
        print(f"Média geral das imagens [{total}]: "
              f"{(sevenCorrected+sixCorrected)/(total)*100:.2f}%")
        print(f"Tempo Médio por Imagem: {round(mediumTime / total, 4)}s")
    print("--------------------------------------------------")
    print("Detalhe por TIPO (baseado no GT) – TEXTO:")
    for t in ("velha", "nova", "desconhecida"):
        bt = by_type_text[t]
        if bt["infer"] > 0:
            perc_infer = (bt["acc7"] + bt["acc6"]) / bt["infer"] * 100.0
        else:
            perc_infer = 0.0
        if bt["total"] > 0:
            perc_total = (bt["acc7"] + bt["acc6"]) / bt["total"] * 100.0
        else:
            perc_total = 0.0
        print(f"  {t}: total={bt['total']}, infer={bt['infer']}, "
              f"7ac={bt['acc7']}, 6ac={bt['acc6']}, "
              f"acc(infer)={perc_infer:.2f}%, acc(total)={perc_total:.2f}%")
    print("--------------------------------------------------")
    print("Classificação de TIPO (velha × nova):")
    print(f"  Casos com GT velha/nova e predição de tipo feita: {type_cls['total']}")
    print(f"  Acertos: {type_cls['hits']} | Erros: {type_cls['miss']} | None: {type_cls['none']}")
    if type_cls["total"] > 0:
        type_acc = type_cls["hits"] / type_cls["total"] * 100.0
    else:
        type_acc = 0.0
    print(f"  Acurácia de tipo: {type_acc:.2f}%")
    print("--------------------------------------------------")

# ==========================
# Execução
# ==========================
if __name__ == "__main__":
    images_dir = "PODI-LPR-01/"   # ajuste para o caminho correto
    model = ALPR_ATA(type_conf_threshold=0.60)
    evaluate_all(images_dir, model)
