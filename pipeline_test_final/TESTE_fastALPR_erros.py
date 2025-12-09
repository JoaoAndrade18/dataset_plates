import os, time
import cv2
from AppFastALPR_tests import ALPR_ATA

model = ALPR_ATA()

def clean_plate_text(text):
    """Remove caracteres não alfanuméricos e converte para maiúsculas."""
    return ''.join(c for c in str(text).upper() if c.isalnum())

def read_ground_truth(txt_path):
    """Lê o arquivo .txt e extrai a placa ground truth."""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('plate:'):
                    plate = line.split('plate:')[1].strip()
                    return clean_plate_text(plate)
    except Exception as e:
        print(f"Erro ao ler {txt_path}: {e}")
    return None

def check_match(pred, expected):
    """Retorna o número de caracteres corretos (posição exata na string)."""
    pred = str(pred).upper()
    expected = str(expected).upper()
    return sum(p == t for p, t in zip(pred, expected))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def draw_annotated(img, pred_text="", bbox=None, plate_found=False, source="none"):
    """Desenha bbox (se houver) e o texto/aviso + source."""
    out = img.copy()

    # Texto de status
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Pred: {pred_text} | Src: {source}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(0, y1 - 8)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 4, ty), (0, 0, 0), -1)
        cv2.putText(out, label, (x1 + 2, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    else:
        if plate_found:
            # Houve texto (fast detectou), mas sem bbox disponível -> só escreve o texto + source
            if pred_text:
                label = f"Pred: {pred_text} | Src: {source}"
                cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            # Nem texto nem bbox -> plate not found
            label = f"Plate not found | Src: {source}"
            cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2, cv2.LINE_AA)
    return out

# ==================== CONFIGURAÇÕES ====================
path = "RodoSol-ALPR/images/motorcycles-br/"

output_root = "erros_alpr_br"
dir_6 = os.path.join(output_root, "6_acertos")
dir_1_5 = os.path.join(output_root, "1_a_5_acertos")
dir_0 = os.path.join(output_root, "0_acertos")
dir_sem_marcacao = os.path.join(output_root, "sem_marcacao")  # só quando não há texto e bbox None

for d in [output_root, dir_6, dir_1_5, dir_0, dir_sem_marcacao]:
    ensure_dir(d)

# Se quiser separar por source dentro de cada pasta, descomente:
# for parent in [dir_6, dir_1_5, dir_0, dir_sem_marcacao]:
#     for src in ["fast", "yolo", "none"]:
#         ensure_dir(os.path.join(parent, f"src_{src}"))
# =======================================================

total = 0
mediumTime = 0.0
sevenCorrected = 0
sixCorrected = 0
totalImagesInference = 0

all_files = os.listdir(path)
images = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for img_name in images:
    base_name = os.path.splitext(img_name)[0]
    ext = os.path.splitext(img_name)[1]
    txt_name = base_name + '.txt'

    image_path = os.path.join(path, img_name)
    txt_path = os.path.join(path, txt_name)
    ground_truth_plate = read_ground_truth(txt_path)

    if ground_truth_plate is None:
        print(f"Ground truth não encontrado para {img_name}, pulando...")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"Não foi possível ler a imagem {image_path}, pulando...")
        continue

    start_time = time.time()
    model_out = model.predict([image_path])[0]  # {'text','bbox','conf','source'}
    end_time = time.time()

    mediumTime += end_time - start_time
    total += 1

    pred_text_raw = model_out.get("text", "") or ""
    bbox = model_out.get("bbox", None)
    source = model_out.get("source", "none")

    resultText_clear = clean_plate_text(pred_text_raw) if pred_text_raw else ""
    plate_found = bool(resultText_clear)  # há texto -> consideramos placa detectada
    if resultText_clear:
        totalImagesInference += 1

    acertos = check_match(resultText_clear, ground_truth_plate)

    # métricas originais
    if resultText_clear:
        if acertos == 7:
            sevenCorrected += 1
        elif acertos == 6:
            sixCorrected += 1

    print(f"CORRETO: {ground_truth_plate} | PREDICT: {pred_text_raw} | Acertos: {acertos}/7 | BBox: {bbox} | Source: {source}")

    # Salvar SOMENTE FALHAS (<7 acertos)
    if acertos < 7:
        if not plate_found and bbox is None:
            # Sem texto e sem bbox -> sem_marcacao
            out_dir = dir_sem_marcacao
        else:
            # houve texto OU bbox -> classifica por acertos
            if acertos == 6:
                out_dir = dir_6
            elif acertos >= 1:
                out_dir = dir_1_5
            else:
                out_dir = dir_0

        annotated = draw_annotated(img, pred_text=resultText_clear, bbox=bbox, plate_found=plate_found, source=source)

        # Inclui o source no nome do arquivo salvo
        out_name = f"{base_name}__src-{source}{ext}"
        out_img_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_img_path, annotated)

        # Se quiser separar por source em subpastas, use isto em vez das duas linhas acima:
        # out_dir_src = os.path.join(out_dir, f"src_{source}")
        # ensure_dir(out_dir_src)
        # out_img_path = os.path.join(out_dir_src, img_name)
        # cv2.imwrite(out_img_path, annotated)

# Prints finais
print("\n------------------- Final Results ------------------")
print("Total Images: ", total)
print("Total Images Inference: ", totalImagesInference)
print("--------------------------------------------------")
print("Resumo:")
print(f"  7 Acertos: {sevenCorrected} ({round(sevenCorrected / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
print(f"  6 Acertos: {sixCorrected} ({round(sixCorrected / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
print("--------------------------------------------------")
if totalImagesInference > 0:
    print(f"Média geral das imagens inferidas [{totalImagesInference}]: {(sevenCorrected+sixCorrected)/(totalImagesInference)*100:.2f}%")
print(f"Média geral das imagens [{total}]: {(sevenCorrected+sixCorrected)/(total)*100:.2f}%")
if total > 0:
    print(f"Tempo Médio por Imagem: {round(mediumTime / total, 4)}s")
print("--------------------------------------------------")