import cv2, os, time
import csv 
from fast_alpr import ALPR
from ultralytics import YOLO

from AppFastALPR import FastALPR

RESULTS_FILE = "resultFastALPR_RODOSOL_MERCOSUL.csv"

def clean_plate_text(text):
    """Remove caracteres não alfanuméricos e converte para maiúsculas."""
    return ''.join(c for c in text.upper() if c.isalnum())

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

def corrige_placa(text):
    text = clean_plate_text(text)
    if len(text) < 7:
        return text, text
    def is_letra(ch): return ch.isalpha()
    def is_num(ch): return ch.isdigit()
    padrao_antigo = [is_letra, is_letra, is_letra, is_num, is_num, is_num, is_num]
    padrao_novo =   [is_letra, is_letra, is_letra, is_num, is_letra, is_num, is_num]
    def aplica_padrao(padrao, chars):
        corr = []
        for ch, f in zip(chars, padrao):
            if f(ch):
                corr.append(ch)
            else:
                if f == is_letra:
                    mapa = {'0':'O', '1':'I', '2':'Z', '3':'E', '4':'A', '5':'S', '6':'G', '7':'T', '8':'B', '9':'B'}
                    corr.append(mapa.get(ch, ch))
                else:
                    mapa = {'O':'0', 'Q':'0', 'D':'0', 'I':'1', 'Z':'2', 'A':'4', 'S':'5', 'G':'6', 'T':'7', 'B':'8', 'E':'3', 'C':'6', 'Y':'4',
                            'L':'1', 'F':'5', 'H':'4', 'X':'8', 'J':'1', 'K':'6', 'M':'1', 'N':'1', 'P':'9', 'R':'2', 'U':'0', 'V':'8', 'W':'3'}
                    corr.append(mapa.get(ch, ch))
        return "".join(corr)
    
    placa_antiga = aplica_padrao(padrao_antigo, text[:7])
    placa_nova = aplica_padrao(padrao_novo, text[:7])

    return placa_antiga, placa_nova

def check_match(pred, expected):
    """Retorna o número de caracteres corretos."""
    pred = str(pred).upper()
    expected = str(expected).upper()
    correct = sum(p == t for p, t in zip(pred, expected))
    return correct

alpr = ALPR(
    detector_model="yolo-v9-s-608-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model",
)
yolo_model = YOLO('best_placa.pt').to(device=0)

total = 0
mediumTime = 0
sevenCorrected = 0
sevenCorrectedFixed = 0
sixCorrected = 0
sixCorrectedFixed = 0
totalImagesInference = 0 

path = "RodoSol-ALPR/images/cars-me/"
all_files = os.listdir(path)

images = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['image', 'true_plate', 'predicted_plate', 'q_char_corrected']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + '.txt'
        
        image_path = os.path.join(path, img_name)
        txt_path = os.path.join(path, txt_name)
        
        ground_truth_plate = read_ground_truth(txt_path)
        
        if ground_truth_plate is None:
            print(f"Ground truth não encontrado para {img_name}, pulando...")
            continue
        
        start_time = time.time()
        
        alpr_results = alpr.predict(image_path)

        if not alpr_results:
            try:
                yolo_results = yolo_model(image_path, verbose=False)
                if yolo_results and len(yolo_results[0].boxes) > 0:
                    image = cv2.imread(image_path)
                    box = yolo_results[0].boxes[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    imageRoi = image[y1:y2, x1:x2]
                    alpr_results = alpr.predict(imageRoi)
            except Exception as e:
                continue

        end_time = time.time()
        mediumTime += end_time - start_time
        total += 1
    
        resultText_clear = alpr_results[0].ocr.text.upper() if alpr_results and alpr_results[0].ocr and alpr_results[0].ocr.text else ""
        
        resultText_clear = clean_plate_text(resultText_clear)

        if resultText_clear:
            totalImagesInference += 1
        
        corrected_old, corrected_new = corrige_placa(resultText_clear)
        
        acertos_bruto = check_match(resultText_clear, ground_truth_plate)
        acertos_corrigido_old = check_match(corrected_old, ground_truth_plate)
        acertos_corrigido_new = check_match(corrected_new, ground_truth_plate)

        if acertos_corrigido_new > acertos_corrigido_old:
            best_corrected_plate = corrected_new
            acertos_corrigido = acertos_corrigido_new
        else:
            best_corrected_plate = corrected_old
            acertos_corrigido = acertos_corrigido_old
        
        plate = ""
        acertos = 0

        if resultText_clear:
            if acertos_bruto == 7:
                sevenCorrected += 1
                plate = resultText_clear
                acertos = acertos_bruto 
            elif acertos_corrigido == 7:
                sevenCorrectedFixed += 1
                plate = best_corrected_plate
                acertos = acertos_corrigido
            elif acertos_bruto == 6:
                sixCorrected += 1
                plate = resultText_clear
                acertos = acertos_bruto
            elif acertos_corrigido == 6:
                sixCorrectedFixed += 1
                plate = best_corrected_plate
                acertos = acertos_corrigido
            else:
                plate = resultText_clear
                acertos = acertos_bruto
        
        writer.writerow({
            'image': img_name,
            'true_plate': ground_truth_plate,
            'predicted_plate': plate,
            'q_char_corrected': acertos,
        })
        
        print(f"CORRETO: {ground_truth_plate} | PREDICT: {resultText_clear} | [ADJUSTED] PREDICT: {best_corrected_plate} | Acertos: {acertos}/{7}")

print("\n------------------- Final Results ------------------")
print("Total Images: ", total)
print("Total Images Inference: ", totalImagesInference)
print("--------------------------------------------------")
print("Sem Correção (Bruto):")
print(f"  7 Acertos: {sevenCorrected} ({round(sevenCorrected / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
print(f"  6 Acertos: {sixCorrected} ({round(sixCorrected / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
print("Com Correção (adjusted):")
print(f"  7 Acertos: {sevenCorrectedFixed} ({round(sevenCorrectedFixed / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
print(f"  6 Acertos: {sixCorrectedFixed} ({round(sixCorrectedFixed / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
print("--------------------------------------------------")
print(f"Média geral das imagens inferidas [{totalImagesInference}]: {(sevenCorrected+sixCorrected+sevenCorrectedFixed+sixCorrectedFixed)/(totalImagesInference)*100:.2f}%")
print(f"Média geral das imagens [{total}]: {(sevenCorrected+sixCorrected+sevenCorrectedFixed+sixCorrectedFixed)/(total)*100:.2f}%")
if total > 0:
    print(f"Tempo Médio por Imagem: {round(mediumTime / total, 4)}s")
print("--------------------------------------------------")