import os, time
from AppFastALPR import ALPR_ATA

model = ALPR_ATA()

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

def check_match(pred, expected):
    """Retorna o número de caracteres corretos."""
    pred = str(pred).upper()
    expected = str(expected).upper()
    correct = sum(p == t for p, t in zip(pred, expected))
    return correct

total = 0
mediumTime = 0
sevenCorrected = 0
sixCorrected = 0
totalImagesInference = 0 

path = "RodoSol-ALPR/images/cars-br/"
all_files = os.listdir(path)

images = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

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

    alpr_results = model.predict([image_path])[0]

    end_time = time.time()
    mediumTime += end_time - start_time
    total += 1

    resultText_clear = alpr_results if alpr_results else ""
    resultText_clear = clean_plate_text(resultText_clear)

    if resultText_clear:
        totalImagesInference += 1
    
    plate = ""
    acertos = 0
    acertos = check_match(resultText_clear, ground_truth_plate)

    if resultText_clear:
        if acertos== 7:
            sevenCorrected += 1
            plate = resultText_clear
        elif acertos== 6:
            sixCorrected += 1
            plate = resultText_clear

    print(f"CORRETO: {ground_truth_plate} | PREDICT: {alpr_results} | Acertos: {acertos}/{7}")

print("\n------------------- Final Results ------------------")
print("Total Images: ", total)
print("Total Images Inference: ", totalImagesInference)
print("--------------------------------------------------")
print("Resumo:")
print(f"  7 Acertos: {sevenCorrected} ({round(sevenCorrected / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
print(f"  6 Acertos: {sixCorrected} ({round(sixCorrected / totalImagesInference * 100, 2) if totalImagesInference else 0}%)")
print("--------------------------------------------------")
print(f"Média geral das imagens inferidas [{totalImagesInference}]: {(sevenCorrected+sixCorrected)/(totalImagesInference)*100:.2f}%")
print(f"Média geral das imagens [{total}]: {(sevenCorrected+sixCorrected)/(total)*100:.2f}%")
if total > 0:
    print(f"Tempo Médio por Imagem: {round(mediumTime / total, 4)}s")
print("--------------------------------------------------")