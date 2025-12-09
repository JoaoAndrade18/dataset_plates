import os
import requests
import time
import csv

URL = "http://201.148.100.122:5000/recognizer-dataset"
IMAGE_FOLDER = "RodoSol-ALPR/images/motorcycles-br/"
RESULTS_FILE = "resultUltimateALPR_RODOSOL_motorcycles.csv" 
FIELD_NAME = "image" 
HEADERS = {
    "accept": "application/json"
}

total = 0
total_inference = 0
total_nulo = 0
sevenCorrected = 0     
sevenCorrectedFixed = 0 
sixCorrected = 0       
sixCorrectedFixed = 0   
times = []

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
    correct = sum(p == t for p, t in zip(str(pred).upper(), str(expected).upper()))
    return correct

all_files = os.listdir(IMAGE_FOLDER)
# Filtra apenas arquivos de imagem
image_list = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

with open(RESULTS_FILE, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['image', 'true_plate', 'predicted_plate', 'q_char_corrected']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for filename in image_list:
        # Obtém o nome base sem extensão
        base_name = os.path.splitext(filename)[0]
        txt_name = base_name + '.txt'
        
        filepath = os.path.join(IMAGE_FOLDER, filename)
        txt_path = os.path.join(IMAGE_FOLDER, txt_name)
        
        # Lê o ground truth do arquivo .txt
        expected = read_ground_truth(txt_path)
        
        if expected is None:
            print(f"Ground truth não encontrado para {filename}, pulando...")
            continue

        plate = ""
        acertos = 0
        prediction_adjusted_old = ""
        prediction_adjusted_new = ""

        total += 1

        with open(filepath, "rb") as f:
            files = {FIELD_NAME: (filename, f, "image/jpeg")}
            start = time.time()
            try:
                response = requests.post(URL, files=files, headers=HEADERS, timeout=20)
            except requests.exceptions.RequestException as e:
                print(f"ERRO ao conectar ({filename}): {e}")
                continue
            elapsed = time.time() - start
            times.append(elapsed)

        prediction_raw = ""

        if response.status_code == 200:
            try:
                data = response.json()
                plates = data.get("plates", [])
                if plates:
                    prediction_raw = plates[0].get("text", "")[:7].upper() 
            except Exception as e:
                print(f"ERRO ao processar JSON ({filename}): {e}")
                continue

            if not prediction_raw:
                total_nulo += 1
            else:
                total_inference += 1

            prediction_adjusted_old, prediction_adjusted_new = corrige_placa(prediction_raw)
            
            correct_raw = check_match(prediction_raw, expected)
            correct_adjusted_old = check_match(prediction_adjusted_old, expected)
            correct_adjusted_new = check_match(prediction_adjusted_new, expected)

            if correct_adjusted_new > correct_adjusted_old:
                best_adjusted_plate = prediction_adjusted_new
                correct_adjusted = correct_adjusted_new
            else:
                best_adjusted_plate = prediction_adjusted_old
                correct_adjusted = correct_adjusted_old

            if correct_raw == 7:
                sevenCorrected += 1
                plate = prediction_raw
                acertos = correct_raw
            elif correct_adjusted == 7:
                sevenCorrectedFixed += 1
                plate = best_adjusted_plate
                acertos = correct_adjusted
            elif correct_raw == 6:
                sixCorrected += 1
                plate = prediction_raw
                acertos = correct_raw
            elif correct_adjusted == 6:
                sixCorrectedFixed += 1
                plate = best_adjusted_plate
                acertos = correct_adjusted
            else:
                plate = prediction_raw
                acertos = correct_raw

            writer.writerow({
                'image': filename,
                'true_plate': expected,
                'predicted_plate': plate,
                'q_char_corrected': acertos
            }) 
            print(f"{filename} | Esperado: {expected} | Predito: {prediction_raw} | Ajustado: {best_adjusted_plate} | Acertos: {acertos}/7 | Tempo: {elapsed:.3f}s")

        else:
            writer.writerow({
                'image': filename,
                'true_plate': expected,
                'predicted_plate': f"HTTP Error {response.status_code}",
                'q_char_corrected': 0
            })
            print(f"ERRO HTTP {response.status_code} para {filename}: {response.text[:200]}")

print("\n=== Resultados Finais ===")
print(f"Total de imagens processadas: {total}")
print(f"Imagens com detecção (inferência): {total_inference}")
print(f"Imagens sem detecção (nulo): {total_nulo}")
print("-" * 40)
print("Resultados Brutos (Sem Correção):")
print(f"  7/7 Acertos: {sevenCorrected} ({round(sevenCorrected / total_inference * 100, 2) if total_inference else 0}%)")
print(f"  6/7 Acertos: {sixCorrected} ({round(sixCorrected / total_inference * 100, 2) if total_inference else 0}%)")
print("\nResultados Corrigidos (Ajustados):")
print(f"  7/7 Acertos: {sevenCorrectedFixed} ({round(sevenCorrectedFixed / total_inference * 100, 2) if total_inference else 0}%)")
print(f"  6/7 Acertos: {sixCorrectedFixed} ({round(sixCorrectedFixed / total_inference * 100, 2) if total_inference else 0}%)")
print("-" * 40)

total_hits = sevenCorrected + sixCorrected + sevenCorrectedFixed + sixCorrectedFixed
print("Métricas Gerais:")
if total > 0:
    print(f"  Acurácia Geral (sobre TODAS as imagens): {round(total_hits / total * 100, 2)}%")
if total_inference > 0:
    print(f"  Acurácia de Inferência (sobre imagens com detecção): {round(total_hits / total_inference * 100, 2)}%")
if times:
    print(f"  Tempo médio por requisição: {sum(times)/len(times):.3f}s")