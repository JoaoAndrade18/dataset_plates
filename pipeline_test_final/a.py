from AppFastALPR import FastALPR
import os
import cv2
import numpy as np

def resize_to_360p(image_path):
    """Redimensiona a imagem para 640x360 (360p) mantendo a proporção."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Calcula proporção para manter aspect ratio
    height, width = img.shape[:2]
    target_width = 640
    target_height = 360
    
    # Redimensiona mantendo proporção
    ratio = min(target_width/width, target_height/height)
    new_size = (int(width * ratio), int(height * ratio))
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    
    # Cria canvas 640x360 preto
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Centraliza imagem redimensionada
    y_offset = (target_height - new_size[1]) // 2
    x_offset = (target_width - new_size[0]) // 2
    canvas[y_offset:y_offset+new_size[1], x_offset:x_offset+new_size[0]] = resized
    
    return canvas

alpr_system = FastALPR()
path = "PODI-LPR-01/"
image_paths = []

for img in os.listdir(path):
    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_paths.append(os.path.join(path, img))

# Processa apenas as 5 primeiras imagens
selected_paths = image_paths[:5]
processed_images = []

# Redimensiona cada imagem para 360p
for img_path in selected_paths:
    resized = resize_to_360p(img_path)
    if resized is not None:
        # Salva temporariamente
        temp_path = f"temp_{os.path.basename(img_path)}"
        cv2.imwrite(temp_path, resized)
        processed_images.append(temp_path)
    else:
        print(f"Erro ao ler imagem: {img_path}")

# Processa imagens redimensionadas
if processed_images:
    results = alpr_system.process_images(processed_images)
    print(f"Processadas: {processed_images}")
    print(f"Textos detectados: {results}")
    
    # Remove arquivos temporários
    for temp_file in processed_images:
        try:
            os.remove(temp_file)
        except Exception as e:
            print(f"Erro ao remover arquivo temporário {temp_file}: {e}")
else:
    print("Nenhuma imagem foi processada com sucesso.")