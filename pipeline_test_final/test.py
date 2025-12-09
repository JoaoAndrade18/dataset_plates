from ultralytics import YOLO # --- Optional ---
import cv2, os

model = YOLO('best_placa.pt')

images = "RodoSol-ALPR/images/cars-me/"

images_old = []
aux = 0

for image in os.listdir(images):
    if image.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(images, image)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro ao ler imagem: {image_path}")
            continue
        
        # Processa imagem
        results = model(image)[0]

        box = results.boxes
        try:
            cls_ = int(box.cls[0])     # 4 = velha, 5 = nova
            conf = float(box.conf[0])    # confiança da detecção
            name = results.names[cls_]  
        except Exception as e:
            continue

        print(f"Classe: {name}, Confiança: {conf:.2f}")
        if cls_ != 5 and conf < 0.6:
            images_old.append([image_path, conf])

        aux += 1
        if aux > 1000:
            break

print("\nimages old: ", images_old)