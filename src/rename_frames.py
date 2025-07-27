import os
import cv2

images = "results/frames2"

contador = 1100

for nome_antigo in sorted(os.listdir(images)):
    if not nome_antigo.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    caminho_antigo = os.path.join(images, nome_antigo)

    # Novo nome
    novo_nome = f"imagem_{contador:04}.jpg"
    caminho_novo = os.path.join(images, novo_nome)

    # Leitura e salvamento padronizado
    img = cv2.imread(caminho_antigo)
    if img is None:
        continue
    cv2.imwrite(caminho_novo, img)

    contador += 1
