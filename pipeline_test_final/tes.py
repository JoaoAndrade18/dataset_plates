import cv2
import numpy as np

img_path = "RodoSol-ALPR/images/cars-br/img_000001.jpg"
corners_str = "558,438 687,439 687,482 558,481"
plate = "ODE2510"

# Parse dos pontos
pts = np.array([[int(x), int(y)] for x,y in (p.split(",") for p in corners_str.split())], dtype=np.int32)

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(img_path)

# Desenhar polígono (retângulo) e pontos
cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
for (x,y) in pts:
    cv2.circle(img, (x,y), 3, (0,0,255), -1)

# Opcional: escrever a placa
cv2.putText(img, plate, (pts[0][0], max(0, pts[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

# Exibir (em notebooks, converta BGR->RGB)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6)); plt.imshow(img_rgb); plt.axis("off"); plt.show()