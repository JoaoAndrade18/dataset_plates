from AppFastALPR import FastALPR
import os

alpr_system = FastALPR()
path = "PODI-LPR-01/"
image_paths = []

for img in os.listdir(path):
    if img.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_paths.append(os.path.join(path, img))

results = alpr_system.process_images(image_paths[:5])
print(f"Processed {image_paths[:5]}.")
print(f"Detected Text: {results}")