import os

from pipeline_OCR import PipelineOCR

images_path = os.listdir("amostras_ufpr/")

pipeline = PipelineOCR()

for image in images_path:
    if image.lower().endswith(('.png', '.jpg', '.jpeg')):
        image = os.path.join("amostras_ufpr/", image)
        result = pipeline.run(image)
        print(result)

        