from infer_plate import InferPlate
import torch

if __name__ == "__main__":
    engine = InferPlate(
        csv_path="gold_images.csv",
        image_dir="frames_gold", 
        device="cuda" if torch.cuda.is_available() else "cpu",
        models=["yolo"], # Escolha entre "tesseract" e "yolo"
    )
    
    engine.run_evaluation()
