import os
import sys
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from AppFastALPR_tests import alpr_service

# Constants
RODOOSOL_PATH = "RodoSol-ALPR"
PODIALPR_PATH = "PODI-LPR-01"
FRAMES_GOLD_PATH = "frames_gold"
FRAMES_GOLD_CSV = "gold_images.csv"
YOLO_MODEL_PATH = "yolo11m.pt"


def get_rodoosol_data(split_selection, vehicle_selection):
    """
    Load RODOOSOL dataset based on split and vehicle type.
    split_selection: 'validation' or 'testing'
    vehicle_selection: 'cars', 'motorcycles', or 'both'
    """
    split_file = os.path.join(RODOOSOL_PATH, "split.txt")
    data = []
    
    print(f"Loading RODOOSOL data (Split: {split_selection}, Vehicle: {vehicle_selection})...")
    
    try:
        with open(split_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {split_file} not found.")
        return []

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(';')
        if len(parts) != 2:
            continue
            
        rel_image_path, split_tag = parts
        
        # Filter by split
        if split_tag != split_selection:
            continue
            
        # Filter by vehicle type
        # Path format: ./images/cars-br/img_000003.jpg
        if vehicle_selection != 'both':
            if vehicle_selection == 'cars' and 'cars' not in rel_image_path:
                continue
            if vehicle_selection == 'motorcycles' and 'motorcycles' not in rel_image_path:
                continue
        
        # Construct full paths
        if rel_image_path.startswith('./'):
            rel_image_path = rel_image_path[2:]
            
        full_image_path = os.path.join(RODOOSOL_PATH, rel_image_path)
        
        # Get Ground Truth from txt file
        txt_path = os.path.splitext(full_image_path)[0] + ".txt"
        
        gt_plate = ""
        try:
            with open(txt_path, 'r') as tf:
                for tline in tf:
                    if tline.startswith("plate:"):
                        gt_plate = tline.split(":")[1].strip()
                        break
        except FileNotFoundError:
            print(f"Warning: Label file not found for {full_image_path}")
            continue
            
        if gt_plate:
            data.append((full_image_path, gt_plate))
            
    return data


def get_podialpr_data(vehicle_selection):
    """
    Load PODIALPR dataset and filter by vehicle type using YOLOv11.
    vehicle_selection: 'cars', 'motorcycles', or 'both'
    """
    data = []
    print(f"Loading PODIALPR data (Vehicle: {vehicle_selection})...")
    
    if not os.path.exists(PODIALPR_PATH):
        print(f"Error: {PODIALPR_PATH} not found.")
        return []
        
    files = [f for f in os.listdir(PODIALPR_PATH) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Load YOLO model for classification
    print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
    try:
        yolo = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return []
        
    for f in files:
        full_image_path = os.path.join(PODIALPR_PATH, f)
        gt_plate = f[:7]  # First 7 chars are GT
        
        if vehicle_selection == 'both':
            data.append((full_image_path, gt_plate))
            continue
            
        # Run inference to classify
        results = yolo(full_image_path, verbose=False)
        
        # COCO classes: car=2, motorcycle=3, bus=5, truck=7
        is_car = False
        is_moto = False
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls in [2, 5, 7]:  # Car, Bus, Truck
                    is_car = True
                elif cls == 3:        # Motorcycle
                    is_moto = True
        
        if vehicle_selection == 'cars' and is_car:
            data.append((full_image_path, gt_plate))
        elif vehicle_selection == 'motorcycles' and is_moto:
            data.append((full_image_path, gt_plate))
            
    return data


def get_frames_gold_data():
    """
    Load FRAMES_GOLD dataset from CSV.
    CSV format (example):
    ID_image,bbox_car,class,bbox_plate,plate_car
    frame_10.jpg,"[552, 200, 1128, 646]",car,"(912, 509, 1036, 565)",SAP8J42

    Para o teste de acurácia, só precisamos de:
      - ID_image -> nome do arquivo
      - plate_car -> ground truth da placa
    """
    data = []
    print("Loading FRAMES_GOLD data (only cars)...")
    
    if not os.path.exists(FRAMES_GOLD_PATH):
        print(f"Error: {FRAMES_GOLD_PATH} not found.")
        return []
    
    if not os.path.exists(FRAMES_GOLD_CSV):
        print(f"Error: CSV file not found: {FRAMES_GOLD_CSV}")
        return []
    
    try:
        with open(FRAMES_GOLD_CSV, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                img_name = row.get('ID_image')
                gt_plate = row.get('plate_car')
                
                if not img_name or not gt_plate:
                    continue
                
                full_image_path = os.path.join(FRAMES_GOLD_PATH, img_name)
                if not os.path.exists(full_image_path):
                    print(f"Warning: image not found: {full_image_path}")
                    continue
                
                data.append((full_image_path, gt_plate))
    except Exception as e:
        print(f"Error reading CSV {FRAMES_GOLD_CSV}: {e}")
        return []
    
    print(f"Loaded {len(data)} annotated frames from FRAMES_GOLD.")
    return data


def calculate_metrics(results):
    """
    Calculate and print accuracy metrics.
    results: list of (gt_plate, predicted_plate)
    """
    total = len(results)
    if total == 0:
        print("No images processed.")
        return

    correct_7 = 0
    correct_6 = 0
    correct_1_to_5 = 0
    correct_0 = 0
    
    for gt, pred in results:
        # Clean strings
        gt = gt.upper().replace("-", "").strip()
        pred = pred.upper().replace("-", "").strip()
        
        # Calculate character matches
        matches = 0
        length = min(len(gt), len(pred))
        for i in range(length):
            if gt[i] == pred[i]:
                matches += 1
        
        if matches == 7 and len(gt) == 7 and len(pred) == 7:
            correct_7 += 1
        elif matches == 6:
            correct_6 += 1
        elif matches == 0:
            correct_0 += 1
        else:
            correct_1_to_5 += 1
            
    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    print(f"Total Images: {total}")
    print(f"7 matches (100%): {correct_7} ({correct_7/total*100:.2f}%)")
    print(f"6 matches:        {correct_6} ({correct_6/total*100:.2f}%)")
    print(f"1-5 matches:      {correct_1_to_5} ({correct_1_to_5/total*100:.2f}%)")
    print(f"0 matches:        {correct_0} ({correct_0/total*100:.2f}%)")


def main():
    print("ALPR Accuracy Test Tool")
    print("1. RODOOSOL")
    print("2. PODIALPR")
    print("3. FRAMES_GOLD")
    
    ds_choice = input("Select Dataset (1/2/3): ").strip()
    
    data = []
    
    if ds_choice == '1':
        print("\nSelect Split:")
        print("1. Validation")
        print("2. Testing")
        split_choice = input("Choice (1/2): ").strip()
        split = 'validation' if split_choice == '1' else 'testing'
        
        print("\nSelect Vehicle Type:")
        print("1. Cars")
        print("2. Motorcycles")
        print("3. Both")
        veh_choice = input("Choice (1/2/3): ").strip()
        vehicle = 'cars' if veh_choice == '1' else ('motorcycles' if veh_choice == '2' else 'both')
        
        data = get_rodoosol_data(split, vehicle)
        
    elif ds_choice == '2':
        print("\nSelect Vehicle Type (Filtered by YOLOv11):")
        print("1. Cars")
        print("2. Motorcycles")
        print("3. Both")
        veh_choice = input("Choice (1/2/3): ").strip()
        vehicle = 'cars' if veh_choice == '1' else ('motorcycles' if veh_choice == '2' else 'both')
        
        data = get_podialpr_data(vehicle)
    
    elif ds_choice == '3':
        # FRAMES_GOLD: só carros
        vehicle = 'cars'
        data = get_frames_gold_data()
        
    else:
        print("Invalid choice.")
        return

    if not data:
        print("No data found matching criteria.")
        return
        
    print(f"\nStarting inference on {len(data)} images...")
    
    # Initialize ALPR
    alpr_system = alpr_service
    
    # Initialize Vehicle Detector
    print(f"Loading Vehicle Detector from {YOLO_MODEL_PATH}...")
    try:
        vehicle_detector = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"Error loading Vehicle Detector: {e}")
        return
    
    results = []
    
    for i, (img_path, gt) in enumerate(data):
        print(f"Processing {i+1}/{len(data)}: {os.path.basename(img_path)} (GT: {gt})", end='\r')
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"\nError reading image: {img_path}")
                results.append((gt, ""))
                continue

            # Detect vehicle with YOLO
            detections = vehicle_detector(img, verbose=False)
            
            best_crop = None
            max_area = 0
            
            # COCO: 2=car, 3=motorcycle, 5=bus, 7=truck
            target_classes = []
            if vehicle == 'cars':
                target_classes = [2, 5, 7]
            elif vehicle == 'motorcycles':
                target_classes = [3]
            else:
                target_classes = [2, 3, 5, 7]

            for r in detections:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls in target_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            h, w, _ = img.shape
                            x1 = max(0, x1)
                            y1 = max(0, y1)
                            x2 = min(w, x2)
                            y2 = min(h, y2)
                            best_crop = img[y1:y2, x1:x2]

            # Pass crop or full image if no vehicle detected
            if best_crop is not None:
                input_img = best_crop
            else: 
                input_img = img
                
            # ALPR_ATA.predict aceita lista de imagens (ou paths)
            preds = alpr_system.predict([input_img])
            pred = preds[0] if preds else ""
            print("3", pred)
            pred = pred.get('text', '')
            results.append((gt, pred))

            # save imagem with annotated results
            # cv2.imwrite(f"results/{os.path.basename(img_path)}", in-put_img)
            # Draw prediction on the image before saving
            # if pred:
            #     annotated_img = input_img.copy()
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     font_scale = 1
            #     font_thickness = 2
            #     text_color = (0, 255, 0)  # Green color for text
            #     text_position = (10, 30)  # Top-left corner
            #     cv2.putText(annotated_img, pred, text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            #     cv2.imwrite(f"results/{os.path.basename(img_path)}", annotated_img)
            # else:
            #     cv2.imwrite(f"results/{os.path.basename(img_path)}", input_img)

            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            results.append((gt, ""))
            
    print("\nProcessing complete.")
    calculate_metrics(results)


if __name__ == "__main__":
    main()
