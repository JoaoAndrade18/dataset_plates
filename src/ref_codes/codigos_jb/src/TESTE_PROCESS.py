import cv2
import numpy as np
import math
from PIL import Image, ImageEnhance
import pytesseract
import os

def rotate_image(image, angle):
    """
    Rotaciona a imagem em torno de seu centro pelo ângulo especificado.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def compute_skew(image):
    """
    Calcula o ângulo de inclinação (skew) da imagem com base em detecção de bordas e linhas.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # -- Filtro para reduzir ruído --
    blur = cv2.medianBlur(gray, 5)

    # -- Detecção de bordas --
    edges = cv2.Canny(blur, threshold1=100, threshold2=150, apertureSize=5, L2gradient=True)

    # -- Detecção de linhas -- 
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=gray.shape[1] // 2, maxLineGap=30)

    if lines is None:
        return 0.0 

    angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = angle_rad * 180.0 / np.pi

        if -30 < angle_deg < 30:
            angles.append(angle_deg)

    if len(angles) == 0:
        return 0.0

    # media dos ângulos
    return np.mean(angles)

def deskew(image):
    """
    Corrige a inclinação da imagem automaticamente.
    """
    angle = compute_skew(image)
    return rotate_image(image, angle)

def upscale_image(image):
    """
    Aumenta a resolução da imagem de forma mais suave.
    """
    # -- Aplica (ESPCN) --
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("../../ESPCN_x3.pb")
    sr.setModel("espcn", 3)
    result = sr.upsample(image)
    
    return result

def recorte_somente_letras(image_path):
    # Binariza com Otsu
    _, binary = cv2.threshold(image_path, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontra contornos
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra contornos por área (letras costumam ter área intermediária)
    letras = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 1000:  # Ajuste conforme o tamanho da imagem/letras
            letras.append(cnt)

    if not letras:
        print("Nenhuma letra detectada.")
        return image_path

    # Cria a bounding box que envolve todos os contornos de letras
    todos_pontos = np.vstack(letras)
    x, y, w, h = cv2.boundingRect(todos_pontos)

    # Recorta a imagem
    cropped = image_path[y:y+h, x:x+w]

    return cropped

def remove_lines_and_noise(image):
    """
    Remove linhas e ruídos de forma mais suave, preservando melhor o texto.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # -- Suavização leve --
    blurred = cv2.GaussianBlur(gray, (7, 7), 10)
    
    # -- Binarização adaptativa mais suave --
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 19, 3)
    
    # -- Remover pequenos ruidos -- 
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    #cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary

def enhance_text(image):
    """
    Melhora a qualidade do texto de forma mais suave.
    """
    # -- Suavização leve para reduzir ruído no texto --
    denoised = cv2.medianBlur(image, 5)
    
    # -- Afinar leve os caracteres -- 
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)) # acima de 2 da erro
    #enhanced = cv2.dilate(denoised, kernel, iterations=2)
    
    return denoised

def full_process(image):
    """
    Processa completamente uma imagem de placa para OCR de forma mais suave.
    
    Sequência de processamento:
    1. Upscaling suave (2x)
    2. Deskew (correção de rotação)
    3. Remoção de bordas
    4. Limpeza suave de ruídos
    5. Melhoria leve de qualidade
    
    Args:
        image: Imagem de entrada (BGR ou Grayscale)
        
    Returns:
        Imagem processada otimizada para OCR
    """
    if image is None or image.size == 0:
        raise ValueError("Imagem inválida ou vazia")
    
    print("[INFO] Iniciando processamento suave da placa...")
    
    # -- UPSCALING SUAVE --
    print("[INFO] Aplicando upscaling suave com ESPCN (3x)...")
    processed = upscale_image(image) 

    # -- DESKEW --
    print("[INFO] Corrigindo rotação (deskew)...")
    processed = deskew(processed)
    
    # -- LIMPEZA SUAVE --
    print("[INFO] Limpeza suave de ruídos...")
    processed = remove_lines_and_noise(processed)
    
    # -- MELHORIA LEVE --
    print("[INFO] Melhoria leve da qualidade...")
    processed = enhance_text(processed)

    # -- REMOÇÃO DE BORDAS --
    #print("[INFO] Removendo bordas...")
    #processed = recorte_somente_letras(processed)
    
    print("[INFO] Processamento feito")
    return processed

# Função auxiliar para testar o processamento
def test_full_process(image_path, save_result=True):
    """
    Testa a função full_process com uma imagem específica.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro: Não foi possível carregar a imagem {image_path}")
        return None
    
    try:
        processed = full_process(image)
        
        if save_result: 
            output_path = image_path.replace('.', '_processed.')
            cv2.imwrite(output_path, processed)
            print(f"Imagem processada salva em: {output_path}")
        
        return processed
        
    except Exception as e:
        print(f"Erro durante processamento: {e}")
        return None

def process_folder_with_full_process(folder_path):
    """
    Aplica o processamento completo em todas as imagens de uma pasta.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"\nProcessando: {filename}")
            
            image = cv2.imread(image_path)
            processed = full_process(image)
            
            output_path = os.path.join(folder_path, f"processed_{filename}")
            cv2.imwrite(output_path, processed)
            
            print(f"Salvo: processed_{filename}")

test_full_process("original_plate_images/frame_10.jpg_plate.jpg", save_result=True) 