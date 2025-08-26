import cv2
import numpy as np
from scipy import ndimage
from skimage import restoration, exposure, morphology, transform
from skimage.filters import unsharp_mask

import warnings
warnings.filterwarnings('ignore')

def upsampling_super_resolution(image):
    """
    Aumenta a resolução da imagem usando ESPCN_x3.
    
    Args:
        image: imagem de entrada
    """
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("ESPCN_x3.pb")
    sr.setModel("espcn", 3)
    result_upsampling = sr.upsample(image)

    return result_upsampling

def deblurring_sharpening(image, method='unsharp_mask', strength=1.5):
    """
    Remove blur e aumenta nitidez da imagem
    
    Args:
        image: imagem de entrada
        method: 'unsharp_mask', 'laplacian'
        kernel_size: tamanho do kernel
        strength: intensidade do efeito
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'unsharp_mask':
        return (unsharp_mask(image, radius=1, amount=strength) * 255).astype(np.uint8)
    
    elif method == 'laplacian':
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpened = image - strength * laplacian
        return np.clip(sharpened, 0, 255).astype(np.uint8)

def contrast_enhancement(image, method='clahe', clip_limit=3.0):
    """
    Melhora o contraste da imagem
    
    Args:
        image: imagem de entrada
        method: 'clahe', 'histogram_eq', 'gamma'
        clip_limit: limite para CLAHE
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
        return clahe.apply(image)
    
    elif method == 'histogram_eq':
        return cv2.equalizeHist(image)

def noise_reduction(image, method='bilateral', kernel_size=5):
    """
    Remove ruído da imagem
    
    Args:
        image: imagem de entrada
        method: 'bilateral', 'gaussian', 'median', 'non_local_means'
        kernel_size: tamanho do kernel
    """
    if method == 'bilateral':
        return cv2.bilateralFilter(image, 9, 75, 75)
    
    elif method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)

def adaptive_binarization(image, method='adaptive', block_size=11, c=2):
    """
    Converte imagem para binário usando técnicas adaptativas
    
    Args:
        image: imagem de entrada (grayscale)
        method: 'adaptive', 'otsu', 'triangle'
        block_size: tamanho do bloco para adaptive
        c: constante subtraída da média
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'adaptive':
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, block_size, c)
    
    elif method == 'otsu':
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

def morphological_operations(image, operation='opening', kernel_size=3, iterations=1):
    """
    Aplica operações morfológicas para limpar a imagem
    
    Args:
        image: imagem binária de entrada
        operation: 'opening', 'closing', 'erosion', 'dilation'
        kernel_size: tamanho do elemento estruturante
        iterations: número de iterações
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    if operation == 'opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == 'erosion':
        return cv2.erode(image, kernel, iterations=iterations)

def perspective_correction(image, points=None):
    """
    Corrige distorção de perspectiva
    
    Args:
        image: imagem de entrada
        points: pontos da placa [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] ou None para detecção automática
    """
    if points is None:
        # Detecção automática de contornos retangulares
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Encontrar o maior contorno retangular
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                points = approx.reshape(4, 2)
                break
        else:
            return image  # Retorna original se não encontrar retângulo
    
    # Ordenar pontos: top-left, top-right, bottom-right, bottom-left
    points = np.array(points, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # top-left
    rect[2] = points[np.argmax(s)]  # bottom-right
    
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # top-right
    rect[3] = points[np.argmax(diff)]  # bottom-left
    
    # Calcular dimensões da saída
    width = max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))
    height = max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2]))
    
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
    
    # Aplicar transformação de perspectiva
    matrix = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, matrix, (int(width), int(height)))

def edge_enhancement(image, method='sobel', threshold1=50, threshold2=150):
    """
    Realça bordas dos caracteres
    
    Args:
        image: imagem de entrada
        method: 'sobel', 'canny', 'laplacian'
        threshold1, threshold2: thresholds para Canny
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if method == 'sobel':
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    
    elif method == 'canny':
        return cv2.Canny(image, threshold1, threshold2)

def histogram_stretching(image, low_percentile=1, high_percentile=99):
    """
    Expande o range dinâmico da imagem
    
    Args:
        image: imagem de entrada
        low_percentile: percentil inferior
        high_percentile: percentil superior
    """
    if len(image.shape) == 3:
        # Para imagens coloridas, aplicar em cada canal
        result = np.zeros_like(image)
        for i in range(3):
            channel = image[:, :, i]
            p_low, p_high = np.percentile(channel, (low_percentile, high_percentile))
            result[:, :, i] = exposure.rescale_intensity(channel, in_range=(p_low, p_high))
        return result
    else:
        # Para imagens em grayscale
        p_low, p_high = np.percentile(image, (low_percentile, high_percentile))
        return exposure.rescale_intensity(image, in_range=(p_low, p_high))

def preprocess_pipeline(image, techniques=['upsampling', 'deblurring', 'contrast', 'noise_reduction']):
    """
    Pipeline completo de pré-processamento
    
    Args:
        image: imagem de entrada
        techniques: lista de técnicas a aplicar em ordem
    """
    result = image.copy()
    
    for technique in techniques:
        if technique == 'upsampling':
            result = upsampling_super_resolution(result)
            cv2.imshow("Upsampled Image", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # elif technique == 'deblurring':
        #     result = deblurring_sharpening(result, method='laplacian')
        #     cv2.imshow("Deblurred Image", result)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        elif technique == 'contrast':
            result = contrast_enhancement(result, method='clahe')
            cv2.imshow("Contrast Enhanced Image", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # elif technique == 'noise_reduction':
        #     result = noise_reduction(result, method='gaussian')
        #     cv2.imshow("Noise Reduced Image", result)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        

    return result