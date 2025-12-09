import cv2
import numpy as np

def upsampling(image):
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel("ESPCN_x3.pb")
    sr.setModel("espcn", 3)
    result_upsampling = sr.upsample(image)

    return result_upsampling

def skew_correction(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # blur = cv2.medianBlur(gray, 5)
    # edges = cv2.Canny(blur, threshold1=100, threshold2=150, apertureSize=5, L2gradient=True)
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=gray.shape[1] // 2, maxLineGap=30)

    # if lines is None:
    #     return 0.0 

    angles = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = angle_rad * 180.0 / np.pi

        if -30 < angle_deg < 30:
            angles.append(angle_deg)

    if len(angles) == 0:
        return 0.0
    
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return result

def noise_reduction(image):
    img_sem_ruido = cv2.bilateralFilter(img_cinza, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_contrastada = clahe.apply(img_sem_ruido)

    # 5. Binarização Adaptativa
    # Usamos THRESH_BINARY_INV para ter texto branco em fundo preto, que pode ajudar o Tesseract
    img_binaria = cv2.adaptiveThreshold(
        img_contrastada, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        ADAPTIVE_THRESH_BLOCK_SIZE, 
        ADAPTIVE_THRESH_C
    )

    return img_binaria
    return denoised_image

def run():
    pass