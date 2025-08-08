import pytesseract
import cv2

def predict(image) -> str:
        """  """
        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(image, config=config)

        print(text)


# open image
image = cv2.imread("original_plate_images/frame_25_processed.jpg_plate_processed.jpg")
predict(image)

