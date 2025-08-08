import pytesseract

class OCRTesseract:
    def __init__(self):
        pass

    def predict(self, image: str) -> str:
        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(image, config=config)

        return text.strip().replace(" ", "").replace("\n", "")
