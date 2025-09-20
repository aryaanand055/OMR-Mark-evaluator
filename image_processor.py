# image_processor.py

import numpy as np
import cv2


def process_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return "Could not decode image."
    return f"Image shape (OpenCV): {img.shape}"

def process_ocr_sheet(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    # Dummy logic: find contours and draw circles around them
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > 10:  # Only draw circles for larger contours
            cv2.circle(img, center, radius, (0, 255, 0), 2)
    # Encode image back to bytes
    _, img_encoded = cv2.imencode('.jpg', img)
    return img_encoded.tobytes()
