import cv2
import numpy as np
import pandas as pd

# ------------------------------
# Step 1: Preprocessing
# ------------------------------
def preprocess_omr(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Find contours (sheet boundary)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    doc_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx
            break

    if doc_cnt is None:
        raise Exception("OMR sheet not detected.")

    # Order points
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect = order_points(doc_cnt.reshape(4, 2))
    dst = np.array([[0,0],[600,0],[600,800],[0,800]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (600,800))

    # Illumination correction (CLAHE)
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray_warped)

    return warped


# ------------------------------
# Step 2: Bubble Detection
# ------------------------------
def detect_answers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    answers = {}

    q_num = 1
    for c in contours:
        area = cv2.contourArea(c)
        if 300 < area < 1200:   # bubble size filter
            x,y,w,h = cv2.boundingRect(c)
            roi = thresh[y:y+h, x:x+w]
            fill_ratio = cv2.countNonZero(roi) / (w*h)

            if fill_ratio > 0.4:   # threshold for filled bubble
                answers[q_num] = "A"  # TODO: map to actual A/B/C/D
                q_num += 1
    print("Answers: ", answers)
    return answers


# ------------------------------
def load_answer_key_xlsx(xlsx_path):
    df = pd.read_excel(xlsx_path)


    answer_key = {}

    for subject in df.columns:
        answer_key[subject] = {}
        for cell in df[subject].dropna():  # skip empty cells
            try:
                q_part, ans_part = str(cell).split("-")
                q_no = int(q_part.strip())
                ans = ans_part.strip().lower()  # e.g., "a", "a,b", etc.
                answer_key[subject][q_no] = ans
            except Exception as e:
                print(f"Skipping invalid cell in {subject}: {cell} ({e})")
    print(answer_key)
    return answer_key


# ------------------------------
# Step 4: Evaluation
# ------------------------------
def evaluate(detected, answer_key):
    results = {}
    total_score = 0

    for subject, q_dict in answer_key.items():
        score = 0
        for q_num, correct in q_dict.items():
            detected_ans = detected.get(q_num, None)
            if detected_ans == correct:
                score += 1
        results[subject] = score
        total_score += score

    results["Total"] = total_score
    return results
