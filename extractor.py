import re
import cv2
import pytesseract
from PIL import Image
from rapidfuzz import fuzz
import numpy as np
from typing import Dict, Optional
import fitz

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

def load_image_or_pdf(path):
    if path.lower().endswith(".pdf"):
        try:
            doc = fitz.open(path)
            page = doc.load_page(0)  # first page
            pix = page.get_pixmap(dpi=300)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except Exception as e:
            print("PDF read error (PyMuPDF):", e)
            return None
    return cv2.imread(path)


# PREPROCESS FOR IMAGES (SCANNED IMAGE)
def preprocess_image(img):
    h = 2500
    scale = h / img.shape[0]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 15, 75, 75)

    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 6
    )
    return th

# PREPROCESS FOR PDF
def preprocess_pdf(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return gray


# OCR TEXT
def get_text(img, is_pdf):
    if img is None:
        return ""
    if is_pdf:
        config = "--oem 1 --psm 6 -c preserve_interword_spaces=1"
    else:
        config = "--oem 3 --psm 4"   # for image setting

    return pytesseract.image_to_string(img, config=config)


# EXTRACT CBC VALUES
def extract_cbc(text):
    raw = text
    text = raw.lower()
    text = re.sub(r"[:|]+", " ", text)
    text = text.replace("-", " ").replace("_", " ").replace("/", " ")
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    extracted = {
        "HGB": None, "WBC": None, "RBC": None, "PLT": None, "HCT": None,
        "MCV": None, "MCH": None, "MCHC": None, "RDWSD": None, "RDWCV": None
    }

    patt = {
        "HGB":   r"\b(hgb|hemoglobin)\b",
        "WBC":   r"\b(wbc|leukocyte|tlc|white\s*blood\s*(cell|count)?)\b",
        "RBC":   r"\brbc\b",
        "PLT":   r"\b(plt|platelet|platelets)\b",
        "HCT":   r"\b(hct|hematocrit|pcv)\b",
        "MCV":   r"\bmcv\b",
        "MCHC":  r"\bmchc\b",
        "MCH":   r"(?<![a-z])mch(?!c)\b",
    }

    def first_number(s):
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        return float(m.group(1)) if m else None

    # PASS 1 — Regex matches
    for field, rx in patt.items():
        for line in lines:
            if re.search(rx, line):
                num = first_number(line)
                if num is not None:
                    extracted[field] = num
                break

    # RDW special cases
    fuzzy_rdw = ["rdw", "raw", "row", "rdwcv", "rdwsd", "rd w"]

    def is_rdw_line(s):
        s2 = s.replace(" ", "")
        return any(fuzz.partial_ratio(fr, s2) > 80 for fr in fuzzy_rdw)

    for line in lines:
        if is_rdw_line(line):
            nums = re.findall(r"\d+\.?\d*", line)
            nums = [float(x) for x in nums]

            for n in nums:
                if 8 <= n <= 25:
                    extracted["RDWCV"] = n
                if 30 <= n <= 80:
                    extracted["RDWSD"] = n

    return extracted


# -----------------------------------------------------
# MAIN ENTRY
# -----------------------------------------------------
def extract_from_image_or_pdf(path):
    is_pdf = path.lower().endswith(".pdf")
    img = load_image_or_pdf(path)

    if img is None:
        return {}

    # Select proper preprocessing
    if is_pdf:
        proc = preprocess_pdf(img)
    else:
        proc = preprocess_image(img)

    text = get_text(proc, is_pdf)

    data = extract_cbc(text)
    return data
