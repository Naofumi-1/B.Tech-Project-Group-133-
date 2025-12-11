import cv2
import numpy as np

# def preprocess_prescription(path):
#     img = cv2.imread(path)

#     # Downscale
#     h, w = img.shape[:2]
#     scale = 1200 / max(h, w)
#     img = cv2.resize(img, (int(w*scale), int(h*scale)))

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Remove ruled lines
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
#     lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
#     gray2 = cv2.subtract(gray, lines)

#     # Adaptive threshold
#     bin_img = cv2.adaptiveThreshold(
#         gray2, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         35, 15
#     )

#     # Strengthen handwriting
#     kernel2 = np.ones((2,2), np.uint8)
#     thick = cv2.dilate(bin_img, kernel2, iterations=1)

#     # Crop only handwriting
#     coords = cv2.findNonZero(255 - thick)  # invert for handwriting
#     x, y, w, h = cv2.boundingRect(coords)
#     crop = img[y:y+h, x:x+w]

#     cv2.imwrite("cleaned.jpg", crop)

#     return crop
import cv2
import numpy as np
from PIL import Image

def preprocess_prescription_bytes(image_bytes: bytes):
    # -------------------------------
    # 1) Decode image bytes → OpenCV image
    # -------------------------------
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image bytes")

    # -------------------------------
    # 2) Downscale large images
    # -------------------------------
    h, w = img.shape[:2]
    max_dim = 1200
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # -------------------------------
    # 3) Convert to grayscale
    # -------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # 4) Remove horizontal/ruled lines
    # -------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    gray2 = cv2.subtract(gray, lines)

    # -------------------------------
    # 5) Adaptive threshold
    # -------------------------------
    bin_img = cv2.adaptiveThreshold(
        gray2, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 15
    )

    # -------------------------------
    # 6) Bolden handwriting
    # -------------------------------
    kernel2 = np.ones((2,2), np.uint8)
    thick = cv2.dilate(bin_img, kernel2, iterations=1)

    # -------------------------------
    # 7) Crop to handwriting area
    # -------------------------------
    coords = cv2.findNonZero(255 - thick)  # invert: handwriting = black
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        crop = img[y:y+h, x:x+w]
    else:
        crop = img  # fallback if nothing detected

    # -------------------------------
    # 8) Convert OpenCV BGR → RGB → PIL
    # -------------------------------
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    return pil_img


# preprocess_prescription("sample.jpg")
