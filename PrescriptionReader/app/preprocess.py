# preprocess.pypip install opencv-python numpy pillow

"""
Preprocessing utilities for OCR input images.
Produces enhanced crops ready to feed into TrOCR or other OCR models.
Dependencies: opencv-python, numpy, Pillow
pip install opencv-python numpy pillow
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image, ImageFilter
import math

# ----------------------------
# Utility I/O helpers
# ----------------------------
def read_image(path: str) -> np.ndarray:
    """Read image from disk in BGR (OpenCV) format."""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return img

def save_image(path: str, img: np.ndarray):
    """Save image (handles unicode paths). Expects BGR or single-channel."""
    ext = path.split('.')[-1]
    cv2.imencode(f'.{ext}', img)[1].tofile(path)

def to_pil(img: np.ndarray) -> Image.Image:
    """Convert BGR numpy to PIL RGB."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def from_pil(pil: Image.Image) -> np.ndarray:
    """Convert PIL image to BGR numpy."""
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# ----------------------------
# Core preprocessing steps
# ----------------------------
def resize_keep_aspect(img: np.ndarray, target_max_dim: int = 1600) -> np.ndarray:
    """Resize image so largest dimension <= target_max_dim, keep aspect ratio."""
    h, w = img.shape[:2]
    scale = min(1.0, target_max_dim / max(h, w))
    if scale == 1.0:
        return img
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(img_gray: np.ndarray, h: int = 10) -> np.ndarray:
    """Non-local means denoising for grayscale."""
    return cv2.fastNlMeansDenoising(img_gray, None, h, 7, 21)

def apply_clahe(img_gray: np.ndarray, clipLimit: float = 2.0, tileGridSize: Tuple[int,int]=(8,8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img_gray)

def sharpen(img_gray: np.ndarray) -> np.ndarray:
    """Unsharp mask style sharpening (works on grayscale)."""
    blurred = cv2.GaussianBlur(img_gray, (0,0), sigmaX=3)
    sharpened = cv2.addWeighted(img_gray, 1.5, blurred, -0.5, 0)
    return sharpened

def adaptive_binarize(img_gray: np.ndarray, method='gauss', blockSize: int = 31, C: int = 10) -> np.ndarray:
    if blockSize % 2 == 0:
        blockSize += 1
    if method == 'gauss':
        return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, blockSize, C)
    else:
        return cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, blockSize, C)

def remove_horizontal_lines(bin_img: np.ndarray, line_size_ratio: float = 0.02) -> np.ndarray:
    """
    Remove ruled/horizontal lines (common in notebook/prescription photos).
    bin_img: single-channel binary image (0 or 255).
    line_size_ratio: approximate fraction of image width for morphological element size.
    """
    h, w = bin_img.shape[:2]
    horizontal_size = max(1, int(w * line_size_ratio))
    # Create structure element for extracting horizontal lines
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    # detect horizontal lines
    detected_lines = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, horizontalStructure, iterations=1)
    # subtract lines from image
    no_lines = cv2.subtract(bin_img, detected_lines)
    # optionally inpaint to fill gaps where lines were removed
    mask = cv2.bitwise_not(no_lines)
    # Convert mask to 8-bit single channel for inpaint
    inpainted = cv2.inpaint(bin_img, mask, 3, cv2.INPAINT_TELEA)
    return inpainted

def deskew(img_gray: np.ndarray) -> np.ndarray:
    """Estimate skew angle and rotate to deskew. Works on binary/grayscale images."""
    # Use edges/threshold for better coords
    thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh < 255))
    if coords.shape[0] < 10:
        return img_gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img_gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# ----------------------------
# Text region detection & cropping
# ----------------------------
def morphological_text_regions(img_gray: np.ndarray,
                               scale: float = 1.0,
                               min_area: int = 500,
                               dilate_iter: int = 2) -> List[Tuple[int,int,int,int]]:
    """
    Return list of bounding boxes (x,y,w,h) that likely contain text.
    - Uses morphological operations on a gradient to cluster text strokes.
    """
    # compute gradient (Sobel)
    grad_x = cv2.Sobel(img_gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_16S, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.absdiff(grad_x, 0), 1.0, cv2.absdiff(grad_y, 0), 0, 0))
    # binarize gradient
    _, bin_grad = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # dilate to connect characters into words/lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(5 * scale), int(3 * scale)))
    dilated = cv2.dilate(bin_grad, kernel, iterations=dilate_iter)
    # find connected components
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area >= min_area:
            boxes.append((x,y,w,h))
    # sort top-to-bottom
    boxes = sorted(boxes, key=lambda b: b[1])
    return boxes

def merge_boxes(boxes: List[Tuple[int,int,int,int]], gap: int = 20) -> List[Tuple[int,int,int,int]]:
    """Merge boxes that are vertically close to each other to form paragraphs."""
    if not boxes:
        return []
    merged = []
    cur = list(boxes[0])
    for b in boxes[1:]:
        if b[1] <= cur[1] + cur[3] + gap:
            # overlap/close vertically => merge
            x1 = min(cur[0], b[0])
            y1 = min(cur[1], b[1])
            x2 = max(cur[0]+cur[2], b[0]+b[2])
            y2 = max(cur[1]+cur[3], b[1]+b[3])
            cur = [x1, y1, x2-x1, y2-y1]
        else:
            merged.append(tuple(cur))
            cur = list(b)
    merged.append(tuple(cur))
    return merged

def crop_largest_text_region(img: np.ndarray, boxes: List[Tuple[int,int,int,int]]) -> np.ndarray:
    """Crop image to the largest box (presumed to be main handwriting)."""
    if not boxes:
        return img
    areas = [w*h for (x,y,w,h) in boxes]
    idx = int(np.argmax(areas))
    x,y,w,h = boxes[idx]
    pad = int(0.05 * max(w,h))
    h_img, w_img = img.shape[:2]
    x1 = max(0, x-pad)
    y1 = max(0, y-pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)
    return img[y1:y2, x1:x2]

# ----------------------------
# High-level pipeline
# ----------------------------
def enhance_for_ocr(img_bgr: np.ndarray,
                    target_max_dim: int = 1600,
                    remove_lines: bool = False,
                    final_width: Optional[int] = None) -> np.ndarray:
    """
    Full pipeline:
      1. Resize (keep aspect)
      2. Grayscale
      3. Denoise
      4. CLAHE (contrast)
      5. Deskew
      6. Text region detection -> crop to largest region
      7. Sharpen & adaptive binarize
      8. Optionally remove horizontal lines
      9. Return final BGR image (3-channel) ready for OCR
    """
    img = resize_keep_aspect(img_bgr, target_max_dim)
    gray = to_grayscale(img)
    gray = denoise(gray, h=10)
    gray = apply_clahe(gray, clipLimit=2.0)
    # deskew while image still whole
    gray = deskew(gray)
    # detect text boxes
    boxes = morphological_text_regions(gray, scale=1.0, min_area=500, dilate_iter=2)
    merged = merge_boxes(boxes, gap=30)
    cropped = crop_largest_text_region(img, merged)
    # convert cropped to gray and sharpen
    cropped_gray = to_grayscale(cropped)
    cropped_gray = sharpen(cropped_gray)
    # adaptive binarize (keeps handwriting strokes visible)
    bin_img = adaptive_binarize(cropped_gray, method='gauss', blockSize=31, C=9)
    if remove_lines:
        bin_img = remove_horizontal_lines(bin_img, line_size_ratio=0.02)
    # convert binary back to 3-channel BGR (model expects RGB but conversion to PIL done later)
    final = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    if final_width:
        h, w = final.shape[:2]
        scale = final_width / w
        final = cv2.resize(final, (final_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return final

# ----------------------------
# Helper: tile an image into vertical strips for long pages
# ----------------------------
def tile_vertical(img_bgr: np.ndarray, tile_height: int = 800, overlap: int = 50) -> List[np.ndarray]:
    h, w = img_bgr.shape[:2]
    tiles = []
    y = 0
    while y < h:
        y2 = min(h, y + tile_height)
        crop = img_bgr[y:y2, 0:w]
        tiles.append(crop.copy())
        y = y2 - overlap
        if y >= h:
            break
    return tiles

# ----------------------------
# Example quick-run utilities
# ----------------------------
def preprocess_file_to_pil(path_in: str, path_out: Optional[str] = None, **kwargs) -> Image.Image:
    img = read_image(path_in)
    out = enhance_for_ocr(img, **kwargs)
    pil = to_pil(out)
    if path_out:
        pil.save(path_out)
    return pil

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--out", default="preprocessed.jpg")
    parser.add_argument("--no-line-remove", action="store_true")
    args = parser.parse_args()
    pil = preprocess_file_to_pil(args.input, args.out, remove_lines= args.no_line_remove, final_width=1024)
    print("Saved preprocessed image to", args.out)
