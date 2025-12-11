import cv2
import numpy as np
from preprocess import (
    read_image, resize_keep_aspect, to_grayscale, denoise,
    apply_clahe, deskew, morphological_text_regions,
    merge_boxes, crop_largest_text_region, sharpen,
    adaptive_binarize, remove_horizontal_lines
)

def save_stage(name, img):
    if isinstance(img, np.ndarray):
        cv2.imwrite(f"{name}.jpg", img)
    else:
        img.save(f"{name}.jpg")

img = read_image("sample.jpg")
save_stage("0_original", img)

# 1. Resize
img1 = resize_keep_aspect(img, 1600)
save_stage("1_resized", img1)

# 2. Gray
gray = to_grayscale(img1)
save_stage("2_gray", gray)

# 3. Denoise
gray2 = denoise(gray)
save_stage("3_denoised", gray2)

# 4. CLAHE
gray3 = apply_clahe(gray2)
save_stage("4_clahe", gray3)

# 5. Deskew
gray4 = deskew(gray3)
save_stage("5_deskew", gray4)

# 6. Text region detection
boxes = morphological_text_regions(gray4)
merged = merge_boxes(boxes)
img_boxes = img1.copy()
for (x,y,w,h) in merged:
    cv2.rectangle(img_boxes, (x,y), (x+w, y+h), (0,255,0), 2)
save_stage("6_text_regions", img_boxes)

# 7. Crop to largest region
crop = crop_largest_text_region(img1, merged)
save_stage("7_crop", crop)

# 8. Sharpen + binarize
crop_gray = to_grayscale(crop)
crop_gray2 = sharpen(crop_gray)
save_stage("8_sharpen", crop_gray2)

bin_img = adaptive_binarize(crop_gray2)
save_stage("9_binary", bin_img)

# 9. Optional line removal
clean = remove_horizontal_lines(bin_img)
save_stage("10_no_lines", clean)
