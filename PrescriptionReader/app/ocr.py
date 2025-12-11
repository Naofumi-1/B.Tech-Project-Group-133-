from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from .preprocessing import preprocess_prescription_bytes
# from preprocess import preprocess_file_to_pil
# def ocr_prescription(image_bytes: bytes)-> Image.Image:
#     pil=preprocess_prescription(image_bytes)
#     processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
#     model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
#     pixel_values = processor(pil, return_tensors="pt").pixel_values
#     ids = model.generate(pixel_values, max_length=50)
#     text = processor.batch_decode(ids, skip_special_tokens=True)[0]
#     return text

import torch
# Load model ONCE at startup (not in every request)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def ocr_prescription(image_bytes: bytes) -> str:
    # Preprocess image
    pil = preprocess_prescription_bytes(image_bytes).convert("RGB")

    # Convert to tensors
    pixel_values = processor(pil, return_tensors="pt").pixel_values.to(device)

    # Generate text
    ids = model.generate(pixel_values, max_length=256)

    # Decode
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    return text.strip()
