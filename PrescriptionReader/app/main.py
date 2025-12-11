from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .preprocessing import preprocess_prescription_bytes
from .ocr import ocr_prescription
from .FormatOutput import extract_structured_prescription
from .models import ExtractResponse

app =FastAPI(title="Prescription Reader OCR API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health():
    return {"status":"ok"}

@app.post("/extract", response_model=ExtractResponse)
async def extract(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    image_bytes = await file.read()
    try:
        raw_text = ocr_prescription(image_bytes)
    except Exception as e:
        raise HTTPException(500, str(e))
    if not raw_text.strip():
        raise HTTPException(422, "No text extracted from image")
    try:
        prescription = extract_structured_prescription(raw_text)
    except Exception as e:
        raise HTTPException(500, str(e))
    return ExtractResponse(raw_text=raw_text, prescription=prescription.model_dump())
