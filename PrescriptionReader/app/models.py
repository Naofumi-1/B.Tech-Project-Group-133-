from pydantic import BaseModel, Field
from typing import List

class MedicineItem(BaseModel):
    name: str
    dosage: str
    frequency: str
    timing: str = ""
    duration: str = ""

class Prescription(BaseModel):
    medicines: List[MedicineItem]
    notes: str = ""

class ExtractResponse(BaseModel):
    raw_text: str
    prescription: Prescription