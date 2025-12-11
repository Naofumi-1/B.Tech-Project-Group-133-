from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from .ocr import ocr_prescription
model_id = "google/gemma-2b-it"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load model (CPU friendly)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",       # CPU or GPU automatically
    torch_dtype="auto",      
)

# Create HF pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=False,
    temperature=0.0,
    return_full_text=False,
)

from pydantic import BaseModel, Field
from typing import List

class MedicineItem(BaseModel):
    name: str = Field(..., description="Medicine name")
    dosage: str = Field(..., description="Dosage e.g. 500 mg, 1 tablet")
    frequency: str = Field(..., description="How many times a day")
    timing: str = Field("", description="When to take it — morning/night/before food/after food")
    duration: str = Field("", description="For how many days")

class Prescription(BaseModel):
    medicines: List[MedicineItem]
    notes: str = Field("", description="Extra instructions")

FORMAT_INSTRUCTIONS = """
Return a JSON object with this exact structure:

{
  "medicines": [
    {
      "name": "string (medicine name)",
      "dosage": "string (e.g. '625 mg' or '1 tablet')",
      "frequency": "string (e.g. 'twice daily')",
      "timing": "string (e.g. 'after food', 'before breakfast')",
      "duration": "string (e.g. '5 days', '1 week')"
    }
  ],
  "notes": "string with any extra instructions, or empty string"
}

Rules:
- Use these exact keys: "medicines", "name", "dosage", "frequency", "timing", "duration", "notes".
- Do NOT include $defs, properties, type, title, or any schema-like fields.
- Do NOT describe the schema. Fill in values from the prescription.
- Return ONLY JSON. No explanations, no markdown, no comments.
"""


from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=Prescription)


# Wrap with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

prompt_template = """
You are a medical prescription extraction assistant.

Your job:
- Identify all medications
- Extract dosage, frequency, timing, duration
- Output the result in strict JSON format
- Do not add any extra text

{format_instructions}

Prescription text:
\"\"\"{text}\"\"\"
"""


def build_prompt(ocr_text: str):
    return prompt_template.format(
        text=ocr_text,
        format_instructions=FORMAT_INSTRUCTIONS
    )

def extract_structured_prescription(text: str):
    user_prompt = build_prompt(text)

    # Gemma chat formatting
    chat_prompt = (
        "<start_of_turn>user\n"
        + user_prompt +
        "\n<end_of_turn>\n"
        "<start_of_turn>model"
    )

    raw_output = llm.invoke(chat_prompt)
    print("RAW OUTPUT:\n", raw_output)

    # Strip ```json fences if present
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "")
        cleaned = cleaned.replace("```", "")
        cleaned = cleaned.strip()

    # Now cleaned should be plain JSON: pass to the PydanticOutputParser
    return parser.parse(cleaned)


# extract_structured_prescription(ocr_prescription("sample.jpg"))

