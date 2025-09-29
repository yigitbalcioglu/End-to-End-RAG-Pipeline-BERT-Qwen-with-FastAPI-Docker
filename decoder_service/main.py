from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

app = FastAPI(title="Decoder Service")

MODEL_NAME = os.getenv("DECODER_MODEL", "Qwen/Qwen3-0.6B")
DEVICE = os.getenv("DECODER_DEVICE", None)
device = torch.device(DEVICE if DEVICE else ("cuda" if torch.cuda.is_available() else "cpu"))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

class PromptRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200

@app.post("/generate")
def generate(req: PromptRequest):
    inputs = tokenizer([req.prompt], return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=req.max_new_tokens)
    output_text = tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return {"text": output_text}
