from fastapi import FastAPI
from pydantic import BaseModel
from encoder_service.main import Encoder  # az önce yazdığımız encoder.py
import os

# Config
MODEL_NAME = os.getenv("ENCODER_MODEL", "dbmdz/bert-base-turkish-cased")
DEVICE = os.getenv("ENCODER_DEVICE", None)

# Encoder init
encoder = Encoder(model_name=MODEL_NAME, device=DEVICE)

app = FastAPI(title="Encoder Service")

class TextRequest(BaseModel):
    text: str

@app.post("/embed")
def get_embedding(req: TextRequest):
    embedding = encoder.get_embedding(req.text)
    return {"embedding": embedding.tolist()}

# otomatik bu kod ile başlasın uvicorn encoder_api:app --reload --port 8001
