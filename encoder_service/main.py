from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

app = FastAPI(title="Encoder Service")

# Request body için Pydantic model
class EmbedRequest(BaseModel):
    text: str

class EmbedResponse(BaseModel):
    embedding: list

class Encoder:
    def __init__(self, model_name: str = "dbmdz/bert-base-turkish-cased", device: str = None):
        # Device seçimi
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Kullanılan cihaz: {self.device}")
        
        # Tokenizer & Model yükleme
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Hidden size kaydet
        self.embedding_dim = self.model.config.hidden_size
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Metin için embedding döndürür (float32 numpy array)."""
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Ortalama pooling (CLS yerine daha stabil olur)
                embedding = outputs.last_hidden_state.mean(dim=1)
            
            return embedding.squeeze().cpu().numpy().astype("float32")
        except Exception as e:
            print(f"Embedding oluşturulurken hata: {e}")
            return np.zeros((self.embedding_dim,), dtype="float32")

# Global encoder instance
encoder = Encoder()

# ✅ ENDPOINT EKLE
@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Metin için embedding döndürür"""
    try:
        embedding = encoder.get_embedding(request.text)
        return {"embedding": embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding hatası: {str(e)}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "encoder_service",
        "device": str(encoder.device),
        "embedding_dim": encoder.embedding_dim
    }

@app.get("/")
def root():
    return {
        "service": "Encoder Service",
        "endpoints": {
            "/embed": "POST - Metin embedding'i oluştur",
            "/health": "GET - Servis durumu"
        }
    }