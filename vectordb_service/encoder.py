# encoder.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

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
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Ortalama pooling (CLS yerine daha stabil olur)
                embedding = outputs.last_hidden_state.mean(dim=1)
            return embedding.squeeze().cpu().numpy().astype("float32")
        except Exception as e:
            print(f"Embedding oluşturulurken hata: {e}")
            return np.zeros((self.embedding_dim,), dtype="float32")



# encoder.py