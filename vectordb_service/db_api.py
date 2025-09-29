import os
import pandas as pd
import numpy as np
import ast
import chromadb
from pydantic import BaseModel
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI(title="VectorDB Service")

# -----------------------------
# ChromaDB persistent client
# -----------------------------
persist_dir = "./chroma_db"
os.makedirs(persist_dir, exist_ok=True)
client = chromadb.PersistentClient(path=persist_dir)

# Collection isimleri
RAW_COLLECTION = "raw_data"
VECTOR_COLLECTION = "vector_data"

# -----------------------------
# 1️⃣ Raw CSV verisini ekle (embedding olmadan)
# -----------------------------
csv_path = "./final_with_metadatas.csv"
df = pd.read_csv(csv_path)

try:
    try:
        raw_collection = client.get_collection(name=RAW_COLLECTION)
        print(f"Collection '{RAW_COLLECTION}' bulundu.")
    except:
        raw_collection = client.create_collection(name=RAW_COLLECTION)
        print(f"Yeni collection '{RAW_COLLECTION}' oluşturuldu.")

    # Metadata olarak tüm CSV satırını ekle, embedding yok
    ids_raw = [str(i) for i in range(len(df))]
    raw_collection.add(
        ids=ids_raw,
        embeddings=[[0.0]*384]*len(df),  # Dummy embedding
        metadatas=df.to_dict(orient="records")
    )
    print(f"{len(df)} satır '{RAW_COLLECTION}' collection'a eklendi (embedding yok).")
except Exception as e:
    print(f"Raw data ekleme hatası: {e}")

# -----------------------------
# 2️⃣ Seçilen kolonlar için embedding al ve vector collection'a ekle
# -----------------------------
# Örn: 'Başlık', 'Açıklama' kolonlarını embedding almak istiyoruz
columns_to_embed = ["Başlık", "Açıklama"]

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

encoder = Encoder(model_name="dbmdz/bert-base-turkish-cased")  # Senin encoder

try:
    try:
        vector_collection = client.get_collection(name=VECTOR_COLLECTION)
        print(f"Collection '{VECTOR_COLLECTION}' bulundu.")
    except:
        vector_collection = client.create_collection(name=VECTOR_COLLECTION)
        print(f"Yeni collection '{VECTOR_COLLECTION}' oluşturuldu.")

    embeddings = []
    metadata_list = []

    for idx, row in df.iterrows():
        # Seçilen kolonları birleştirip embedding al
        text = " ".join([str(row[col]) for col in columns_to_embed])
        emb = encoder.get_embedding(text)
        embeddings.append(emb.tolist())
        # Metadata olarak tüm satırı ekleyebilirsin
        metadata_list.append(row.to_dict())

    ids_vector = [str(i) for i in range(len(embeddings))]
    vector_collection.add(
        ids=ids_vector,
        embeddings=embeddings,
        metadatas=metadata_list
    )
    print(f"{len(embeddings)} satır '{VECTOR_COLLECTION}' collection'a eklendi (embeddingli).")

except Exception as e:
    print(f"Vector data ekleme hatası: {e}")


@app.get("/show_db")
def show_db(collection_name: str = Query(..., description="Collection adı"), n: int = Query(5, description="Gösterilecek ilk satır sayısı")):
    """
    Verilen collection'ın ilk n satırını ve toplam kayıt sayısını döndürür.
    """
    try:
        collection = client.get_collection(name=collection_name)
        row_count = collection.count()
        
        # İlk n metadata
        results = collection.query(
            query_embeddings=[[0.0]*768],  # dummy embedding sadece metadata almak için
            n_results=n
        )
        metadatas = results.get("metadatas", [[]])[0]

        return {
            "collection_name": collection_name,
            "total_rows": row_count,
            "first_n_rows": metadatas
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    
class QueryRequest(BaseModel):
    embedding: list
    k: int = 3

@app.post("/query")
def query_collection(req: QueryRequest, collection_name: str = VECTOR_COLLECTION):
    """
    Verilen embedding ile en yakın k sonucu döndürür.
    """
    try:
        collection = client.get_collection(name=collection_name)
        results = collection.query(
            query_embeddings=[req.embedding],
            n_results=req.k
        )
        return results
    except Exception as e:
        return {"status": "error", "detail": str(e)}