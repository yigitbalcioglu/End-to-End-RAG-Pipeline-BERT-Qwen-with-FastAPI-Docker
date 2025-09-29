from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

app = FastAPI(title="Orchestrator Service")

ENCODER_URL = os.getenv("ENCODER_URL", "http://encoder_service:8000/embed")
DB_URL = os.getenv("DB_URL", "http://vectordb_service:8000")
DECODER_URL = os.getenv("DECODER_URL", "http://decoder_service:8000/generate")

class QueryRequest(BaseModel):
    text: str

@app.post("/query")
def query(req: QueryRequest):
    # 1️⃣ Encoder'dan embedding al
    emb_resp = requests.post(ENCODER_URL, json={"text": req.text}).json()
    embedding = emb_resp["embedding"]

    # 2️⃣ VectorDB'den k nearest results al
    db_resp = requests.post(f"{DB_URL}/query", json={"embedding": embedding, "k": 1}).json()
    result = db_resp["metadatas"][0][0] if db_resp["metadatas"][0] else {"Açıklama": "Çözüm bulunamadı"}

    # 3️⃣ Decoder ile cevap üret
    prompt = f"Müşteri sorun: {req.text}. Önerilen çözüm: {result['Açıklama']}. Lütfen net cevap ver."
    dec_resp = requests.post(DECODER_URL, json={"prompt": prompt}).json()

    return {"answer": dec_resp["text"], "retrieved_doc": result}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Request body için Pydantic model
class RAGQuery(BaseModel):
    query: str
    k: int = 3

@app.post("/rag-query")
async def rag_query(request: RAGQuery):
    """Tam RAG pipeline'ı test eder"""
    
    # 1. Encoder'dan embedding al
    encoder_resp = requests.post(
        "http://encoder_service:8000/embed",
        json={"text": request.query}
    )
    if encoder_resp.status_code != 200:
        return {"error": "Encoder failed", "details": encoder_resp.text}
    
    embedding = encoder_resp.json()["embedding"]
    
    # 2. VectorDB'den benzer sonuçları al
    db_resp = requests.post(
        "http://vectordb_service:8000/query",
        json={"embedding": embedding, "k": request.k}
    )
    if db_resp.status_code != 200:
        return {"error": "VectorDB failed", "details": db_resp.text}
    
    results = db_resp.json()
    
    # 3. Decoder'a gönder (LLM yanıt)
    top_result = results["metadatas"][0][0] if results.get("metadatas") and results["metadatas"][0] else {}
    explanation = top_result.get("Açıklama", "Çözüm bulunamadı")
    
    prompt = (
        f"Müşteri websitede bir sorunla karşılaştı: {request.query}. "
        f"Sistem kılavuzundan önerilen çözüm: {explanation}. "
        f"Lütfen buna göre kullanıcıya net ve düzgün bir cevap ver."
    )
    
    decoder_resp = requests.post(
        "http://decoder_service:8000/generate",
        json={"prompt": prompt, "max_tokens": 200}
    )
    
    if decoder_resp.status_code != 200:
        return {"error": "Decoder failed", "details": decoder_resp.text}
    
    return {
       
        "llm_response": decoder_resp.json().get("text", "")
    }

@app.get("/health")
def health():
    return {"status": "ok", "service": "api_service"}

""" "query": request.query,
        "retrieved_docs": results.get("metadatas", [[]])[0],
        "distances": results.get("distances", [[]])[0], """