import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Saarthi Vector Brain")

# Load heavy PyTorch Model (CPU mode for 512MB RAM survival)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

pc_api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pc_api_key) if pc_api_key else None
pc_index = pc.Index("saarthi-memory") if pc else None

class UpsertReq(BaseModel):
    id: str
    text: str
    metadata: dict

class SearchReq(BaseModel):
    query: str

@app.post("/upsert")
def upsert_vector(req: UpsertReq):
    if not pc_index: return {"error": "Pinecone not configured"}
    vector = embed_model.encode(req.text).tolist()
    pc_index.upsert(vectors=[(req.id, vector, req.metadata)])
    return {"status": "Vector Saved!"}

@app.post("/search")
def search_vector(req: SearchReq):
    if not pc_index: return {"matches": []}
    vector = embed_model.encode(req.query).tolist()
    res = pc_index.query(vector=vector, top_k=4, include_metadata=True)
    return res.to_dict()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
