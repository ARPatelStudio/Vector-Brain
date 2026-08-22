import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# ==========================================
# 🪵 LOGS SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================================
# 🧠 VECTOR ENGINE SETUP (CPU OPTIMIZED)
# ==========================================
# 🚀 FIX: Prevent PyTorch from using too many threads, saving RAM
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

try:
    # 🚀 FIX: Explicitly set device to 'cpu'
    embed_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    logger.info("🟢 SentenceTransformer (Vector Brain) Loaded Successfully!")
except Exception as embed_err:
    embed_model = None
    logger.error(f"🔴 Vector Engine Load Error: {embed_err}")

# Pinecone Setup
pc_api_key = os.getenv("PINECONE_API_KEY")
pc_index = None
if pc_api_key:
    try:
        pc = Pinecone(api_key=pc_api_key)
        if "saarthi-memory" in [idx.name for idx in pc.list_indexes()]:
            pc_index = pc.Index("saarthi-memory")
            logger.info("🟢 Pinecone Index Connected!")
    except Exception as pc_err:
        logger.error(f"🔴 Pinecone Init Error: {pc_err}")
else:
    logger.warning("⚠️ PINECONE_API_KEY is missing from environment variables!")

app = FastAPI(title="Saarthi Vector Brain (Render 2)", version="1.0.0")

# 📦 Pydantic Models for requests
class UpsertReq(BaseModel):
    id: str
    text: str
    metadata: dict

class SearchReq(BaseModel):
    query: str

class DeleteReq(BaseModel):
    id: str

@app.get("/")
def root():
    return {"status": "🟢 Vector Brain is Online and Ready to serve Embeddings!"}

@app.post("/upsert")
def upsert_vector(req: UpsertReq):
    if not pc_index or not embed_model:
        return {"error": "Pinecone or Embed Model not configured"}
    try:
        vector = embed_model.encode(req.text).tolist()
        pc_index.upsert(vectors=[(req.id, vector, req.metadata)])
        logger.info(f"✅ Upserted vector id: {req.id}")
        return {"status": "Vector Saved!"}
    except Exception as e:
        logger.error(f"Upsert error: {e}")
        return {"error": str(e)}

@app.post("/search")
def search_vector(req: SearchReq):
    if not pc_index or not embed_model:
        return {"matches": []}
    try:
        vector = embed_model.encode(req.query).tolist()
        res = pc_index.query(vector=vector, top_k=4, include_metadata=True)
        return res.to_dict()
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {"matches": []}

@app.post("/delete")
def delete_vector(req: DeleteReq):
    if not pc_index:
        return {"error": "Pinecone not configured"}
    try:
        pc_index.delete(ids=[req.id])
        logger.info(f"🗑️ Deleted vector id: {req.id}")
        return {"status": "Deleted"}
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # 🚀 RENDER FIX: Automatically binds to Render's dynamic port
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
