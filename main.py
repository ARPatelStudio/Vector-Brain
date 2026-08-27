import os
import logging
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastembed import TextEmbedding
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# ==========================================
# 🪵 LOGS SETUP
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Saarthi Vector Brain (Render 2)", version="10.1.0")

# 🚀 NAYA: Security Key (Render 1 se aane wali request ko verify karne ke liye)
SAARTHI_SERVER_KEY = os.getenv("SAARTHI_SERVER_KEY", "AmitPatel_Jarvis_Core_2026")

# ==========================================
# 🧠 ULTRA-LIGHTWEIGHT VECTOR ENGINE (ONNX / FastEmbed)
# ==========================================
logger.info("⏳ Loading FastEmbed Model (all-MiniLM-L6-v2)...")
try:
    embed_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logger.info("🟢 FastEmbed Loaded Successfully in Ultra-Low Memory Mode!")
except Exception as embed_err:
    embed_model = None
    logger.error(f"🔴 FastEmbed Load Error: {embed_err}")

# ==========================================
# 🌲 PINECONE VECTOR DATABASE SETUP
# ==========================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "saarthi-deep-memory")

pc = None
pc_index = None

if PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # 🚀 FIX 1: Updated syntax for latest Pinecone SDK to avoid crash
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            logger.info(f"⏳ Creating Pinecone Index: {PINECONE_INDEX_NAME}...")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384, # Dimension size for all-MiniLM-L6-v2
                metric="cosine", # Semantic matching
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info("✅ Pinecone Index Created!")
        
        pc_index = pc.Index(PINECONE_INDEX_NAME)
        logger.info("🟢 Pinecone Index Connected!")
    except Exception as pc_err:
        logger.error(f"🔴 Pinecone Init Error: {pc_err}")
else:
    logger.warning("⚠️ PINECONE_API_KEY is missing from environment variables!")

# ==========================================
# 📦 PYDANTIC MODELS
# ==========================================
class UpsertRequest(BaseModel):
    id: str
    text: str
    metadata: dict = {}

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class DeleteRequest(BaseModel):
    id: str

# ==========================================
# 🛡️ GLOBAL EXCEPTION HANDLER
# ==========================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"💀 Vector Brain Error: {exc}")
    return JSONResponse(status_code=500, content={"error": "Vector server error boss."})

# ==========================================
# 🚀 API ENDPOINTS
# ==========================================
@app.get("/")
def root():
    return {
        "status": "🟢 Vector Brain is Online (Low Memory Engine Active)!",
        "pinecone_connected": pc_index is not None,
        "model_loaded": embed_model is not None
    }

@app.post("/upsert")
def upsert_vector(req: UpsertRequest, x_api_key: str = Header(None)):
    # 🚀 FIX 2: Security Check
    if x_api_key != SAARTHI_SERVER_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized Access")

    if not pc_index or not embed_model:
        raise HTTPException(status_code=503, detail="Pinecone or Embed Model not configured")
    
    try:
        vector = list(embed_model.embed([req.text]))[0].tolist()
        
        meta = req.metadata
        meta["text_content"] = req.text 
        
        pc_index.upsert(vectors=[{
            "id": req.id,
            "values": vector,
            "metadata": meta
        }])
        
        logger.info(f"✅ Upserted memory ID: {req.id}")
        return {"success": True, "message": "Memory embedded and locked in Pinecone."}
    except Exception as e:
        logger.error(f"Upsert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_vector(req: SearchRequest, x_api_key: str = Header(None)):
    # 🚀 FIX 2: Security Check
    if x_api_key != SAARTHI_SERVER_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized Access")

    if not pc_index or not embed_model:
        raise HTTPException(status_code=503, detail="Pinecone or Embed Model not configured")
    
    try:
        vector = list(embed_model.embed([req.query]))[0].tolist()
        
        res = pc_index.query(vector=vector, top_k=req.top_k, include_metadata=True)
        
        matches = []
        for match in res.get("matches", []):
            matches.append({
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {})
            })
            
        return {"matches": matches}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete")
def delete_vector(req: DeleteRequest, x_api_key: str = Header(None)):
    # 🚀 FIX 2: Security Check
    if x_api_key != SAARTHI_SERVER_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized Access")

    if not pc_index:
        raise HTTPException(status_code=503, detail="Pinecone not configured")
    
    try:
        pc_index.delete(ids=[req.id])
        logger.info(f"🗑️ Deleted memory ID: {req.id}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
