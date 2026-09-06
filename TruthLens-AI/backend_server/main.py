from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import your pipeline
from pipelines.text_pipeline import TextVerificationPipeline

# 1. Initialize the API
app = FastAPI(
    title="TruthLens AI Engine",
    description="Multimodal Fake News Detection API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the AI Engine Globally
# This ensures it only loads once when the server starts, not on every request.
print("[*] Booting up AI Engines. Please wait...")
engine = TextVerificationPipeline()

# 3. Define the Expected Data Structure
# Pydantic ensures the Android app sends the exact JSON format we expect.
class ClaimPayload(BaseModel):
    text: str

# 4. Create the API Endpoint
@app.post("/verify-text")
def verify_text_endpoint(payload: ClaimPayload):
    """
    Receives text from the Android floating bubble and returns the verification payload.
    Note: We use standard 'def' instead of 'async def' because our pipeline 
    currently uses synchronous blocking calls (DuckDuckGo and HF API).
    """
    try:
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty.")
            
        # Pass the extracted text to our mathematical pipeline
        result = engine.verify_claim(payload.text)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. Server Execution
if __name__ == "__main__":
    # Runs the server on localhost port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
