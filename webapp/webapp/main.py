import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from PIL import Image
import io

# Configuration
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / 'input'
CHECKPOINT_DIR = PROJECT_ROOT / 'output'
STATIC_IMAGES_DIR = DATA_DIR / 'train_images'

# Add backend to path
sys.path.append(str(BASE_DIR))
# Add ml_training to path for shopee_matching
sys.path.append(str(PROJECT_ROOT / 'ml_training' / 'ml_training'))

from backend.inference_engine import SearchEngine

app = FastAPI(title="PriceMatch AI")

print(f"Server Config:")
print(f"  - Project Root: {PROJECT_ROOT}")
print(f"  - Data Dir: {DATA_DIR}")
print(f"  - Checkpoints: {CHECKPOINT_DIR}")

# Mount Static Files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
# Mount the actual product images
if STATIC_IMAGES_DIR.exists():
    app.mount("/product_images", StaticFiles(directory=str(STATIC_IMAGES_DIR)), name="product_images")
else:
    print(f"Warning: Static Images Directory not found at {STATIC_IMAGES_DIR}. Images will not load.")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Global Engine
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    print("Initialize Engine...")
    engine = SearchEngine(str(DATA_DIR), str(CHECKPOINT_DIR))
    print("Engine Ready!")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/search")
async def search(
    request: Request,
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    global engine
    if not engine:
        return {"error": "Engine not ready"}

    results = []
    
    pil_image = None
    if image:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
    results = engine.search_multimodal(image_input=pil_image, text_query=text)
    
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
