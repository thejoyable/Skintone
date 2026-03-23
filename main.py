import io
import base64
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from ultralytics import YOLO

# --- App Setup ---
app = FastAPI(title="YOLO Skin Tone Detector")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# --- Load Model Once ---
MODEL_PATH = Path(__file__).parent / "best.pt"
model = YOLO(str(MODEL_PATH))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run inference
    results = model.predict(source=image, conf=0.25, verbose=False)
    result = results[0]

    # Build detections list
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        detections.append({
            "class": result.names[cls_id],
            "confidence": round(float(box.conf[0]), 3),
            "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
        })

    # Render annotated image
    annotated = result.plot()  # numpy BGR array
    annotated_rgb = annotated[:, :, ::-1]  # BGR -> RGB
    pil_img = Image.fromarray(annotated_rgb)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "image": img_b64,
        "detections": detections,
        "count": len(detections),
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
