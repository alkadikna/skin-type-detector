import os
import glob
import pickle
import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from io import BytesIO
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator
# Import functions
from app.preprocessing.transform import detect_and_crop_face
from app.predict import predict_image
from api.database import get_db, save_prediction, update_feedback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inisialisasi FastAPI
app = FastAPI(
    title="Skin Type Identification API",
    description="Automated Skin Type Identification from Selfies Using CNN",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

class PredictionResponse(BaseModel):
    skin_type: str
    confidence: float
    success: bool
    message: Optional[str] = None
    prediction_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    prediction_id: str
    actual_skin_type: str

# Load model dari file
model_files = glob.glob(os.path.join("app", "models", "facial_model_*.pkl"))
if not model_files:
    raise Exception("No model files found. Please train the model first.")

model_files.sort(key=os.path.getmtime, reverse=True)
latest_model_path = model_files[0]

with open(latest_model_path, "rb") as f:
    model = pickle.load(f)

model = model.to(torch.device("cpu"))
model.eval()

# Upload image sama prediksi skin type
@app.post("/predict", response_model=PredictionResponse)
async def predict_skin_type(file: UploadFile = File(...), user_id: str = "anonymous", db=Depends(get_db)):
    try:
        contents = await file.read()
        image_stream = BytesIO(contents)

        image = Image.open(image_stream).convert("RGB")
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        face = detect_and_crop_face(image_cv)
        if face is None:
            return PredictionResponse(
                skin_type="",
                confidence=0.0,
                success=False,
                message="Tidak dapat mendeteksi wajah"
            )

        temp_face_path = "temp_face.jpg"
        cv2.imwrite(temp_face_path, (face * 255).astype("uint8"))

        predicted_class, confidence = predict_image(model, temp_face_path)

        os.remove(temp_face_path)

        skin_types = {
            0: "combination",
            1: "dry",
            2: "normal",
            3: "oily"
        }
        skin_type = skin_types.get(predicted_class, "unknown")

        prediction_id = await save_prediction(db, user_id, skin_type, confidence)

        return PredictionResponse(
            skin_type=skin_type,
            confidence=confidence,
            success=True,
            message="Prediksi berhasil",
            prediction_id=str(prediction_id)
        )

    except Exception as e:
        if os.path.exists("temp_face.jpg"):
            os.remove("temp_face.jpg")
        raise HTTPException(status_code=500, detail=str(e))

# Buat feedback pengguna
# @app.post("/feedback")
# async def submit_feedback(feedback: FeedbackRequest, db=Depends(get_db)):
#     try:
#         updated = await update_feedback(db, feedback.prediction_id, feedback.actual_skin_type)
#         if not updated:
#             raise HTTPException(status_code=404, detail="Prediksi tidak ditemukan")
#         return {"success": True, "message": "Feedback berhasil disimpan"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))