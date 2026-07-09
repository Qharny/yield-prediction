from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import io
import numpy as np

from inference import predict_image

app = FastAPI(
    title="Plant Disease Detection API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():

    return {
        "message": "Plant Disease Detection API Running"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        image = np.array(image).astype(np.float32).copy()

        result = predict_image(image)

        return result

    except Exception as e:

        return {
            "error": str(e)
        }