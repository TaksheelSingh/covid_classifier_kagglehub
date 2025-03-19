from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from io import BytesIO
from scripts.inference import load_model, predict_image

app = FastAPI()
model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    prediction = predict_image(image, model)
    return {"prediction": prediction}
