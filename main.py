''' Этот код демонстрирует базовую структуру приложения FastAPI для машинного обучения и веб-разработки.
    Он включает импорт необходимых библиотек и фреймворков, а также настройку приложения.
    Код не дописан для модели'''

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import io
import os
import base64
from PIL import Image, ImageOps
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import numpy as np
from typing import Dict
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# CORS middleware setup
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Заглушка:
model = None  # Заглушка для yolo11s-seg.pt
print("Заглушка YOLO11_seg: модель yolo11s-seg.pt будет подключена позже")

class_names = ['class_0', 'class_1', 'class_2']  # Заглушка классов
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


@app.get("/")
@app.get("/index.html", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    img = Image.open(image.file).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.indices.item()]
    predicted_probability = predicted.values.item() * 100

    result = {
        "predicted_class": predicted_class,
        "predicted_probability": f"{predicted_probability:.2f}%"
    }
   
    return result

@app.get("/predict.html", response_class=HTMLResponse)
async def predict_page():
    with open("templates/predict.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)