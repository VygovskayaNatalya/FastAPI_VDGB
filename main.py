''' Этот код демонстрирует базовую структуру приложения FastAPI для машинного обучения и веб-разработки.
    Он включает импорт необходимых библиотек и фреймворков, а также настройку приложения.'''

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
app.mount("/static", StaticFiles(directory="D:/Lesson_FastAPI_DIPLOM/static"), name="static")

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

# Model initialization
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
model.load_state_dict(torch.load('resnet50_model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

class_names = ['NORMAL', 'SCOLIOSIS', 'SPONDYLOLISTHESIS']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

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