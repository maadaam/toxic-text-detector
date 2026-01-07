from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from model import ToxityModel
from pydantic import BaseModel
import uvicorn

# FastAPI приложение
app = FastAPI(title="Toxicity Detection API")

# Подключаем шаблоны
templates = Jinja2Templates(directory="templates")

# Загружаем модель
model = ToxityModel(
    russian_stop_words_path="/Users/tomilovdima/good_bad_news/data/russian_stop_words",
    model_path="/Users/tomilovdima/good_bad_news/artefacts/toxic_model_v1.pkl"
)
# Модель для запроса
class PredictionRequest(BaseModel):
    text: str

# Главная страница с интерфейсом
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API эндпоинт
@app.post("/predict")
async def predict(request: PredictionRequest):
    result = model.predict(request.text)
    # return JSONResponse({
    #     "toxicity score": result
    # })
    return result

# Страница здоровья
@app.get("/health")
async def health():
    return {"status": "healthy"}

# Страница документации API
@app.get("/api/docs")
async def api_docs():
    return {"message": "Перейдите на /docs для Swagger документации"}

if __name__ == "__main__":
    
    print("\nЗапуск сервера на http://localhost:8000")
    print("Документация: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)