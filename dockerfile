FROM python:3.12-slim AS builder

WORKDIR /app

# Установка базовых инструментов сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Установка библиотек в отдельную директорию
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim

WORKDIR /app

# Копируем установленные пакеты из слоя builder
COPY --from=builder /install /usr/local

# Загружаем данные NLTK
RUN python -m nltk.downloader stopwords

COPY . .

# Гарантируем наличие папок и файла модели
RUN mkdir -p artefacts data && \
    touch artefacts/toxic_model_v1.pkl

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
