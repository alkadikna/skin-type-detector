FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the necessary files for api
COPY api/main.py api/main.py
COPY api/database.py api/database.py
COPY app/models app/models
COPY app/preprocessing/transform.py app/preprocessing/transform.py
COPY app/predict.py app/predict.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]