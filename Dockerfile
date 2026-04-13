FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    mediapipe==0.10.33 \
    opencv-python-headless==4.13.0.92 \
    fastapi==0.135.3 \
    uvicorn==0.44.0 \
    python-multipart==0.0.26 \
    websockets==16.0 \
    gtts==2.5.4 \
    numpy==2.4.4 \
    scikit-learn==1.8.0

RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu

COPY . .

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]