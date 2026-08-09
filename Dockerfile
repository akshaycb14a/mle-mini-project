FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY data/database ./data/database
COPY data/models/label_encoder.pkl ./data/models/label_encoder.pkl
COPY data/models/training_stats.npy ./data/models/training_stats.npy
COPY main.py ./main.py
COPY config.py ./config.py
COPY models.py ./models.py

ARG MODEL_FILE=model_fp32.tflite
ENV MODEL_PATH=/app/data/tflite/${MODEL_FILE}
COPY data/tflite/${MODEL_FILE} /app/data/tflite/${MODEL_FILE}

CMD ["python", "src/mqtt/subscriber.py"]
