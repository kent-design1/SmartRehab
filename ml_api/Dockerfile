FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY model_pipeline.pkl app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]