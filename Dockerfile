FROM python:3.12.3-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y git && pip install --no-cache-dir huggingface-hub

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-mpnet-base-v2')"

EXPOSE 8501

CMD ["streamlit", "run", "./frontend/chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
