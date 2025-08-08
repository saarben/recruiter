FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT=8501
EXPOSE 8501

ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
