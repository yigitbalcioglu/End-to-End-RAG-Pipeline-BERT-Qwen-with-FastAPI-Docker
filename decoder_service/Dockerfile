FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# PyTorch (decoder için gerekli)
RUN pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]