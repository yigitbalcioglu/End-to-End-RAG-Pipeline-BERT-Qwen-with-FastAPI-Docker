FROM python:3.10-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ana requirements'ı kullan (eğer tüm servisler aynı paketleri kullanıyorsa)
COPY ./requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Eğer vectordb'ye özel paketler varsa
# COPY requirements.txt . 
# RUN pip install -r requirements.txt

# PyTorch (sadece gerekiyorsa)
# RUN pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130

# Servis kodlarını kopyala (EN SONDA - cache için)
COPY . .

CMD ["uvicorn", "db_api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]