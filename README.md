# End-to-End RAG Pipeline: BERT + Qwen with FastAPI & Docker

A production-ready Retrieval-Augmented Generation (RAG) system built with microservices architecture, featuring Turkish BERT embeddings, ChromaDB vector storage, and Qwen LLM for intelligent question answering.

## 🏗️ Architecture

This project implements a complete RAG pipeline using Docker containerization:

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Encoder   │─────▶│  VectorDB    │─────▶│   Decoder   │
│  Service    │      │   Service    │      │   Service   │
│  (BERT)     │      │  (ChromaDB)  │      │   (Qwen)    │
└─────────────┘      └──────────────┘      └─────────────┘
       │                     │                      │
       └─────────────────────┴──────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  API Service   │
                    │   (FastAPI)    │
                    └────────────────┘
```

### Services Overview

| Service | Port | Purpose | GPU Support |
|---------|------|---------|-------------|
| **API Service** | 8000 | Main API gateway, orchestrates all services | ❌ |
| **Encoder Service** | 8001 | Text embedding using BERT (Turkish) | ✅ Optional |
| **VectorDB Service** | 8002 | Vector storage and similarity search (ChromaDB) | ✅ Optional |
| **Decoder Service** | 8003 | Text generation using Qwen LLM | ✅ Required |

## 🚀 Quick Start

### Prerequisites

- **Docker** (20.10+) and **Docker Compose** (v3.9+)
- **NVIDIA GPU** (optional but recommended for faster inference)
- **NVIDIA Container Toolkit** (for GPU support)
- **8GB+ RAM** (16GB recommended)
- **10GB+ disk space**

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yigitbalcioglu/End-to-End-RAG-Pipeline-BERT-Qwen-with-FastAPI-Docker.git
```

#### 2. Project Structure

```
aiagent/
├── api_service/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── encoder_service/
│   ├── Dockerfile
│   ├── main.py (or encoder.py)
│   └── requirements.txt
├── vectordb_service/
│   ├── Dockerfile
│   ├── db_api.py
│   ├── data.csv
│   └── chroma_db/
├── decoder_service/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── docker-compose.yml
└── requirements.txt
```

#### 3. Initial Build (First Time Only)

Build all services (this may take 10-15 minutes on first run):

```bash
docker-compose build
```

**Note:** The first build downloads models, installs PyTorch, and sets up CUDA support. Subsequent builds use cache and complete in 5-10 seconds.

#### 4. Start All Services

```bash
docker-compose up -d
```

Verify all services are running:

```bash
docker-compose ps
```

Expected output:
```
NAME                COMMAND                  SERVICE             STATUS
api_service         "uvicorn main:app..."    api_service         Up 2 minutes
encoder_service     "uvicorn main:app..."    encoder_service     Up 2 minutes
vectordb_service    "uvicorn db_api:a..."    vectordb_service    Up 2 minutes
decoder_service     "uvicorn main:app..."    decoder_service     Up 2 minutes
```

## 📋 Common Commands

### Service Management

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f encoder_service

# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart vectordb_service
```

### Rebuilding Services

#### Full Rebuild (After Dockerfile or requirements.txt changes)

```bash
# Rebuild all services
docker-compose build

# Rebuild specific service
docker-compose build encoder_service

# Rebuild with no cache (clean build)
docker-compose build --no-cache
```

#### Rebuild and Restart Specific Service

```bash
# Example: Update vectordb_service
docker-compose build vectordb_service
docker-compose up -d vectordb_service

# Alternative: One-liner
docker-compose up -d --build vectordb_service
```

#### Quick Restart (Code Changes with Volume Mount)

If you have volume mounts enabled in `docker-compose.yml`, code changes are reflected immediately:

```bash
# Just restart the service (no rebuild needed)
docker-compose restart api_service
```

### Container Management

```bash
# Access container shell
docker exec -it encoder_service bash

# Check GPU availability inside container
docker exec -it decoder_service python -c "import torch; print(torch.cuda.is_available())"

# View container resource usage
docker stats
```

## 🧪 Testing the Pipeline

### Health Checks

Test each service individually:

```bash
# API Service
curl http://localhost:8000/health

# Encoder Service
curl http://localhost:8001/health

# VectorDB Service
curl http://localhost:8002/health

# Decoder Service
curl http://localhost:8003/health
```

### Full RAG Query

Test the complete pipeline with a sample query:

**PowerShell:**
```powershell
curl -X POST http://localhost:8000/rag-query `
  -H "Content-Type: application/json" `
  -d '{\"query\": \"How do I import users from a CSV file?\", \"k\": 3}'
```

**Bash/Linux:**
```bash
curl -X POST http://localhost:8000/rag-query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I import users from a CSV file?", "k": 3}'
```

**Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/rag-query",
    json={
        "query": "How do I import users from a CSV file?",
        "k": 3
    }
)

print(response.json())
```

### Interactive API Documentation

Access Swagger UI for interactive testing:

- **API Service:** http://localhost:8000/docs
- **Encoder Service:** http://localhost:8001/docs
- **VectorDB Service:** http://localhost:8002/docs
- **Decoder Service:** http://localhost:8003/docs

## ⚙️ Configuration

### GPU Support

#### Enable GPU for All Services

Edit `docker-compose.yml`:

```yaml
services:
  encoder_service:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
```

#### Verify GPU Access

```bash
# Test GPU in container
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check encoder service GPU
docker-compose logs encoder_service | grep "cuda"
```

Expected log output:
```
Kullanılan cihaz: cuda:0
```

### Environment Variables

Customize service URLs in `docker-compose.yml`:

```yaml
api_service:
  environment:
    - ENCODER_URL=http://encoder_service:8000/embed
    - DB_URL=http://vectordb_service:8000
    - DECODER_URL=http://decoder_service:8000/generate
```

### Volume Mounts (Development Mode)

Enable hot-reload for code changes:

```yaml
services:
  encoder_service:
    volumes:
      - ./encoder_service:/app  # Maps local code to container
```

With volumes enabled, changes to Python files are reflected immediately without rebuilding.

## 🐛 Troubleshooting

### Service Won't Start

```bash
# Check logs for errors
docker-compose logs encoder_service

# Restart specific service
docker-compose restart encoder_service

# Force rebuild
docker-compose build --no-cache encoder_service
docker-compose up -d encoder_service
```

### GPU Not Detected

```bash
# Verify NVIDIA driver (Windows PowerShell)
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Verify container GPU access
docker exec -it decoder_service nvidia-smi
```

### Port Already in Use

```bash
# Find process using port 8000
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Change port in docker-compose.yml
ports:
  - "8005:8000"  # Map to different host port
```

### Out of Memory

Reduce batch size or limit concurrent requests:

```python
# In service code
MAX_BATCH_SIZE = 8  # Reduce from 32
```

Or increase Docker memory limit:

```bash
# Docker Desktop → Settings → Resources → Memory
# Increase to 8GB or more
```

## 📊 Performance Optimization

### Build Time Optimization

- ✅ First build: 10-15 minutes (downloads models, installs packages)
- ✅ Cached rebuild: 5-10 seconds (only copies changed code)
- ✅ No rebuild needed: With volume mounts, just restart services

### Layer Caching Best Practices

Dockerfile structure for optimal caching:

```dockerfile
# 1. Base image (rarely changes)
FROM python:3.10-slim

# 2. System dependencies (rarely changes)
RUN apt-get update && apt-get install -y git curl

# 3. Python dependencies (changes occasionally)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 4. Application code (changes frequently) - LAST
COPY . .
```

### GPU Memory Management

Monitor GPU usage:

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Inside container
docker exec -it decoder_service nvidia-smi
```

## 🔒 Production Deployment

### Remove Development Features

```yaml
# docker-compose.prod.yml
services:
  encoder_service:
    volumes: []  # Remove volume mounts
    command: ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    # Remove --reload flag
```

### Security Considerations

- Remove debug endpoints
- Add authentication middleware
- Use environment files for secrets
- Enable HTTPS/TLS

### Scaling

```bash
# Scale specific service
docker-compose up -d --scale encoder_service=3
```

## 📝 Development Workflow

### Typical Development Cycle

1. **Make code changes** in local files
2. **Restart service** (if volume mounted):
   ```bash
   docker-compose restart api_service
   ```
3. **Rebuild** (if dependencies changed):
   ```bash
   docker-compose build api_service
   docker-compose up -d api_service
   ```
4. **Test** via API docs or curl
5. **View logs**:
   ```bash
   docker-compose logs -f api_service
   ```

### Hot Reload

Enable auto-reload in Dockerfile CMD:

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

- **BERT Turkish:** dbmdz/bert-base-turkish-cased
- **Qwen LLM:** Qwen/Qwen3-0.6B
- **ChromaDB:** Vector database
- **FastAPI:** Web framework

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review logs: `docker-compose logs -f`

---

**Built with ❤️ using FastAPI, Docker, and Transformers**

