FROM python:3.10-slim

# System dependencies for PyMuPDF and FAISS
RUN apt-get update && \
    apt-get install -y build-essential gcc libglib2.0-0 libsm6 libxrender1 libxext6 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code and input/output folders
COPY . .

# Download models at build time (optional, for faster runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2', cache_folder='./model_cache')"
RUN python -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; T5Tokenizer.from_pretrained('t5-small', cache_dir='./model_cache'); T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir='./model_cache')"

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Default command (can be overridden)
CMD ["python", "challenge1b_final.py", "input"]
