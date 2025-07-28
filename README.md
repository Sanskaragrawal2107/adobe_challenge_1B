# PDF Content Analyzer and Summarizer

An intelligent document processing system that analyzes multiple PDFs, extracts relevant sections based on user personas and tasks, and provides refined summaries. Built with state-of-the-art NLP models and efficient text processing algorithms.

## 🚀 Features

- **Smart Section Extraction**: Automatically identifies and extracts meaningful sections from PDFs using font analysis and content structure
- **Semantic Search**: Uses BERT-based embeddings to find the most relevant content for specific tasks
- **Multi-Document Processing**: Processes multiple PDFs in parallel and extracts relevant sections from each
- **Persona-Based Analysis**: Tailors content selection based on user persona and specific tasks
- **Content Refinement**: Summarizes and refines extracted content for better readability
- **Docker Support**: Fully containerized solution for easy deployment

## 📋 Requirements

- Python 3.10+
- Docker (optional)
- Required Python packages (installed automatically):
  - PyMuPDF (fitz)
  - sentence-transformers
  - transformers
  - torch
  - faiss-cpu
  - numpy

## 🛠️ Installation

### Using Docker (Recommended)

1. Build the Docker image:
```bash
docker build -t pdf-analyzer .
```

2. Run the container:
```bash
docker run --rm -v %cd%/input:/app/input -v %cd%/output:/app/output pdf-analyzer
```

### Manual Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```


2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Download Required Models

Before running the script for the first time (if not using Docker), download the required models to avoid runtime downloads and ensure offline/fast execution:

```bash
# Download the sentence-transformers model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2', cache_folder='./model_cache')"

# Download the T5 summarization model and tokenizer
python -c "from transformers import T5ForConditionalGeneration, T5Tokenizer; T5Tokenizer.from_pretrained('t5-small', cache_dir='./model_cache'); T5ForConditionalGeneration.from_pretrained('t5-small', cache_dir='./model_cache')"
```

## 📖 Usage

1. Place your PDFs in the `input/PDFs/` directory
2. Create/modify `input/challenge1b_input.json` with your configuration:
```json
{
    "persona": {
        "role": "Travel Planner"
    },
    "job_to_be_done": {
        "task": "Plan a trip of 4 days for a group of 10 college friends"
    },
    "documents": [
        {"filename": "PDFs/document1.pdf"},
        {"filename": "PDFs/document2.pdf"}
    ]
}
```

3. Run the analyzer:
```bash
python challenge1b_final.py input
```

4. Find results in `challenge1b_output.json`

## 📤 Output Format

The program generates a JSON file containing:
- Metadata about the processed documents
- Extracted sections with importance rankings
- Refined content analysis for each relevant section

Example output structure:
```json
{
    "metadata": {
        "input_documents": [...],
        "persona": "...",
        "job_to_be_done": "...",
        "processing_timestamp": "..."
    },
    "extracted_sections": [...],
    "subsection_analysis": [...]
}
```

## 🔧 Advanced Configuration

- Adjust `TOP_K` in the code to control the number of sections per document
- Modify model parameters in the code for different summarization lengths
- Configure embedding models by changing `EMBEDDING_MODEL` and `SUMMARIZER_MODEL`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- PyMuPDF for PDF processing
- FAISS for efficient similarity search
