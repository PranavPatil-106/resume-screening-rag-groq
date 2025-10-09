# Resume Screening RAG with Hugging Face & Groq

A Resume Screening application using Retrieval-Augmented Generation (RAG) with semantic embeddings + FAISS and Groq's LLM for candidate analysis.

## Features

- **Semantic Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` by default (local) or Hugging Face Inference API (if `HF_TOKEN` is provided)
- **FAISS Vector Store**: Fast similarity search with cosine similarity (L2-normalized inner product)
- **Groq Integration**: Uses `llama-3.3-70b-versatile` for high‑quality reasoning
- **Streamlit UI**: Upload resumes (PDF), run queries, view results
- **Quick Queries**: One-click pre-written role queries (Python, DS, Full-Stack, ML, etc.)
- **Right Sidebar**: Shows loaded candidates (name extracted when possible; falls back to PDF name)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
# Optional: use Hugging Face Inference API for embeddings instead of local model
HF_TOKEN=your_hugging_face_token
```

Get your Groq API key from `https://console.groq.com`. HF token is optional (only needed for HF Inference API or private models).

### 3. Run the Application

```bash
streamlit run demo/app.py
```

## How It Works

1. **Document Processing**: Upload PDF resumes which are extracted and chunked
2. **Embeddings**: Each chunk is embedded using Sentence Transformers (local) or HF Inference API (if `HF_TOKEN` is set)
3. **Vector Store**: Embeddings are indexed in FAISS (inner product + L2 normalization ≈ cosine)
4. **Retrieval**: A job description/query is embedded and top‑K similar chunks are retrieved from FAISS
5. **Reasoning**: Groq LLM receives the context and prompt to analyze candidates and provide ranking + rationale

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PDF Resumes   │───▶│ Embeddings (ST/HF)│──▶│    FAISS Index   │
│                 │    │ (all-MiniLM-L6-v2)│   │  Similarity      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Job           │───▶│  Groq LLM        │◀───│  Top K Results  │
│   Description   │    │  Reasoning       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Dependencies

- **python-dotenv**: Environment variable management
- **pypdf**: PDF text extraction
- **streamlit**: Web application framework
- **groq**: Groq API client for LLM inference
- **numpy**: Numerical computations
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Local embeddings (optional if using HF API)
- **protobuf (pinned)**: For compatibility with TF/Transformers

## File Structure

```
resume-screening-rag-groq/
├── demo/
│   └── app.py                 # Streamlit web application
├── rag_pipeline/
│   └── rag.py                 # RAG pipeline with Groq integration
├── vectorstore/
│   └── hf_store.py            # Embeddings + FAISS vector store (ST or HF Inference API)
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## Usage

1. **Upload Resumes**: Use the sidebar to upload PDF resumes (automatically clears old data)
2. **Quick Queries**: Click a pre-written query or enter a custom job description
3. **Find Matches**: Click "Find Best Matches" to retrieve and analyze candidates
4. **Right Sidebar**: View loaded candidate names (or PDF names if not detected)

## Troubleshooting

### Common Issues

1. **Groq API Errors**: Ensure `GROQ_API_KEY` is set in `.env`
2. **Model Downloads**: Without `HF_TOKEN`, the first run of local `sentence-transformers` will download the model; with `HF_TOKEN`, embeddings can be served from the HF Inference API
3. **TensorFlow/Protobuf Warnings**: We pin `protobuf` for compatibility; if you see protobuf import errors, reinstall per `requirements.txt`
4. **Ports**: If `:8501` is busy, Streamlit will pick another port and display it in terminal

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed: `pip install -r requirements.txt`
2. Verify your Groq API key is valid
3. Run the test script: `python test_setup.py`

## Performance Notes

- all-MiniLM-L6-v2 embeddings are lightweight and fast; FAISS scales well
- Groq provides fast, high-quality completions
- Memory and index size scale linearly with number of chunks

## License

This project is open source and available under the MIT License.
