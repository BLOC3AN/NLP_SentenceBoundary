# NLP Sentence Boundary Detection (SBD) Pipeline

A highly optimized, SOLID-compliant, and production-ready pipeline for detecting sentence boundaries using a lightweight ONNX BERT model and SentencePiece. Designed specifically to handle real-world documents (like PDFs) with speed and precision.

## 🚀 Key Features

* **Multi-threading Architecture**: Utilizes Python's `ThreadPoolExecutor` to process multiple document pages concurrently. Because ONNX and SentencePiece run in C++ natively, this bypasses Python's GIL to maximize throughput.
* **Two-Pointers Text Alignment**: Completely replaced slow Regex string manipulations with an ultra-fast Two-Pointers algorithm. It perfectly aligns normalized tokens back to original raw text character mappings while flawlessly handling typographical errors out of the box.
* **Vectorized Inference**: Uses `numpy` for probability threshold filtering to maximize CPU hardware utilization.
* **Environment Configuration**: Fully adheres to the Twelve-Factor app configuration methodology. Every threshold, thread count, and model path is flexibly manipulated via the `.env` file.
* **PyMuPDF Integration**: Clean extraction from PDFs page-by-page to guarantee continuous memory efficiency (often under 200MB of RAM for entire books).

## 📂 Project Structure

```text
.
├── models/
│   ├── sbd_49lang_bert_small.onnx
│   └── spe_mixed_case_64k_49lang.model
├── src/
│   ├── main_pipeline.py           # Entry point and dependency injection
│   └── sbd_pipeline/
│       ├── interfaces.py          # Abstract schemas
│       ├── models.py              # ONNX runtime and Two Pointers algorithm
│       ├── normalizers.py         # Text cleaning and typographical handlers
│       ├── pipeline.py            # Multithreaded orchestration
│       └── readers.py             # PyMuPDF implementation
├── benchmark_pdf.py               # Stress test and profiling script
├── benchmark.py
├── .env.example
└── README.md
```

## ⚙️ Configuration (.env)

Duplicate `.env.example` to `.env` and adjust the variables as needed:

```env
ONNX_MODEL_PATH=models/sbd_49lang_bert_small.onnx
SP_MODEL_PATH=models/spe_mixed_case_64k_49lang.model
THRESHOLD=0.5
MAX_SEQ_LEN=200
MAX_LEN_SENT=4
MAX_WORKERS=4
```
* **MAX_SEQ_LEN**: Hard limit for positional embeddings in the ONNX model (Ensure it stays well below the 256 or 512 absolute limits of the exported ONNX matrix).
* **MAX_LEN_SENT**: Minimum character threshold to aggressively merge trailing detached phrases (like "A.", "S.").
* **MAX_WORKERS**: Number of parallel CPU threads to dedicate to document page processing.

## 💻 Usage

Make sure to install dependencies first:
```bash
pip install pymupdf psutil python-dotenv onnxruntime sentencepiece numpy tqdm
```

To run the full pipeline on a target PDF:
```bash
python src/main_pipeline.py --pdf "path/to/your/document.pdf"
```

## 📊 Benchmarks

On an average instance profiling a 305-page PDF (~30,000 words / ~670,000 chars):
- **Sequential**: ~63,000 chars/sec
- **Multi-thread (4 Cores)**: ~137,000 chars/sec (1.8 - 4 seconds to completion)
- **RAM**: Extremely stable at ~177 MB memory footprint entirely.