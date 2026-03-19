import os
import time
import threading
import concurrent.futures
import psutil
from typing import List

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF is not installed. Please install it with 'pip install pymupdf'.")
    exit(1)

from src.sbd_pipeline.models import OnnxSentenceBoundaryDetector

PDF_PATH = "documents/Lê Thành Hải - Thực hành Học Máy - Tập 1.pdf"

def extract_pages(pdf_path: str) -> List[str]:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            pages.append(text)
    return pages

class ResourceMonitor:
    def __init__(self):
        self.keep_measuring = True
        self.max_cpu = 0.0
        self.max_ram_mb = 0.0
        self.avg_cpu = 0.0
        self.avg_ram_mb = 0.0
        self._thread = None
        self.process = psutil.Process(os.getpid())

    def _measure(self):
        cpu_samples = []
        ram_samples = []
        # Khởi tạo CPU percent (lần đầu sẽ trả về 0)
        self.process.cpu_percent(interval=None)
        
        while self.keep_measuring:
            cpu = self.process.cpu_percent(interval=0.1)
            ram = self.process.memory_info().rss / (1024 * 1024)
            cpu_samples.append(cpu)
            ram_samples.append(ram)
            if cpu > self.max_cpu: self.max_cpu = cpu
            if ram > self.max_ram_mb: self.max_ram_mb = ram
            
        if cpu_samples:
            self.avg_cpu = sum(cpu_samples) / len(cpu_samples)
            self.avg_ram_mb = sum(ram_samples) / len(ram_samples)

    def start(self):
        self.keep_measuring = True
        self.max_cpu = 0.0
        self.max_ram_mb = 0.0
        self._thread = threading.Thread(target=self._measure)
        self._thread.start()

    def stop(self):
        self.keep_measuring = False
        if self._thread:
            self._thread.join()

def run_benchmark(detector, pages, mode="sequential", workers=1):
    monitor = ResourceMonitor()
    time.sleep(0.5) # Để process ổn định CPU
    monitor.start()
    
    start_time = time.time()
    total_sentences = 0
    total_chars = sum(len(p) for p in pages)
    
    if mode == "sequential":
        for page in pages:
            sents = detector.detect_sentences(page)
            total_sentences += len(sents)
    elif mode == "multithread":
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(detector.detect_sentences, pages))
            total_sentences = sum(len(r) for r in results)
            
    end_time = time.time()
    monitor.stop()
    
    elapsed = end_time - start_time
    throughput = total_chars / elapsed
    
    print(f"\n======================================")
    print(f"[{mode.upper()} - {workers} WORKERS]")
    print(f"======================================")
    print(f"Execution Time : {elapsed:.4f} seconds")
    print(f"Total Sentences: {total_sentences:,}")
    print(f"Throughput     : {throughput:,.0f} chars/sec")
    print(f"CPU Usage      : Avg {monitor.avg_cpu:.1f}%, Max {monitor.max_cpu:.1f}%")
    print(f"RAM Usage      : Avg {monitor.avg_ram_mb:.1f} MB, Max {monitor.max_ram_mb:.1f} MB")

def main():
    print(">> Processing PDF extraction...")
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF not found at '{PDF_PATH}'")
        return
        
    pages = extract_pages(PDF_PATH)
    total_chars = sum(len(p) for p in pages)
    print(f"Loaded {len(pages)} pages with {total_chars:,} characters.")
    
    print("\n>> Loading SBD ONNX Model...")
    detector = OnnxSentenceBoundaryDetector(
        onnx_model_path="models/sbd_49lang_bert_small.onnx",
        sp_model_path="models/spe_mixed_case_64k_49lang.model",
        max_seq_len=200
    )
    
    # Warm up ONNX components to allocate internal matrices
    detector.detect_sentences("Warm up ONNX session...")
    
    print("\n>> Starting Phase 1: Sequential Benchmark")
    run_benchmark(detector, pages, mode="sequential", workers=1)
    
    print("\n>> Starting Phase 2: Multithread (1 Core)")
    run_benchmark(detector, pages, mode="multithread", workers=1)
    
    print("\n>> Starting Phase 3: Multithread (4 Cores)")
    run_benchmark(detector, pages, mode="multithread", workers=4)

if __name__ == "__main__":
    main()
