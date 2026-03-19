from typing import List
from .interfaces import DocumentReader, TextNormalizer, BoundaryDetector


import concurrent.futures

class SBDPipeline:
    """Orchestrator linking all components for Sentence Boundary Detection (Dependency Inversion)."""

    def __init__(self, reader: DocumentReader, normalizer: TextNormalizer, detector: BoundaryDetector, max_len_sent: int = 4, max_workers: int = 4):
        self.reader = reader
        self.normalizer = normalizer
        self.detector = detector
        self.max_len_sent = max_len_sent
        self.max_workers = max_workers

    def process_document(self, file_path: str) -> List[str]:
        # 1. Trích xuất text thô thành danh sách các trang/blocks
        raw_pages = self.reader.read(file_path)

        # Định nghĩa luồng xử lý cục bộ cho từng trang
        def process_page(page_text: str) -> List[str]:
            if not page_text.strip():
                return []
            # 2. Chuẩn hoá text
            clean_text = self.normalizer.normalize(page_text)
            # 3. Chẩn đoán ranh giới
            return self.detector.detect_sentences(clean_text)

        # Sử dụng Multi-threading để xử lý song song các trang (Giúp tăng tốc tối đa)
        sentences = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for page_sentences in executor.map(process_page, raw_pages):
                sentences.extend(page_sentences)

        # 4. Hậu xử lý (Post-processing)
        final_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            # Cấu hình ngưỡng linh hoạt qua biến môi trường (mặc định <= 4 ký tự)
            if len(sent) <= self.max_len_sent or " " not in sent:
                if final_sentences:
                    final_sentences[-1] = final_sentences[-1] + " " + sent
                else:
                    final_sentences.append(sent)
            else:
                final_sentences.append(sent)

        return final_sentences
