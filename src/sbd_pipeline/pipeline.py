from typing import List
from .interfaces import DocumentReader, TextNormalizer, BoundaryDetector


class SBDPipeline:
    """Orchestrator linking all components for Sentence Boundary Detection (Dependency Inversion)."""

    def __init__(self, reader: DocumentReader, normalizer: TextNormalizer, detector: BoundaryDetector):
        self.reader = reader
        self.normalizer = normalizer
        self.detector = detector

    def process_document(self, file_path: str) -> List[str]:
        # 1. Trích xuất text thô
        raw_text = self.reader.read(file_path)

        # 2. Chuẩn hoá text (Dọn rác Unicode, Typography)
        clean_text = self.normalizer.normalize(raw_text)

        # 3. Chẩn đoán ranh giới và trả về các câu hoàn chỉnh
        sentences = self.detector.detect_sentences(clean_text)

        # 4. Hậu xử lý (Post-processing): Sửa lỗi mô hình ngắt sai ở các từ viết tắt (S. A. S., v.v...)
        final_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            # Tiêu chí: Các đoạn text quá ngắn (<=4 ký tự) hoặc chữ cộc lốc không có dấu cách ("A.", "S.") 
            # thì chủ động ghép nối dính liền vào câu liền trước đó để bảo toàn thông tin.
            if len(sent) <= 4 or " " not in sent:
                if final_sentences:
                    final_sentences[-1] = final_sentences[-1] + " " + sent
                else:
                    final_sentences.append(sent)
            else:
                final_sentences.append(sent)

        return final_sentences
