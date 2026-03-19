import fitz  # PyMuPDF
from .interfaces import DocumentReader


class PyMuPDFReader(DocumentReader):
    """Concrete implementation for reading PDF files using PyMuPDF."""
    
    def read(self, file_path: str) -> str:
        """Extracts text from all pages of the given PDF."""
        doc = fitz.open(file_path)
        text_content = []
        for page in doc:
            text_content.append(page.get_text())
        doc.close()
        
        # Trả về danh sách text theo từng trang (pages) để dễ dàng Multi-threading
        return text_content
