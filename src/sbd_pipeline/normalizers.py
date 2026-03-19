import unicodedata
from .interfaces import TextNormalizer


class VietnameseTextNormalizer(TextNormalizer):
    """Concrete implementation for cleaning Vietnamese text from layout issues."""

    def normalize(self, text: str) -> str:
        # 1. Ép toàn bộ mã Unicode về dạng Dựng Sẵn (NFC)
        text = unicodedata.normalize('NFC', text)

        # 2. Xử lý các ký tự Typography dính từ format PDF
        replacements = {
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "–": "-",
            "—": "-",
            "•": ""
        }
        for old_char, new_char in replacements.items():
            text = text.replace(old_char, new_char)

        # 3. Dọn dẹp khoảng trắng và ngắt dòng thủ công
        text = text.replace("\n", " ").replace("\r", " ")
        while "  " in text:
            text = text.replace("  ", " ")

        return text.strip()
