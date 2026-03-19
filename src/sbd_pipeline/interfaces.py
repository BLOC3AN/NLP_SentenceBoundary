from abc import ABC, abstractmethod
from typing import List


class DocumentReader(ABC):
    """Interface for reading documents (PDF, Word, TXT, etc.)."""
    
    @abstractmethod
    def read(self, file_path: str) -> str:
        pass


class TextNormalizer(ABC):
    """Interface for cleaning and normalizing raw text."""
    
    @abstractmethod
    def normalize(self, text: str) -> str:
        pass


class BoundaryDetector(ABC):
    """Interface for detecting sentence boundaries from a text string."""
    
    @abstractmethod
    def detect_sentences(self, text: str) -> List[str]:
        pass
