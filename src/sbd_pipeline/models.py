import numpy as np
import onnxruntime as ort
from sentencepiece import SentencePieceProcessor
from typing import List
from tqdm import tqdm
from .interfaces import BoundaryDetector


class OnnxSentenceBoundaryDetector(BoundaryDetector):
    """Wrapper for ONNX Sentence Boundary Detection model handling Tokenizing and Inference."""

    def __init__(self, onnx_model_path: str, sp_model_path: str, threshold: float = 0.5, max_seq_len: int = 500):
        self.sp = SentencePieceProcessor(sp_model_path)
        self.session = ort.InferenceSession(onnx_model_path)
        self.threshold = threshold
        self.max_seq_len = max_seq_len
        self.expected_inputs = [i.name for i in self.session.get_inputs()]

    def detect_sentences(self, text: str) -> List[str]:
        ids = self.sp.EncodeAsIds(text)
        if not ids:
            return []

        all_break_points = []

        # Tiến hành Chunking để vượt qua giới hạn Positional Embeddings
        for chunk_idx in tqdm(range(0, len(ids), self.max_seq_len), desc="Suy luận mô hình SBD", unit=" chunk"):
            chunk_ids = ids[chunk_idx : chunk_idx + self.max_seq_len]
            input_ids = [self.sp.bos_id()] + chunk_ids + [self.sp.eos_id()]

            input_tensor = np.array([input_ids], dtype=np.int64)
            onnx_inputs = {"input_ids": input_tensor}

            # Nạp Inputs động dựa trên ONNX Graph
            if "attention_mask" in self.expected_inputs:
                onnx_inputs["attention_mask"] = np.ones_like(input_tensor, dtype=np.int64)
            if "token_type_ids" in self.expected_inputs:
                onnx_inputs["token_type_ids"] = np.zeros_like(input_tensor, dtype=np.int64)

            outputs = self.session.run(None, onnx_inputs)[0]
            probs = outputs[0, 1:-1]

            chunk_bps = np.squeeze(np.argwhere(probs > self.threshold), axis=1).tolist()
            if not isinstance(chunk_bps, list):
                chunk_bps = [chunk_bps]

            # Dịch chuyển offset boundary ra index chuẩn vị trí gốc
            for bp in chunk_bps:
                all_break_points.append(bp + chunk_idx)

        # Chặn cuối cầu đề phòng Model bị lố
        if not all_break_points or all_break_points[-1] != len(ids) - 1:
            all_break_points.append(len(ids) - 1)

        sentences = []
        import re
        remaining_text = text.strip()

        for i, bp in enumerate(all_break_points):
            start = 0 if i == 0 else (all_break_points[i - 1] + 1)
            sub_ids = ids[start : bp + 1]
            if sub_ids:
                sub_text = self.sp.DecodeIds(sub_ids).strip()
                if sub_text:
                    # Thuật toán Alignment (Ánh xạ lại văn bản gốc):
                    # Vì model 64k nhỏ của SP không chứa đủ chữ cái tiếng Việt In Hoa (như Ứ, Ậ) hoặc ký tự @
                    # và trả về ⁇, ta dùng Regex đối chiếu lại văn bản gốc chưa token hóa.
                    escaped = re.escape(sub_text)
                    pattern_str = escaped.replace("⁇", ".*?")
                    pattern_str = pattern_str.replace(r"\ ", r"\s*")
                    # Nhượng bộ các cụm bị gãy do typographic
                    pattern_str = pattern_str.replace(r"\'", r".*?").replace(r"\"", r".*?")
                    
                    # Dò khớp chuỗi gốc ở vị trí đầu
                    match = re.search(r"^\s*([\s\S]*?)" + pattern_str, remaining_text, flags=re.IGNORECASE | re.DOTALL)
                    
                    if match:
                        matched_str = match.group(0).strip()
                        sentences.append(matched_str)
                        # Dịch con trỏ text cắt bỏ phần đã nhận diện
                        remaining_text = remaining_text[match.end():]
                    else:
                        # Fallback (Thấy gì báo nấy) nếu việc Regex match quá chặt/thất bại
                        try:
                            # Khớp mềm
                            soft_match = re.match(r"^\s*([\s\S]*?(?=\s))", remaining_text)
                            sentences.append(sub_text)
                            remaining_text = remaining_text[len(sub_text):]
                        except:
                            sentences.append(sub_text)

        return sentences
