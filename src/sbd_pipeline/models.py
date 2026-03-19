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
        
        # Tiền tính toán để tránh check lại nhiều lần trong inference
        self.has_attention_mask = "attention_mask" in self.expected_inputs
        self.has_token_type = "token_type_ids" in self.expected_inputs
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        # Fallback pad_id nếu SentencePiece model không quy định pad_id
        self.pad_id = self.sp.pad_id() if self.sp.pad_id() != -1 else 0

    def detect_sentences(self, text: str) -> List[str]:
        ids = self.sp.EncodeAsIds(text)
        if not ids:
            return []

        all_break_points = []
        
        # --- TỐI ƯU 1: Gọn nhẹ lại quy trình Sequential Inference ---
        # Vì model ONNX (như báo lỗi Expand) bị cố định batch_size = 1 khi export,
        # chúng ta không thể ném cả ma trận batch 2D vào được. Ta sẽ chạy tuần tự
        # bằng vòng lặp gọn nhưng vẫn đẩy nhanh khâu vector hóa kết quả ra numpy.
        
        for chunk_idx in range(0, len(ids), self.max_seq_len):
            chunk_ids = ids[chunk_idx : chunk_idx + self.max_seq_len]
            padded_chunk = [self.bos_id] + chunk_ids + [self.eos_id]

            input_tensor = np.array([padded_chunk], dtype=np.int64)
            onnx_inputs = {"input_ids": input_tensor}

            if self.has_attention_mask:
                onnx_inputs["attention_mask"] = np.ones_like(input_tensor, dtype=np.int64)
            if self.has_token_type:
                onnx_inputs["token_type_ids"] = np.zeros_like(input_tensor, dtype=np.int64)

            # Gọi ONNX cho từng chunk
            outputs = self.session.run(None, onnx_inputs)[0]
            
            # Cắt bỏ prob của token BOS và EOS
            probs = outputs[0, 1:-1] 

            chunk_bps = np.squeeze(np.argwhere(probs > self.threshold), axis=1)
            if chunk_bps.ndim == 0:
                chunk_bps = np.expand_dims(chunk_bps, axis=0)
            
            if chunk_bps.size > 0:
                all_break_points.extend((chunk_bps + chunk_idx).tolist())

        # Chặn cuối cầu đề phòng Model bị lố
        if not all_break_points or all_break_points[-1] != len(ids) - 1:
            all_break_points.append(len(ids) - 1)

        # Vòng lặp 2: Thuật toán Alignment Two Pointers thay vì Regex
        sentences = []
        text_ptr = 0
        text_len = len(text)
        
        def is_quote(c):
            return c in "'\"”“‘’`"
            
        def chars_match(c1, c2):
            if c1.lower() == c2.lower(): return True
            if is_quote(c1) and is_quote(c2): return True
            return False

        for i, bp in enumerate(all_break_points):
            start = 0 if i == 0 else (all_break_points[i - 1] + 1)
            sub_ids = ids[start : bp + 1]
            if not sub_ids:
                continue
                
            sub_text = self.sp.DecodeIds(sub_ids).strip()
            if not sub_text:
                continue
                
            j = 0
            k = text_ptr
            sub_len = len(sub_text)
            
            # --- Two Pointers Matching ---
            while j < sub_len and k < text_len:
                if sub_text[j].isspace():
                    j += 1
                    continue
                if text[k].isspace():
                    k += 1
                    continue
                    
                c_sub = sub_text[j]
                
                # Khớp ký tự wildcard '⁇' của SentencePiece
                if c_sub == '⁇':
                    j += 1
                    while j < sub_len and sub_text[j].isspace():
                        j += 1
                    if j < sub_len:
                        anchor_c = sub_text[j]
                        # Tìm mỏ neo (anchor) trong text gốc với lookahead 50 ký tự
                        for lookahead in range(k, min(text_len, k + 50)):
                            if chars_match(anchor_c, text[lookahead]):
                                k = lookahead
                                break
                    else:
                        # Kết thúc token, bỏ qua cụm từ cho đến khi gặp khoảng trắng
                        while k < text_len and not text[k].isspace():
                            k += 1
                    continue
                
                # Cùng khớp ký tự bình thường
                if chars_match(c_sub, text[k]):
                    j += 1
                    k += 1
                else:
                    # Lookahead 20 ký tự trên text gốc phòng trường hợp SP đánh rơi ký tự (bị drop)
                    found = False
                    for lookahead_k in range(k + 1, min(text_len, k + 20)):
                        if not text[lookahead_k].isspace() and chars_match(c_sub, text[lookahead_k]):
                            k = lookahead_k
                            found = True
                            break
                    if found:
                        j += 1
                        k += 1
                    else:
                        # Trường hợp chữ bị nở ra hoặc mismatch lố, ta đẩy skip token trên sub_text
                        j += 1
            
            # Quét xong câu này, k đánh dấu điểm cắt trong gốc
            end_ptr = k
            
            # Đảm bảo câu cuối cùng sẽ cover hết toàn bộ cả các khoảng trắng/chấm lẻ tẻ bị SP bỏ quên
            if i == len(all_break_points) - 1:
                end_ptr = text_len
                
            matched_str = text[text_ptr:end_ptr].strip()
            if matched_str:
                sentences.append(matched_str)
                
            text_ptr = end_ptr

        return sentences
