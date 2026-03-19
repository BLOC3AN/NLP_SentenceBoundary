import os
import argparse
import sys
from dotenv import load_dotenv

from sbd_pipeline.readers import PyMuPDFReader
from sbd_pipeline.normalizers import VietnameseTextNormalizer
from sbd_pipeline.models import OnnxSentenceBoundaryDetector
from sbd_pipeline.pipeline import SBDPipeline

def main():
    # Nạp biến môi trường từ file .env
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="A SOLID pipeline for PDF Sentence Boundary Detection.")
    parser.add_argument("--pdf", type=str, required=True, help="Đường dẫn đến file PDF")
    
    # Các path model thay đổi ưu tiên lấy từ ENV, nếu không có mới lấy mặc định
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_onnx = os.path.join(base_dir, "models", "sbd_49lang_bert_small.onnx")
    default_sp = os.path.join(base_dir, "models", "spe_mixed_case_64k_49lang.model")
    
    env_onnx = os.getenv("ONNX_MODEL_PATH", default_onnx)
    env_sp = os.getenv("SP_MODEL_PATH", default_sp)
    env_threshold = float(os.getenv("THRESHOLD", 0.5))
    env_max_seq_len = int(os.getenv("MAX_SEQ_LEN", 200))
    env_max_len_sent = int(os.getenv("MAX_LEN_SENT", 4))
    env_max_workers = int(os.getenv("MAX_WORKERS", 4))

    parser.add_argument("--onnx", type=str, default=env_onnx, help="Đường dẫn tới file ONNX")
    parser.add_argument("--sp", type=str, default=env_sp, help="Đường dẫn tới mô hình SentencePiece")
    parser.add_argument("--threshold", type=float, default=env_threshold, help="Ngưỡng cắt câu (0.0-1.0)")
    parser.add_argument("--max_seq_len", type=int, default=env_max_seq_len, help="Giới hạn positional embedding tối đa")
    parser.add_argument("--max_len_sent", type=int, default=env_max_len_sent, help="Ngưỡng ký tự nối câu cộc lốc")
    parser.add_argument("--max_workers", type=int, default=env_max_workers, help="Số luồng (Thread workers) tối đa")
    args = parser.parse_args()

    if not os.path.exists(args.pdf):
        print(f"Error: File {args.pdf} không tồn tại.")
        sys.exit(1)

    # Khởi tạo các Component theo nguyên lý Dependency Injection
    reader = PyMuPDFReader()
    normalizer = VietnameseTextNormalizer()
    detector = OnnxSentenceBoundaryDetector(
        onnx_model_path=args.onnx,
        sp_model_path=args.sp,
        threshold=args.threshold,
        max_seq_len=args.max_seq_len
    )

    # Đưa các Interface cụ thể vào trong Pipeline
    pipeline = SBDPipeline(
        reader=reader, 
        normalizer=normalizer, 
        detector=detector,
        max_len_sent=args.max_len_sent,
        max_workers=args.max_workers
    )
    
    print(f"⏳ Đang xử lý tài liệu: {args.pdf} ...")
    sentences = pipeline.process_document(args.pdf)
    
    print(f"\n✅ Đã trích xuất được {len(sentences)} câu.")
    
    # In thủ nghiệm 10 câu đầu
    for idx, sentence in enumerate(sentences[:10]):
        print(f"Câu {idx + 1}: {sentence}")


if __name__ == "__main__":
    main()
