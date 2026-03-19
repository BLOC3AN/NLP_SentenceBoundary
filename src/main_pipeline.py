import os
import argparse
import sys
from sbd_pipeline.readers import PyMuPDFReader
from sbd_pipeline.normalizers import VietnameseTextNormalizer
from sbd_pipeline.models import OnnxSentenceBoundaryDetector
from sbd_pipeline.pipeline import SBDPipeline


def main():
    parser = argparse.ArgumentParser(description="A SOLID pipeline for PDF Sentence Boundary Detection.")
    parser.add_argument("--pdf", type=str, required=True, help="Đường dẫn đến file PDF")
    
    # Các path model lưu theo dạng Hardcode mặc định nếu không truyền argument
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_onnx = os.path.join(base_dir, "models", "sbd_49lang_bert_small.onnx")
    default_sp = os.path.join(base_dir, "models", "spe_mixed_case_64k_49lang.model")
    
    parser.add_argument("--onnx", type=str, default=default_onnx, help="Đường dẫn tới file ONNX")
    parser.add_argument("--sp", type=str, default=default_sp, help="Đường dẫn tới mô hình SentencePiece")
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
        threshold=0.5,
        max_seq_len=200
    )

    # Đưa các Interface cụ thể vào trong Pipeline
    pipeline = SBDPipeline(reader=reader, normalizer=normalizer, detector=detector)
    
    print(f"⏳ Đang xử lý tài liệu: {args.pdf} ...")
    sentences = pipeline.process_document(args.pdf)
    
    print(f"\n✅ Đã trích xuất được {len(sentences)} câu.")
    
    # In thủ nghiệm 10 câu đầu
    for idx, sentence in enumerate(sentences[:10]):
        print(f"Câu {idx + 1}: {sentence}")


if __name__ == "__main__":
    main()
