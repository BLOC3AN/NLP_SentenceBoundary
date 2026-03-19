import time
import os
from src.sbd_pipeline.models import OnnxSentenceBoundaryDetector

# Đường dẫn model
onnx_path = "models/sbd_49lang_bert_small.onnx"
sp_path = "models/spe_mixed_case_64k_49lang.model"

print(">> Loading SBD model...")
start_load = time.time()
detector = OnnxSentenceBoundaryDetector(
    onnx_model_path=onnx_path,
    sp_model_path=sp_path,
    max_seq_len=200
)
print(f"Loaded in {time.time() - start_load:.4f}s")

text_block = """Trong những ngày gần đây, giá vàng thế giới liên tục ghi nhận các mức kỷ lục mới, xô đổ mọi dự báo trước đó của giới chuyên gia. 
Sáng nay lúc 8h30 (giờ Hà Nội), giá vàng giao ngay trên thị trường quốc tế neo ở mức 2.385 USD/ounce, tăng nhẹ so với chốt phiên cuối tuần trước.
Có nhiều nguyên nhân dẫn đến đợt tăng giá phi mã này.
Thứ nhất là lực mua trú ẩn an toàn do lo ngại căng thẳng hình sự gia tăng ở khu vực Trung Đông.
Thứ hai là lực mua vào liên tục của các ngân hàng trung ương, đặc biệt là Trung Quốc, nhằm đa dạng hóa dự trữ ngoại hối và giảm sự phụ thuộc vào đồng bạc xanh.
Hơn nữa, kỳ vọng Cục Dự trữ Liên bang Mỹ (Fed) sẽ sớm cắt giảm lãi suất cũng là một động lực quan trọng hỗ trợ giá vàng.
Tuy nhiên, đợt tăng nóng này cũng dấy lên những lo ngại về khả năng bong bóng vàng sẽ vỡ trong tương lai gần.
Nhiều nhà phân tích cảnh báo nhà đầu tư nên thận trọng và đa dạng hóa danh mục để tránh rủi ro không lường trước.
Tại Việt Nam, giá vàng SJC cũng tăng mạnh theo xu hướng chung, nhưng với biên độ rộng hơn nhiều do khan hiếm nguồn cung.
Khoảng cách giữa giá vàng trong nước và thế giới ngày càng nới rộng, lên mức cao nhất trong vòng vài tháng qua.
Ngân hàng Nhà nước đã lên kế hoạch tổ chức đấu thầu vàng miếng nhằm tăng cung, hy vọng thu hẹp khoảng cách chênh lệch này. 
Chúng ta hãy cùng chờ xem diễn biến tiếp theo trên thị trường."""

# Tạo văn bản rất dài bằng cách nhân bản text 100 lần ~ gần 20.000 từ
long_text = text_block * 100
chars_len = len(long_text)
words_len = len(long_text.split())

print(f"\n>> Benchmarking detect_sentences()...")
print(f"Text length: {chars_len:,} characters, {words_len:,} words.")

# Warm-up run
_ = detector.detect_sentences("Khởi động model để compile ONNX và Numpy. Xin chào thế giới.")

start_infer = time.time()
sentences = detector.detect_sentences(long_text)
end_infer = time.time()
time_taken = end_infer - start_infer

print(f"Total Sentences Extracted: {len(sentences):,}")
print(f"Execution Time: {time_taken:.4f} seconds")
print(f"Throughput: {chars_len / time_taken:,.0f} chars/sec")
print(f"Throughput: {words_len / time_taken:,.0f} words/sec")

print("\nSample first 3 sentences:")
for idx, s in enumerate(sentences[:3]):
    print(f"{idx+1}. {s}")
