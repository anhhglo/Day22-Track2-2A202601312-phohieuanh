# Evidence — Day 22: LangSmith + Prompt Versioning

**GitHub repository:** https://github.com/anhhglo/Day22-Track2-2A202601312-phohieuanh
**Sinh viên:** 2A202601312 — Phó Hiếu Anh
**LangSmith project:** `day22-2A202601312-phohieuanh` (region **APAC**)
**Provider:** OpenAI — `gpt-4o-mini` (LLM) + `text-embedding-3-small` (embeddings)

> Ghi chú: cột Latency trong ảnh `01_langsmith_traces.png` hiển thị giá trị âm là do đồng hồ
> hệ thống WSL2 trên máy chạy lab bị nhảy thời gian, không phải lỗi của pipeline. Nội dung
> input/output/token/cost của mọi trace vẫn đầy đủ và chính xác.

---

## Danh sách bằng chứng

| Tệp | Nội dung |
|---|---|
| `01_langsmith_traces.png` | LangSmith dashboard — traces của Bước 1 và Bước 2 (ảnh chụp khi project có 200 traces) |
| `01_rag_pipeline_log.txt` | Console log Bước 1 — 50/50 câu hỏi, 0 lỗi |
| `01_trace_count.txt` | Kết quả đếm traces qua LangSmith API — 300 root runs (100 + 200) |
| `02_prompt_hub.png` | Prompt Hub — 2 prompt `phohieuanh-rag-prompt-v1` và `-v2` |
| `02_ab_routing_log.txt` | Console log Bước 2 — 50 câu có nhãn v1/v2, phân bổ 19/31 |
| `03_ragas_scores.png` | Bảng so sánh RAGAS V1 vs V2 trên terminal |
| `03_ragas_report.json` | Báo cáo RAGAS (bản sao của `data/ragas_report.json`) |
| `03_ragas_console_log.txt` | Console log đầy đủ Bước 3 |
| `04_pii_demo_log.txt` | 6 test case PII detection & redaction |
| `04_json_demo_log.txt` | 5 test case JSON repair |

**Trace public (không cần đăng nhập):**
- Bước 1 `rag-query` — https://apac.smith.langchain.com/public/275a334b-553b-405b-ba13-98be7d400a25/r
- Bước 2 `ab-rag-query` v1 — https://apac.smith.langchain.com/public/5d191f9a-4366-4802-8f7f-5753be915b00/r
- Bước 2 `ab-rag-query` v2 — https://apac.smith.langchain.com/public/5bd43afb-21f4-4cf8-9e12-178045ac88a5/r

**Prompt Hub URLs** (đã chuyển sang public, handle `phohieuanh`):
- V1 — https://apac.smith.langchain.com/hub/phohieuanh/phohieuanh-rag-prompt-v1
- V2 — https://apac.smith.langchain.com/hub/phohieuanh/phohieuanh-rag-prompt-v2

> Cả 2 prompt truy cập được không cần đăng nhập, nhưng **chỉ trên host `apac.`** —
> host mặc định `smith.langchain.com` trả 404 vì workspace nằm ở region APAC.

---

## Kết quả RAGAS — V1 vs V2

| Chỉ số | V1 (ngắn gọn) | V2 (có cấu trúc) | Chênh lệch |
|---|---|---|---|
| faithfulness | **0.9622** | 0.9525 | +0.0097 cho V1 |
| answer_relevancy | **0.9064** | 0.9011 | +0.0053 cho V1 |
| context_recall | 1.0000 | 1.0000 | bằng nhau |
| context_precision | 0.9417 | **0.9483** | +0.0066 cho V2 |

**Cả 2 phiên bản đều đạt faithfulness ≥ 0.9** (vượt ngưỡng bắt buộc 0.8 và đạt mốc điểm thưởng).

## Phân tích: prompt tác động tới chỉ số nào?

**Chỉ faithfulness và answer_relevancy chịu tác động của system prompt.** Hai chỉ số còn lại đo
khâu truy xuất — cùng FAISS index, cùng retriever `k=3`, cùng 50 câu hỏi — nên prompt (chỉ tác động
ở bước sinh câu trả lời, *sau* khi retrieve) không thể làm chúng đổi. `context_recall` giữ đúng
1.0000 ở cả hai. Riêng `context_precision` lệch 0.0066 dù về lý thuyết phải bằng nhau: đây là nhiễu
của LLM đóng vai giám khảo (RAGAS chấm chỉ số này bằng LLM, `temperature=0` không đảm bảo tất định
tuyệt đối qua API), không phải khác biệt thật giữa 2 prompt.

**Lần chạy đầu tiên V2 chỉ đạt faithfulness 0.8669, thấp hơn V1 (0.9225) tới 0.0556.** Nguyên nhân
nằm ở hai chỉ thị trong `SYSTEM_V2` bản đầu: *"viết câu trả lời có tổ chức 3-5 câu"* và *"nêu rõ mức
độ chắc chắn dựa trên context"*. Faithfulness đo tỉ lệ mệnh đề trong câu trả lời được context chứng
minh — mà lệnh "nêu mức độ chắc chắn" buộc mô hình sinh ra câu tự đánh giá (*"Thông tin này có độ
tin cậy cao"*), thứ không hề tồn tại trong 3 đoạn context. Yêu cầu viết dài hơn cũng kéo theo câu
chuyển ý và câu khái quát hóa không có nguồn. V2 bị phạt đúng vì cái làm nó "chuyên nghiệp" hơn.

**Cách sửa và kết quả.** Bỏ mệnh lệnh tự đánh giá độ tin cậy, thay bằng ràng buộc bám nguồn rõ ràng
(*"mỗi câu phải bám vào một dữ kiện có trong context: không thêm ví dụ, không khái quát hóa, không
suy đoán"*), đồng thời giữ nguyên giọng chuyên gia và cấu trúc 3-5 câu để V2 vẫn khác V1 về phong
cách. Kết quả: **V2 tăng 0.0856** (0.8669 → 0.9525). Ràng buộc tương tự thêm vào V1 cũng nâng nó
**+0.0397** (0.9225 → 0.9622).

**Vì sao V1 vẫn nhỉnh hơn?** Khoảng cách thu hẹp còn 0.0097 — gần như bằng nhau. Phần dư còn lại đến
từ độ dài: V1 viết 2-4 câu, V2 viết 3-5 câu, mà mỗi câu thêm vào là một mệnh đề nữa phải được context
chứng minh. Với cùng chất lượng grounding, câu trả lời ngắn hơn luôn có lợi thế nhỏ về faithfulness.

**Kết luận thực tiễn:** trong hệ RAG chấm theo faithfulness, thủ phạm hạ điểm không phải là giọng văn
hay độ dài, mà là những chỉ thị buộc mô hình **sinh nội dung nó phải tự nghĩ ra** — điển hình là yêu
cầu tự đánh giá độ tin cậy. Tách yêu cầu định dạng khỏi yêu cầu nội dung, rồi thêm ràng buộc bám nguồn
tường minh, là đủ để giữ phong cách mong muốn mà không mất điểm grounding.
