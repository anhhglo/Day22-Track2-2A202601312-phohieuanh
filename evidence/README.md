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
| `01_langsmith_traces.png` | LangSmith dashboard — 200 traces (50 `rag-query` + 150 `ab-rag-query`) |
| `01_rag_pipeline_log.txt` | Console log Bước 1 — 50/50 câu hỏi, 0 lỗi |
| `01_trace_count.txt` | Kết quả đếm traces qua LangSmith API — 200 root runs (50 + 150) |
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
| faithfulness | **0.9225** | 0.8669 | +0.0556 cho V1 |
| answer_relevancy | **0.9160** | 0.9016 | +0.0144 cho V1 |
| context_recall | 1.0000 | 1.0000 | bằng nhau |
| context_precision | 0.9450 | 0.9450 | bằng nhau |

Cả 2 phiên bản đều vượt ngưỡng faithfulness ≥ 0.8. **V1 thắng ở 2/4 chỉ số**, 2 chỉ số còn lại hòa.

## Phân tích: vì sao V1 cao điểm hơn?

**Hai chỉ số retrieval hòa nhau là điều được dự đoán trước.** `context_recall` và `context_precision` chỉ đo chất lượng của khâu truy xuất — cùng một FAISS index, cùng retriever `k=3`, cùng 50 câu hỏi. System prompt chỉ tác động tới bước sinh câu trả lời, nằm *sau* khâu retrieve, nên không thể làm thay đổi 2 chỉ số này. Kết quả giống hệt nhau tới 9 chữ số thập phân xác nhận pipeline hoạt động đúng như thiết kế.

**V1 thắng faithfulness (0.9225 vs 0.8669) vì prompt của nó ít khuyến khích mô hình nói thêm.** Faithfulness đo tỉ lệ mệnh đề trong câu trả lời được context chứng minh. V1 yêu cầu "ngắn gọn, trực tiếp, 2-4 câu" → mô hình gần như chỉ diễn đạt lại nội dung đã retrieve. V2 lại yêu cầu "viết câu trả lời có tổ chức 3-5 câu" và "nêu rõ mức độ chắc chắn" — hai chỉ thị này đẩy mô hình sinh thêm câu chuyển ý, câu khái quát hóa và câu tự đánh giá độ tin cậy. Những mệnh đề thêm vào đó không có trong 3 đoạn context, nên RAGAS đếm chúng là không được chứng minh và trừ điểm. Nói cách khác, V2 bị phạt đúng vì cái làm nó "chuyên nghiệp" hơn.

**Chênh lệch answer_relevancy nhỏ hơn nhiều (0.0144)** vì chỉ số này đo mức độ câu trả lời bám sát câu hỏi, mà cả 2 prompt đều buộc mô hình chỉ dùng context. Phần dư thừa của V2 làm loãng nhẹ độ liên quan chứ không lạc đề.

**Kết luận thực tiễn:** với hệ RAG chấm theo faithfulness, prompt càng nhiều chỉ thị "trình bày đẹp" thì càng dễ mất điểm grounding. Muốn giữ giọng văn chuyên nghiệp của V2 mà không tụt faithfulness, nên tách yêu cầu định dạng ra khỏi yêu cầu nội dung — ví dụ bỏ mệnh lệnh "nêu mức độ chắc chắn" (thứ mô hình phải tự suy đoán, không có trong context) và giữ lại phần cấu trúc câu.
