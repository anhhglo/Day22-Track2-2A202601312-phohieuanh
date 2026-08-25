# Bài nộp — Day 22: LangSmith + Prompt Versioning

**Sinh viên:** Phó Hiếu Anh — 2A202601312
**Repository:** https://github.com/anhhglo/Day22-Track2-2A202601312-phohieuanh
**Ngày hoàn thành:** 25/08/2026

---

## 1. Các đường dẫn nộp bài

| Mục | Đường dẫn | Truy cập |
|---|---|---|
| **GitHub repository** | https://github.com/anhhglo/Day22-Track2-2A202601312-phohieuanh | Public |
| **LangSmith project** | [day22-2A202601312-phohieuanh](https://apac.smith.langchain.com/o/e877a741-b9e3-43d7-b2c0-118e447c6b10/projects/p/9583aecb-6391-443c-8e2d-61d0c754a951) | Cần đăng nhập workspace — xem [trace public](#trace-công-khai-không-cần-đăng-nhập) bên dưới |
| **Prompt Hub — V1** | https://apac.smith.langchain.com/hub/phohieuanh/phohieuanh-rag-prompt-v1 | Public, không cần đăng nhập |
| **Prompt Hub — V2** | https://apac.smith.langchain.com/hub/phohieuanh/phohieuanh-rag-prompt-v2 | Public, không cần đăng nhập |

### Trace công khai (không cần đăng nhập)

Ba trace tiêu biểu đã được share public để người chấm xem trực tiếp cấu trúc chain,
context được truy xuất và câu trả lời:

| Trace | Câu hỏi | Link |
|---|---|---|
| Bước 1 — `rag-query` | *What information do LangSmith traces capture?* | https://apac.smith.langchain.com/public/275a334b-553b-405b-ba13-98be7d400a25/r |
| Bước 2 — `ab-rag-query` (prompt **v1**) | *What is the transformer architecture?* | https://apac.smith.langchain.com/public/5d191f9a-4366-4802-8f7f-5753be915b00/r |
| Bước 2 — `ab-rag-query` (prompt **v2**) | *How does LangSmith help monitor production LLM applications?* | https://apac.smith.langchain.com/public/5bd43afb-21f4-4cf8-9e12-178045ac88a5/r |

Mở trace Bước 1 sẽ thấy đầy đủ chuỗi run con: `VectorStoreRetriever` (3 documents) →
`ChatPromptTemplate` → `ChatOpenAI` → `StrOutputParser` — tức trace chứa cả câu hỏi,
context truy xuất được và câu trả lời (tiêu chí 1.4).

> ⚠️ **Lưu ý về region:** workspace nằm ở **APAC**. Mọi đường dẫn phải dùng host `apac.smith.langchain.com`;
> host mặc định `smith.langchain.com` sẽ trả **404** (với prompt) hoặc **403** (với API) dù link hoàn toàn đúng.

**Nếu không mở được LangSmith project:** toàn bộ số liệu đã được lưu offline trong repo —
xem [`evidence/01_langsmith_traces.png`](evidence/01_langsmith_traces.png) (ảnh giao diện) và
[`evidence/01_trace_count.txt`](evidence/01_trace_count.txt) (kết quả đếm trực tiếp qua LangSmith API).

---

## 2. Kết quả tổng hợp

| Hạng mục | Kết quả |
|---|---|
| Traces trên LangSmith | **200** root runs — 50 `rag-query` (Bước 1) + 150 `ab-rag-query` (Bước 2) |
| Prompt versions trên Hub | 2 (`phohieuanh-rag-prompt-v1`, `phohieuanh-rag-prompt-v2`) |
| A/B routing | Tất định qua MD5(`request_id`) — phân bổ 19 câu V1 / 31 câu V2 |
| RAGAS faithfulness | **V1 = 0.9225** · V2 = 0.8669 → đạt mục tiêu ≥ 0.8 |
| RAGAS answer_relevancy | V1 = 0.9160 · V2 = 0.9016 |
| RAGAS context_recall | V1 = 1.0000 · V2 = 1.0000 |
| RAGAS context_precision | V1 = 0.9450 · V2 = 0.9450 |
| Guardrails | 6/6 test PII redact đúng · 5/5 test JSON (kể cả trường hợp fallback) |
| Cấu hình chạy | OpenAI `gpt-4o-mini` + `text-embedding-3-small`, FAISS 107 chunks, retriever k=3 |

Phân tích chi tiết vì sao V1 cao điểm hơn V2: [`evidence/README.md`](evidence/README.md).

---

## 3. Bản đồ tiêu chí chấm điểm → bằng chứng

### Nhiệm vụ 1 — RAG Pipeline với LangSmith (25đ)

| Tiêu chí | Nằm ở đâu |
|---|---|
| 1.1 Chunk + index FAISS | [`src/utils/data_loader.py`](src/utils/data_loader.py) · [`src/01_langsmith_rag_pipeline.py#setup_vectorstore`](src/01_langsmith_rag_pipeline.py) — 107 chunks |
| 1.2 RAG chain LCEL | [`src/01_langsmith_rag_pipeline.py#build_rag_chain`](src/01_langsmith_rag_pipeline.py) |
| 1.3 `@traceable` + ≥50 traces | [`evidence/01_langsmith_traces.png`](evidence/01_langsmith_traces.png) · [`evidence/01_trace_count.txt`](evidence/01_trace_count.txt) |
| 1.4 Trace chứa question + context + answer | Ảnh trên (cột Input/Output) · mỗi trace có run con `VectorStoreRetriever` trả 3 documents |

### Nhiệm vụ 2 — Prompt Hub & A/B Routing (25đ)

| Tiêu chí | Nằm ở đâu |
|---|---|
| 2.1 Hai prompt khác ngữ nghĩa | [`src/02_prompt_hub_ab_routing.py`](src/02_prompt_hub_ab_routing.py) — `SYSTEM_V1` (ngắn gọn) vs `SYSTEM_V2` (chuyên gia, có cấu trúc) |
| 2.2 Push lên Hub, hiện trong UI | [`evidence/02_prompt_hub.png`](evidence/02_prompt_hub.png) · 2 link public ở mục 1 |
| 2.3 Pull từ Hub khi chạy | [`evidence/02_ab_routing_log.txt`](evidence/02_ab_routing_log.txt) — dòng `↓ Đã pull ... từ Hub` |
| 2.4 Routing tất định | Cùng log — mục "Kiểm tra tính tất định", mỗi `request_id` gọi lại 3 lần đều ra cùng version |
| 2.5 Log có nhãn version | Cùng log — 50 dòng `[NN] [prompt-v1/v2] ...`, tổng kết `V1=19 | V2=31` |

### Nhiệm vụ 3 — RAGAS Evaluation (25đ)

| Tiêu chí | Nằm ở đâu |
|---|---|
| 3.1 50 QA × 2 version | [`evidence/03_ragas_console_log.txt`](evidence/03_ragas_console_log.txt) — 2 vòng `[01/50]…[50/50]` |
| 3.2 `SingleTurnSample` đúng trường | [`src/03_ragas_evaluation.py#build_ragas_dataset`](src/03_ragas_evaluation.py) |
| 3.3 Đủ 4 metrics | [`evidence/03_ragas_scores.png`](evidence/03_ragas_scores.png) · [`evidence/03_ragas_report.json`](evidence/03_ragas_report.json) |
| 3.4 Faithfulness ≥ 0.8 | V1 = 0.9225 ✅ |
| 3.5 Lưu `ragas_report.json` | [`data/ragas_report.json`](data/ragas_report.json) (bản gốc) · [`evidence/03_ragas_report.json`](evidence/03_ragas_report.json) (bản nộp) |
| Thưởng: phân tích V1 vs V2 | [`evidence/README.md`](evidence/README.md) · khối bình luận cuối [`src/03_ragas_evaluation.py`](src/03_ragas_evaluation.py) |

### Nhiệm vụ 4 — Guardrails AI Validators (25đ)

| Tiêu chí | Nằm ở đâu |
|---|---|
| 4.1–4.4 PIIDetector | [`src/04_guardrails_validator.py`](src/04_guardrails_validator.py) — `@register_validator`, 4 loại PII bằng regex, `on_fail` trong constructor · [`evidence/04_pii_demo_log.txt`](evidence/04_pii_demo_log.txt) (6 test case) |
| 4.5–4.8 JSONFormatter | Cùng file — sửa fences / nháy đơn / dấu phẩy thừa + JSON dự phòng · [`evidence/04_json_demo_log.txt`](evidence/04_json_demo_log.txt) (5 test case) |

### An toàn

- Không có API key nào trong mã nguồn đã commit — chỉ dùng biến môi trường qua [`src/config.py`](src/config.py).
- `.env` nằm trong [`.gitignore`](.gitignore), chưa từng được commit. Chỉ [`.env.example`](.env.example) (không chứa giá trị thật) được đưa lên.

---

## 4. Ghi chú kỹ thuật — 3 điểm phải sửa khác với đề bài

Ba chỗ làm đúng theo hướng dẫn trong `Guide.md` sẽ **không chạy được** với phiên bản thư viện hiện tại:

1. **`ragas 0.4.x` không tương thích LangChain 1.x.** `requirements.txt` gốc để `langchain>=0.3.0` không chặn trên, pip kéo LangChain 1.x và `import ragas` chết với `ModuleNotFoundError: langchain_community.chat_models.vertexai`. Đã chặn `<1.0` trong [`requirements.txt`](requirements.txt).

2. **`PassResult(value_override=...)` bị bỏ qua ở `guardrails 0.11`.** Với string guard, `Guard.validate()` không áp dụng `value_override` — làm đúng theo đề thì PII và JSON *không hề* được thay thế, `validated_output` y hệt input. Đã chuyển sang `FailResult(fix_value=...)` để `OnFailAction.FIX` thực sự hoạt động.

3. **Regex PHONE của đề sót dấu ngoặc mở.** `\b` không khớp trước ký tự `(`, nên `(555) 867-5309` chỉ được redact thành `([PHONE_REDACTED]`. Đã thay bằng lookaround `(?<!\d)` / `(?!\d)`.

Ngoài ra: key LangSmith region APAC bị **403 Forbidden** khi dùng endpoint mặc định (US) dù key hoàn toàn hợp lệ — đã ghi chú cả 3 region vào [`.env.example`](.env.example).

---

## 5. Cách chạy lại

```bash
git clone https://github.com/anhhglo/Day22-Track2-2A202601312-phohieuanh.git
cd Day22-Track2-2A202601312-phohieuanh

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # điền LANGCHAIN_API_KEY + OPENAI_API_KEY
                          # nhớ đổi LANGCHAIN_ENDPOINT theo region của bạn
cd src && python config.py   # xác minh cấu hình

python run_all.py            # chạy cả 4 bước (Bước 3 mất 45-60 phút)
python run_all.py --step 4   # hoặc chạy riêng 1 bước
```
