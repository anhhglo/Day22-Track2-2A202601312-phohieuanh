"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


# ── 1. Prompt Templates (copy nguyên văn từ Bước 2) ───────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Chỉ dùng context dưới đây để trả lời. "
    "Giữ câu trả lời ngắn gọn và trực tiếp (2-4 câu), dùng ngôn ngữ đời thường, "
    "không liệt kê đầu dòng. Mọi khẳng định phải có trong context — không thêm "
    "ví dụ, không suy rộng. Nếu context không chứa thông tin cần thiết, "
    "hãy nói thẳng là bạn không biết.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Quy trình trả lời: "
    "1) đọc kỹ context và xác định các dữ kiện liên quan, "
    "2) viết câu trả lời rõ ràng, có tổ chức, dùng thuật ngữ chính xác (3-5 câu). "
    "Mỗi câu bạn viết phải bám vào một dữ kiện có trong context: không thêm ví dụ, "
    "không khái quát hóa, không suy đoán và không tự đánh giá độ tin cậy. "
    "Nếu context thiếu dữ kiện, hãy nói rõ phần nào không xác định được.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs = retriever.invoke(question)

    contexts = [doc.page_content for doc in docs]   # phải là list[str] !

    # Chỉ ghép chuỗi khi truyền vào {context} của prompt
    ctx_str = "\n\n".join(contexts)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy 50 câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        try:
            out = run_rag(retriever, llm, prompt, qa["question"])
        except Exception as e:                      # fallback: giữ nguyên số sample = 50
            print(f"  ⚠️  Câu {i} lỗi: {e}")
            out = {"answer": "", "contexts": []}

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],   # list[str] !
        })
        print(f"  [{i:02d}/50] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]

    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def run_ragas_eval(rag_results: list, version: str) -> dict:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.
    Trả về: dict {metric_name: mean_score}

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
    )

    # Tính mean score cho mỗi metric
    # result["faithfulness"] trả về list of floats → dùng np.mean()
    scores = {}
    for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        raw   = result[key]
        raw   = raw if isinstance(raw, (list, tuple, np.ndarray)) else [raw]
        valid = [v for v in raw if v is not None and not (isinstance(v, float) and np.isnan(v))]
        scores[key] = float(np.mean(valid)) if valid else 0.0

    # In kết quả
    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores


# ── 6. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores = run_ragas_eval(v1_results, "v1")
    v2_scores = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winner  = "← V1" if s1 > s2 else "← V2"
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # Lưu báo cáo vào data/ragas_report.json
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
    }
    report_path = Path(__file__).parent.parent / "data" / "ragas_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")


if __name__ == "__main__":
    main()


# ── PHÂN TÍCH KẾT QUẢ (lần chạy ngày 25/08/2026, gpt-4o-mini) ─────────────
#
#   Metric              V1 (ngắn gọn)   V2 (có cấu trúc)
#   faithfulness             0.9622           0.9525   ← cả hai >= 0.9
#   answer_relevancy         0.9064           0.9011
#   context_recall           1.0000           1.0000     hòa
#   context_precision        0.9417           0.9483
#
# • context_recall/precision đo khâu truy xuất: cùng FAISS index, cùng retriever k=3
#   → system prompt (tác động sau khi retrieve) không làm chúng đổi. Chênh lệch 0.0066
#   ở context_precision là nhiễu của LLM giám khảo, không phải khác biệt thật.
#
# • Bản SYSTEM_V2 ĐẦU TIÊN chỉ đạt faithfulness 0.8669 vì có lệnh "nêu rõ mức độ chắc
#   chắn" — lệnh này buộc mô hình sinh câu tự đánh giá, thứ không có trong context, nên
#   bị RAGAS tính là mệnh đề không được chứng minh.
#
# • Sau khi thay lệnh đó bằng ràng buộc bám nguồn tường minh (giữ nguyên giọng chuyên gia
#   và cấu trúc 3-5 câu để V2 vẫn khác V1 về phong cách): V2 tăng +0.0856 lên 0.9525.
#   Ràng buộc tương tự thêm vào V1 nâng V1 +0.0397 lên 0.9622.
#
# • V1 vẫn nhỉnh hơn 0.0097 do viết ngắn hơn (2-4 câu so với 3-5 câu): mỗi câu thêm vào
#   là một mệnh đề nữa phải được context chứng minh.
#
# Xem phân tích đầy đủ tại evidence/README.md
