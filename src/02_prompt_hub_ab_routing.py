"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "phohieuanh-rag-prompt-v1"
PROMPT_V2_NAME = "phohieuanh-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# V1 — phong cách ngắn gọn, thân thiện: trả lời thẳng vào vấn đề trong 2-4 câu.
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Chỉ dùng context dưới đây để trả lời. "
    "Giữ câu trả lời ngắn gọn và trực tiếp (2-4 câu), dùng ngôn ngữ đời thường, "
    "không liệt kê đầu dòng. Nếu context không chứa thông tin cần thiết, "
    "hãy nói thẳng là bạn không biết.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

# V2 — phong cách chuyên gia, có cấu trúc: phân tích context rồi trả lời 3-5 câu.
SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Quy trình trả lời: "
    "1) đọc kỹ context và xác định các dữ kiện liên quan, "
    "2) viết câu trả lời rõ ràng, có tổ chức, dùng thuật ngữ chính xác (3-5 câu), "
    "3) nêu rõ mức độ chắc chắn dựa trên context. "
    "Tuyệt đối không suy đoán ngoài context; nếu thiếu dữ kiện hãy nói rõ điều đó.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def _push_one(client: Client, label: str, name: str, template, description: str):
    """
    Push 1 prompt lên Hub.

    LangSmith trả 409 "Nothing to commit" khi prompt không đổi so với commit gần nhất —
    đây KHÔNG phải lỗi: prompt vẫn nằm trên Hub, chỉ là không tạo commit mới.
    """
    try:
        url = client.push_prompt(name, object=template, description=description)
        print(f"✅ Đã push {label} → {url}")
    except Exception as e:
        if "Nothing to commit" in str(e) or "409" in str(e):
            print(f"✅ {label} đã có sẵn trên Hub (nội dung không đổi → không tạo commit mới): {name}")
        else:
            print(f"⚠️  {label} lỗi: {e}")


def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    Mỗi lần nội dung prompt thay đổi sẽ tạo một commit mới cho prompt tương ứng.
    """
    _push_one(client, "V1", PROMPT_V1_NAME, PROMPT_V1,
              "V1 – phong cách ngắn gọn, thân thiện (2-4 câu)")
    _push_one(client, "V2", PROMPT_V2_NAME, PROMPT_V2,
              "V2 – phong cách chuyên gia, có cấu trúc (3-5 câu)")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng.

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"↓ Đã pull '{PROMPT_V1_NAME}' từ Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"↓ Đã pull '{PROMPT_V2_NAME}' từ Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"ℹ️  Dùng local fallback cho '{PROMPT_V2_NAME}'")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)

    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Bước:
      a) Retrieve top-3 docs từ retriever
      b) Ghép page_content thành context string
      c) Chạy (prompt | llm | StrOutputParser())
      d) Trả về {"question": ..., "answer": ..., "version": ...}
    """
    docs = retriever.invoke(question)

    context = "\n\n".join(doc.page_content for doc in docs)

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  context,
        "question": question,
    })

    return {"question": question, "answer": answer, "version": version}


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    push_prompts_to_hub(client)

    prompts = pull_prompts_from_hub(client)

    # Chứng minh routing là TẤT ĐỊNH: gọi lại nhiều lần cùng request_id → luôn cùng version
    all_ids = [f"req-{i:04d}" for i in range(len(SAMPLE_QUESTIONS))]
    demo_ids = [
        next(r for r in all_ids if get_prompt_version(r) == PROMPT_V1_NAME),   # ví dụ rơi vào V1
        next(r for r in all_ids if get_prompt_version(r) == PROMPT_V2_NAME),   # ví dụ rơi vào V2
    ]
    print("\n🔁 Kiểm tra tính tất định của router (gọi lại 3 lần mỗi request_id):")
    for rid in demo_ids:
        votes = {get_prompt_version(rid) for _ in range(3)}
        print(f"   {rid} → {votes.pop()}  |  luôn cho cùng kết quả: {len(votes) == 0}")

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        try:
            result = ask_ab(retriever, llm, prompt, question, version_tag)
        except Exception as e:                      # fallback: không để 1 câu lỗi chặn cả run
            result = {"question": question, "answer": f"[LỖI] {e}", "version": version_tag}

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")
        print(f"     → {str(result['answer'])[:90]}")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
