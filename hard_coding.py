import os
import fitz
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import hashlib
from openai import OpenAI
from dotenv import load_dotenv

# =============== 환경 설정 및 전역 변수 ==================
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)
embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

FAISS_DIR = "./faiss_db"
os.makedirs(FAISS_DIR, exist_ok=True)

# =============== PDF 문제/정답 추출 함수 ================
def extract_qa_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    answer_blocks = re.findall(r'정\s*답[\s\.·]*([^\n]+)', text)
    answer_str = ' '.join(answer_blocks)
    answer_map = dict(re.findall(r'(\d+)\.\s*([①-④1-4])', answer_str))
    question_blocks = re.findall(
        r'(\d+)\.\s*([^\n]+?(?:\n[①-④][^\n]+){2,4})', text)
    qa_list = []
    for num, body in question_blocks:
        choices = re.findall(r'[①-④]\s*[^\n]+', body)
        question = re.sub(r'[①-④]\s*[^\n]+', '', body).strip()
        answer = answer_map.get(num, "")
        qa_list.append({
            "number": num,
            "question": question,
            "choices": choices,
            "answer": answer
        })
    return qa_list

# =============== 임베딩 & FAISS 인덱스 저장/불러오기 ===============
def get_faiss_index_paths(pdf_path):
    abs_path = os.path.abspath(pdf_path)
    pdf_hash = hashlib.md5(abs_path.encode()).hexdigest()
    faiss_path = os.path.join(FAISS_DIR, f"{pdf_hash}.faiss")
    embeddings_path = os.path.join(FAISS_DIR, f"{pdf_hash}_embeddings.npy")
    questions_path = os.path.join(FAISS_DIR, f"{pdf_hash}_qa.npy")
    return faiss_path, embeddings_path, questions_path

def save_faiss_index(index, path):
    faiss.write_index(index, path)

def load_faiss_index(path):
    return faiss.read_index(path)

def save_numpy(arr, path):
    np.save(path, arr)

def load_numpy(path):
    return np.load(path, allow_pickle=True)

def create_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def get_or_create_index_and_embeddings(pdf_path):
    faiss_path, embeddings_path, qa_path = get_faiss_index_paths(pdf_path)
    if os.path.exists(faiss_path) and os.path.exists(embeddings_path) and os.path.exists(qa_path):
        print("[INFO] 기존 인덱스/임베딩/QA 불러오는 중...")
        index = load_faiss_index(faiss_path)
        embeddings = load_numpy(embeddings_path)
        qa_list = load_numpy(qa_path).tolist()
    else:
        print("[INFO] 새로 임베딩/인덱스 생성 중...")
        qa_list = extract_qa_from_pdf(pdf_path)
        texts = [
            f"{q['question']} {' '.join(q['choices'])}" if q['choices'] else q['question']
            for q in qa_list
        ]
        embeddings = embedder.encode(texts, convert_to_numpy=True)
        index = create_faiss_index(embeddings)
        save_faiss_index(index, faiss_path)
        save_numpy(embeddings, embeddings_path)
        save_numpy(np.array(qa_list, dtype=object), qa_path)
    return index, embeddings, qa_list

# =============== 유사 문제 검색 ===============
def search_similar(embeddings, qa_list, user_query, top_k=3):
    query_emb = embedder.encode([user_query], convert_to_numpy=True)[0]
    sims = np.dot(embeddings, query_emb) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb))
    idx_sorted = np.argsort(-sims)[:top_k]
    return [qa_list[i] for i in idx_sorted]

# =============== 문제 생성 (GPT API) ===============
def generate_new_question(user_condition, similar_qas):
    context = ""
    for qa in similar_qas:
        choices = "\n".join(qa["choices"]) if qa["choices"] else ""
        context += f"문제: {qa['question']}\n보기:\n{choices}\n정답: {qa['answer']}\n\n"
    prompt = f""" 당신은 주어진 문제들을 바탕으로 새로운 4지선다형 문제를 생성하는 AI입니다.
{context}
이 문제들을 바탕으로  '{user_condition}'라는 조건 또는 키워드에 부합하는 새로운 4지선다형 문제와 보기를 4개, 그리고 정답을 만들어주세요.

[출력 예시]
문제: ...
보기:
① ...
② ...
③ ...
④ ...
정답: (보기 중 하나, 예: ③ ... )
"""
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "너는 측량 기사 시험 문제를 잘 만드는 AI야."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content.strip()

# =============== 메인 실행 ===============
if __name__ == "__main__":
    pdf_path = "app/rag/11-3 모의고사(2025).pdf"  # 파일경로 수정 필요
    index, embeddings, qa_list = get_or_create_index_and_embeddings(pdf_path)
    print(f"문제 추출: {len(qa_list)}개, 임베딩 shape: {embeddings.shape}")

    user_condition = input("출제할 문제 조건(예: 삼각측량, 거리측량, 수준측량 등): ").strip()
    similar_qas = search_similar(embeddings, qa_list, user_condition, top_k=3)
    print("\n🔍 검색된 유사 문제 예시:")
    for qa in similar_qas:
        print(f" - {qa['question']} {' '.join(qa['choices'])} (정답: {qa['answer']})")
    generated = generate_new_question(user_condition, similar_qas)
    print("\n=== 생성된 새 문제 ===")
    print(generated)
