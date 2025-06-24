import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from paddleocr import PaddleOCR
import fitz  # PyMuPDF

DB_DIR = "./chroma_db"
PDF_PATH = "app/rag/11-3 모의고사(2025).pdf"

# OCR 모델 준비 (한글+영문)
ocr = PaddleOCR(use_angle_cls=True, lang='korean')

def patch_broken_formula(page_num, orig_text):
    if any(char in orig_text for char in ["□", "�", "◆","",""]):
        doc = fitz.open(PDF_PATH)
        page = doc[page_num]
        img = page.get_pixmap(dpi=300)
        img_path = f"ocr_page_{page_num+1}.png"
        img.save(img_path)
        # 'cls=True'를 제거!
        result = ocr.ocr(img_path)
        ocr_text = "\n".join([line[1][0] for line in result[0]])
        print(f"[수식 OCR 적용] PAGE {page_num+1}")
        os.remove(img_path)
        return ocr_text
    return orig_text


# 1. 임베딩/DB 생성 or 기존 DB 불러오기
if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    # 텍스트+OCR 하이브리드
    patched_pages = []
    for i, page in enumerate(pages):
        txt = page.page_content
        patched = patch_broken_formula(i, txt)
        # 패치된 텍스트로 새로운 Document 객체 생성
        page.page_content = patched
        patched_pages.append(page)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    docs = splitter.split_documents(patched_pages)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    vectordb.persist()
    with open("patched_output.txt", "w", encoding="utf-8") as f:
        for i, page in enumerate(patched_pages):
            f.write(f"\n==== [PAGE {i+1}] ====\n")
            f.write(page.page_content)
            f.write("\n\n")

    print("패치된 전체 텍스트가 patched_output.txt로 저장되었습니다.")
else:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

print("임베딩/DB 구성 완료 (수식 OCR 자동 패치 적용)")

