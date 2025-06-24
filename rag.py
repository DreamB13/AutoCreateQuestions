import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

DB_DIR = "./chroma_db"
PDF_PATH = "app/rag/11-3 모의고사(2025).pdf"

if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    docs = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
else:
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

llm = ChatOpenAI(model="gpt-4o")

custom_prompt = PromptTemplate(
    template="""
당신은 아래 문서에 등장하는 문제와 보기, 정답 스타일을 정확하게 모방하여, 
문서에 실제 등장한 정보를 바탕으로 새로운 4지선다형 문제를 한 개 만들어주세요.
창작/상상 금지, pdf 내 문장/용어만 응용.
반드시 문서에 실제 등장하는 지식(내용)만 사용하고, 문제와 보기는 문서에서 쓰인 문장이나 용어, 형식을 참고해 작성하세요. 
창의적으로 새로운 내용을 추가하지 마세요.
사용자 조건에 맞는 문제에 나오는 보기들을 응용해서 사용해주세요.

[출력 예시]
문제: ...
보기:
① ...
② ...
③ ...
④ ...
정답: (보기 중 하나, 예: ③ ... )

문서 내용:
{context}

사용자 질문: {question}
""",
    input_variables=["context", "question"],
)

# MMR 기반 retriever 단 한 번만 생성!
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt}
)

user_query = input("문제 생성 조건을 입력하세요: ").strip()

# 검색된 청크를 확인하기 위한 설정 (retriever 재사용)
docs = retriever.get_relevant_documents(user_query)

print("\n======= [사용된 청크 목록] =======")
for i, doc in enumerate(docs, 1):
    print(f"\n--- [청크 {i}] ---\n{doc.page_content}")

# LLM 호출
response = qa_chain.invoke({"query": user_query})

print("\n======= [LLM 출력 결과] =======")
print(response['result'])
