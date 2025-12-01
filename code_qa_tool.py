import os
import sys
from operator import itemgetter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda,
    RunnableParallel,
)
from langchain_classic.schema import StrOutputParser

# 1. 전역 설정
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# 2. Ollama LLM 설정
OLLAMA_MODEL_NAME = "codellama"
OLLAMA_BASE_URL = "http://localhost:11434"  # ollama API 주소 (기본값)

# 3. RAG 프롬프트 템플릿
PROMPT_TEMPLATE = """
당신은 이 프로젝트의 코드를 모두 알고 있는 경험 많은 소프트웨어 엔지니어입니다.
주어진 맥락(Context)을 기반으로 사용자 질문에 대해 정확하고 간결하게 답변해 주세요.

- 답변은 반드시 제공된 맥락 내의 코드와 정보를 사용해야 합니다.
- 만약 맥락에 정보가 없다면, "제공된 코드베이스 맥락 내에서 답변을 찾을 수 없습니다."라고 응답하십시오.
- 코드를 제안할 때는 원본 파일 경로를 명시하고, 해당 코드 조각을 재현해 주세요.

맥락:
---
{context}
---

질문: {question}

답변:
"""


def format_docs(docs):
    """검색된 문서(list[Document])를 LLM에 전달할 문자열로 변환합니다."""
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


def create_qa_chain():
    """
    LCEL 기반 RAG 파이프라인 (DB 연결, LLM 연결, 체인 생성)을 구축합니다.
    """
    # 1. 임베딩 모델 로드
    try:
        print("임베딩 모델을 로드합니다.")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"임베딩 모델 로드 실패: {e}")
        print("HuggingFace 모델 다운로드 및 인터넷 연결을 확인하세요.")
        return None, None

    # 2. Local ChromaDB에 연결 및 검색기(Retriever) 설정
    if not os.path.exists(CHROMA_DB_PATH):
        print(f"벡터 DB 폴더 '{CHROMA_DB_PATH}'를 찾을 수 없습니다.")
        print("code_indexer.py를 먼저 실행하여 프로젝트를 색인해야 합니다.")
        return None, None

    print(f"벡터 DB({CHROMA_DB_PATH})에 연결했습니다.")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH, embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 3. Ollama Local LLM 서버에 연결
    try:
        print(f"Ollama 서버({OLLAMA_BASE_URL})에 연결 시도.")
        llm = Ollama(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL)
        # 테스트
        llm.invoke("안녕", temperature=0.0)
    except Exception as e:
        print(f"Ollama LLM 연결 실패: {e}")
        print(
            f"1. Ollama가 실행 중인지 확인하세요. 2. '{OLLAMA_MODEL_NAME}' 모델이 Ollama에 다운로드되어 있는 지 확인하세요."
        )
        return None, None

    print(f"Ollama LLM({OLLAMA_MODEL_NAME})에 연결했습니다.")

    # 4. Prompt Template 설정
    custom_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # 5. LCEL 기반 RAG 체인 구축
    # 5-1. 검색된 문서를 가져오면서 동시에 입력된 질문을 유지하는 맵핑 체인
    setup_and_retrieval = RunnableParallel(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
    )
    # 5-2. 답변 생성 체인: 검색 결과를 프롬프트에 넣고 LLM 호출
    generation_chain = setup_and_retrieval | custom_prompt | llm | StrOutputParser()

    # 5-3. 최종 RAG 체인: 답변과 원본 문서 검색 결과를 모두 반환
    rag_chain_with_source = RunnableParallel(
        answer=generation_chain, source_documents=retriever
    )

    return rag_chain_with_source


def run_qa_tool(rag_chain_with_source):
    """
    사용자로부터 질문을 입력받고 QA 체인을 실행합니다.
    """
    if not rag_chain_with_source:
        return

    print("\n-----")
    print(
        "로컬 코드 QA 도구가 준비되었습니다. 질문을 입력하세요. (종료: 'exit' 또는 'quit')"
    )
    print("\n-----")

    while True:
        try:
            query = input("질문: ")
        except EOFError:
            print("입력 종료. 도구를 종료합니다.")
            break

        if query.lower() in ["exit", "quit"]:
            print("도구를 종료합니다.")
            break

        if not query.strip():
            continue

        try:
            print("답변을 생성 중입니다. 잠시만 기다려주세요.")
            # 1. RAG 체인을 실행하여 답변을 생성합니다.
            result = rag_chain_with_source.invoke(query)

            # 결과 출력
            print("\n AI 답변:")
            print(result["answer"].strip())

            # 출처 코드 문서 출력
            print("\n 참고한 코드 출처:")
            source_files = set()
            for doc in result["source_documents"]:
                source_files.add(doc.metadata.get("source", "알 수 없음"))

            for source in sorted(list(source_files)):
                print(f"    - {source}")

            print("\n-----")

        except Exception as e:
            print(f"\n 처리 중 오류가 발생했습니다. Ollama 서버를 확인하세요. : {e}")
            print("-----")


if __name__ == "__main__":
    rag_chain_with_source = create_qa_chain()
    run_qa_tool(rag_chain_with_source)
