import os
import sys
import argparse
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 1. 전역 설정
# 벡터 DB가 저장될 로컬 디렉토리
CHROMA_DB_PATH = "./chroma_db"
# 사용할 로컬 임베딩 모델의 이름.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# 분석할 파일 확장자 정의
# 분석 대상이 아닌 파일(예: 이미지, 바이너리)은 제외
CODE_EXTENSIONS = (
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".html",
    ".css",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".rs",
    ".php",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".sql",
    ".xml",
    ".properties",
    ".toml",
)


def load_documents(root_dir: str):
    """
    프로젝트 폴더를 재귀적으로 탐색하여 코드 파일을 로드
    """
    print(f"[{root_dir} 폴더 분석을 시작합니다.]")

    documents = []
    # os.walk를 사용하여 폴더를 재귀적으로 탐색.
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 숨김 폴더(.git, .venv, .vscode 등)는 탐색에서 제외.
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for file in filenames:
            # 숨김 파일(.gitignore, .env 등)은 로드에서 제외
            if file.startswith("."):
                continue

            if file.endswith(CODE_EXTENSIONS):
                filepath = os.path.join(dirpath, file)
                try:
                    # LangChain의 TextLoader를 사용하여 파일을 로드.
                    loader = TextLoader(filepath, encoding="utf-8")
                    docs = loader.load()
                    for doc in docs:
                        # 메타데이터에 파일 경로를 저장하여 검색 후 출처를 명시할 수 있게 함.
                        doc.metadata["source"] = filepath.replace(root_dir, "").lstrip(
                            os.sep
                        )
                        documents.append(doc)
                except Exception as e:
                    # 파일 인코딩 오류 등을 대비한 예외 처리
                    print(f"로드 실패: {filepath} - {e}")

    print(f"총 {len(documents)}개의 코드 파일을 메모리에 로드했습니다.")
    return documents


def index_codebase(documents: list[Document], project_name: str):
    """
    로드된 코드를 벡터화하여 프로젝트 전용 DB에 저장합니다.
    """
    if not documents:
        print("로드된 문서가 없습니다. 경로를 확인하세요.")
        return

    print("코드 문맥 분할을 시작합니다.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )

    texts = text_splitter.split_documents(documents)
    print(f"생성된 코드 청크: {len(texts)}개")

    print("벡터 임베딩 생성 중.")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # 프로젝트별 격리된 DB 경로 생성
    persist_dir = os.path.join(CHROMA_DB_PATH, project_name)
    print(f"데이터베이스 저장 경로: {persist_dir}")

    # 기존 데이터가 있으면 삭제하고 새로 생성 (Clean Build)
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_dir)
    db.persist()
    print(f"프로젝트 '{project_name}' 학습 완료. 이제 app.py를 사용할 수 있습니다.")

    return len(texts)


def embed_project(root_dir, project_name):
    """
    Streamlit 등 외부 앱에서 호출하기 위한 통합 함수.
    성공 여부와 메시지를 반환합니다.
    """
    try:
        if not os.path.isdir(root_dir):
            return False, f"❌ 경로가 유효하지 않습니다: {root_dir}"

        docs = load_documents(root_dir)
        if not docs:
            return (
                False,
                "⚠️ 로드된 파일이 없습니다. 경로 내에 소스 코드가 있는지 확인하세요.",
            )

        chunk_count = index_codebase(docs, project_name)
        return (
            True,
            f"✅ 학습 완료! 총 {len(docs)}개 파일, {chunk_count}개 청크가 저장되었습니다.",
        )
    except Exception as e:
        return False, f"❌ 오류 발생: {str(e)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="프로젝트 코드를 AI에게 학습시키는 도구"
    )
    parser.add_argument(
        "project_path", type=str, help="분석할 프로젝트 폴더의 절대 또는 상대 경로"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="default",
        help="프로젝트 식별 이름 (기본값: default)",
    )
    args = parser.parse_args()

    # 입력된 경로를 절대 경로로 변환하여 파일 로딩의 안정성을 높입니다.
    PROJECT_ROOT = os.path.abspath(args.project_path)

    if not os.path.isdir(PROJECT_ROOT):
        print(
            f"오류: 지정된 경로 '{PROJECT_ROOT}'를 찾을 수 없습니다. 올바른 폴더 경로를 입력하세요."
        )
        sys.exit(1)

    docs = load_documents(PROJECT_ROOT)

    if docs:
        prj_name = (
            args.name if args.name != "default" else os.path.basename(PROJECT_ROOT)
        )
        index_codebase(docs, prj_name)

        print(f"🚀 '{prj_name}' 프로젝트 학습을 시작합니다...")
        success, msg = embed_project(PROJECT_ROOT, prj_name)
        print(msg)
