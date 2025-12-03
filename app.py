import os
import csv
import time
from datetime import datetime
import streamlit as st

try:
    from code_indexer import embed_project
except ImportError:
    st.error(
        "'code_indexer.py' 파일을 찾을 수 없습니다. 같은 폴더에 있는지 확인하세요."
    )
    st.stop()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 설정
BASE_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL_NAME = "qwen2.5-coder:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
FEEDBACK_FILE = "rag_feedback.csv"

# 페이지 설정
st.set_page_config(page_title="AI Co-Developer", layout="wide")
st.markdown(
    """
### AI Co-Developer: 도메인을 넘나드는 코딩 파트너
백엔드, 프론트엔드, DB 등 **전체 프로젝트 맥락을 이해하고 협업**하는 로컬 AI 도구입니다.
"""
)


# 유틸리티 함수
def get_existing_projects():
    """chroma_db 폴더를 스캔하여 학습된 프로젝트 목록을 반환합니다."""
    if not os.path.exists(BASE_DB_PATH):
        return []
    # 폴더이면서 숨김 파일이 아닌 것들만 리스트업
    projects = [
        d
        for d in os.listdir(BASE_DB_PATH)
        if os.path.isdir(os.path.join(BASE_DB_PATH, d)) and not d.startswith(".")
    ]
    return sorted(projects)


# 함수: 파일 트리 생성 (Context Map)
def generate_file_tree(startpath):
    """프로젝트의 전체 지도를 그려주어, 개발자가 어디를 수정해야 할지 위치를 파악하게 돕습니다."""
    if not startpath or not os.path.exists(startpath):
        return "(경로가 설정되지 않았거나 유효하지 않습니다.)"
    tree_str = ""
    for root, dirs, files in os.walk(startpath):
        dirs[:] = [d for d in dirs if not d.startswith(".")]  # 숨김 폴더 제외
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        base = os.path.basename(root)
        if base:
            subindent = " " * 4 * (level + 1)
            tree_str += f"{indent} {base}/\n"
            for f in files:
                if not f.startswith("."):
                    tree_str += f"{subindent} {f}\n"
    return tree_str if tree_str else "(구조를 읽을 수 없습니다.)"


# 사이드바: 사용자 및 서버 설정
with st.sidebar:
    st.header("프로젝트 선택")

    # 1. 기존 프로젝트 목록 가져오기
    existing_projects = get_existing_projects()

    # 2. 프로젝트 선택 방식
    tab1, tab2 = st.tabs(["불러오기", "새로만들기"])

    with tab1:
        if existing_projects:
            select_project = st.selectbox(
                "학습된 프로젝트 선택", existing_projects, index=0
            )
            project_name = select_project
            st.success(f"'{project_name}' 로드 준비 완료")
        else:
            st.info("아직 학습된 프로젝트가 없습니다. '새로 만들기' 탭을 이용하세요.")
            project_name = None
    with tab2:
        new_project_name = st.text_input(
            "새 프로젝트 이름(DB명)", placeholder="예: my-new-project"
        )
        new_root_path = st.text_input(
            "실제 파일 경로 (Root Path)", placeholder="C:/Work/MyProject"
        )

        if st.button("DB 학습 시작", type="primary"):
            if not new_project_name or not new_root_path:
                st.error("이름과 경로를 모두 입력하세요.")
            else:
                with st.spinner(f"'{new_project_name}' 학습 중."):
                    success, msg = embed_project(new_root_path, new_project_name)
                    if success:
                        st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
        # 탭2가 활성화 되었고 입력값이 있으면 그것을 프로젝트 이름으로 사용
        if new_project_name and not project_name:
            project_name = new_project_name

    # 파일 트리 경로 (불러오기 모드일 때도 트리를 보고 싶다면 경로 입력 필요)
    # DB에는 파일 내용만 있고 트리 구조를 그리기 위한 실제 경로는 저장되지 않으므로 입력받음
    st.divider()
    project_root_path = st.text_input(
        "파일 트리 경로",
        help="현재 프로젝트의 실제 폴더 경로를 입력하면 파일 구조를 시각화합니다.",
    )

    user_id = st.text_input("개발자 ID", value="Dev User")

    if project_root_path and os.path.isdir(project_root_path):
        with st.expander("파일 구조 보기"):
            st.code(generate_file_tree(project_root_path), language="text")


# RAG 파이프라인 로드
@st.cache_resource
def load_rag_pipeline(prj_name):
    if not prj_name:
        return None, "프로젝트 이름을 입력하세요."

    db_path = os.path.join(BASE_DB_PATH, prj_name)
    if not os.path.exists(db_path):
        return None, f"'{prj_name}' DB가 없습니다."

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    try:
        vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
        llm = Ollama(model=OLLAMA_MODEL_NAME, base_url=OLLAMA_BASE_URL)
        return retriever, llm
    except Exception as e:
        return None, str(e)


# 피드백 로깅
def log_feedback(project, user, question, answer, rating, docs):
    file_exists = os.path.isfile(FEEDBACK_FILE)
    with open(FEEDBACK_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "Time",
                    "Project",
                    "User",
                    "Question",
                    "Answer",
                    "Rating",
                    "Context_Files",
                ]
            )

        sources = [d.metadata.get("source") for d in docs]
        writer.writerow(
            [datetime.now(), project, user, question, answer, rating, str(sources)]
        )


# 프롬프트 (협업 및 설명 중심)
PROMPT_TEMPLATE = """
당신은 이 프로젝트의 모든 기술 스택(Full-stack)을 이해하고 있는 **수석 테크 리드(Tech Lead)**입니다.
사용자는 특정 분야에 익숙하지 않을 수 있습니다. (예: 백엔드 개발자가 DB를 묻거나, 앱 개발자가 서버를 물을 수 있음)
친절하고 상세하게, 전체 구조 관점에서 답변하세요.

[프로젝트 구조도]:
{file_tree}

[참고 코드 맥락]:
{context}

[사용자 질문]: {question}

[답변 가이드]:
1. **연결성 강조:** 질문한 코드가 프로젝트의 다른 부분(DB, API, UI 등)과 어떻게 연결되는지 설명하세요.
2. **위치 안내:** 코드를 수정하거나 추가해야 한다면, [프로젝트 구조도]를 보고 정확한 파일 위치를 제안하세요.
3. **상세 설명:** 사용자가 해당 언어를 잘 모른다고 가정하고, 로직을 명확하게 설명하세요.
4. **한국어 필수:** 모든 설명은 자연스러운 한국어로 작성하세요.

[답변]:
"""


def format_docs(docs):
    return "\n\n".join(
        [
            f"[파일: {d.metadata.get('source')}]\n```\n{d.page_content}\n```"
            for d in docs
        ]
    )


# 메인 실행 로직
retriever, llm = None, None
is_ready = False
system_msg = ""
current_tree = ""

# 파이프라인 로드 시도
if project_name:
    result = load_rag_pipeline(project_name)
    if isinstance(result, tuple):
        retriever, llm = result
        is_ready = True
        # 트리 생성
        if project_root_path:
            current_tree = generate_file_tree(project_root_path)
    else:
        # 로드 실패 시 메시지만 저장하고 중단하지 않음
        system_msg = result
else:
    system_msg = "프로젝트를 선택하거나 새로 학습해주세요."

# 3. 채팅 UI 표시
if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 학습이 안되어 있을 경우 경고 메세지를 채팅창 상단에 토스트나 경고로 살짝 보여줌
if not is_ready and system_msg:
    st.info(f"{system_msg}")

# 4. 입력창 및 답변 로직
if prompt := st.chat_input("질문을 입력하세요."):
    # 사용자 메세지 즉시 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 답변 생성
    with st.chat_message("assistant"):
        # 준비가 안된 경우 안내 메세지 출력
        if not is_ready:
            st.error("AI가 준비되지 않았습니다. 사이드바 설정을 확인하세요.")
        else:
            custom_prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
            chain = RunnableParallel(
                {
                    "context": retriever
                    | (lambda docs: "\n".join([d.page_content for d in docs])),
                    "question": RunnablePassthrough(),
                    "file_tree": lambda x: current_tree,
                }
            ).assign(answer=custom_prompt | llm | StrOutputParser())

            try:
                with st.spinner("분석 중..."):
                    result = chain.invoke(prompt)
                    # 답변 표시
                    st.markdown(result["answer"])

                    # 근거 표시

                    with st.expander("🔍 AI가 참고한 파일 및 근거"):
                        raw_docs = retriever.invoke(prompt)
                        for doc in raw_docs:
                            st.caption(f"{doc.metadata.get('source')}")
                            st.code(doc.page_content)

                    # 채팅 기록 저장
                    st.session_state.messages.append(
                        {"role": "assistant", "content": result["answer"]}
                    )

                    # 피드백을 위한 상태 저장 (키 이름 통일)
                    st.session_state.last_interaction = {
                        "project": project_name,
                        "question": prompt,
                        "answer": result["answer"],
                        "docs": raw_docs,
                    }
                    st.rerun()

            except Exception as e:
                st.error(f"오류: {e}")


# 피드백 UI
if (
    is_ready
    and "last_interaction" in st.session_state
    and st.session_state.last_interaction
):
    st.divider()
    st.caption("📢 답변 품질 평가 (자가 개선 데이터)")
    cols = st.columns([1, 1, 6])
    last = st.session_state.last_interaction

    if cols[0].button("👍 도움됨"):
        log_feedback(
            last["project"],
            user_id,
            last["question"],
            last["answer"],
            "Good",
            last["docs"],
        )
        st.toast("피드백이 저장되었습니다!")
        del st.session_state.last_interaction
        st.rerun()

    if cols[1].button("👎 부족함"):
        log_feedback(
            last["project"],
            user_id,
            last["question"],
            last["answer"],
            "Bad",
            last["docs"],
        )
        st.toast("피드백이 저장되었습니다.")
        del st.session_state.last_interaction
        st.rerun()
