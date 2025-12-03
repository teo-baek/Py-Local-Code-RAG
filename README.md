# 📁 Py-Local-Code-RAG: 도메인 장벽을 넘는 로컬 AI 코딩 파트너

## 🌟 프로젝트 비전

많은 개발자가 자신의 주력 언어 외에 통신, DB, 인프라 등 다른 영역을 다룰 때 어려움을 겪습니다. Py-Local-Code-RAG는 이러한 문제를 해결하기 위해 탄생했습니다.

이 도구는 단순한 코드 검색기가 아닙니다. 프로젝트 전체의 맥락을 이해하는 AI 테크 리드로서, 개발자가 낯선 영역의 코드를 이해하고 수정하며 기능을 확장할 수 있도록 돕는 강력한 협업 파트너입니다.

## 💡 핵심 기능 및 목표

1. 도메인 장벽 해소: Python 개발자가 SQL을, Frontend 개발자가 서버 로직을 다룰 수 있도록 프로젝트 맥락 기반의 상세한 설명과 가이드를 제공합니다.

2. 프로젝트 전체 구조 인식 (Forest View): 파일 트리(File Tree)를 분석하여, 특정 기능이 어디에 위치하고 어떻게 연결되는지 '숲'을 보여줍니다.

3. 완벽한 보안 (Local Only): 모든 데이터와 추론은 로컬 머신(Ollama + ChromaDB)에서 수행되므로, 기밀 프로젝트에서도 안전하게 사용할 수 있습니다.

4. 자가 개선 (Self-Improvement): 팀원들의 피드백 데이터를 축적하여, 우리 회사/팀의 스타일에 맞는 답변을 하도록 지속적으로 발전합니다.

## 🧱 아키텍처 (Hybrid RAG)

| 구성 요소 | 기술 스택 | 역할 |
| User Interface | Streamlit | 채팅형 웹 UI, 파일 트리 시각화, 피드백 수집 |
| Brain (LLM) | Ollama (Llama 3 / CodeLlama) | 로컬 추론 및 답변 생성 |
| Knowledge (DB) | ChromaDB | 프로젝트 코드 벡터 저장소 (프로젝트별 격리) |
| Orchestrator | LangChain (Python) | RAG 파이프라인 및 프롬프트 제어 |

## 🛠️ 설치 및 실행 가이드

### 1. 전제 조건

Python 3.11 이상

Ollama 설치: Ollama.com에서 설치 후 모델 다운로드

```
ollama pull codellama:8b
# 또는 한국어 성능이 더 좋은 모델 추천
ollama pull llama3
```

### 2. 환경 설정

## 1. 가상 환경 생성
python -m venv venv
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

## 2. 필수 라이브러리 설치
pip install -r requirements.txt



## 3. 

$$Step 1$$

 프로젝트 학습시키기 (색인)

분석하고 싶은 프로젝트(회사 코드)를 AI에게 학습시킵니다.

python code_indexer.py "C:/Path/To/Your/Project"



## 4. 

$$Step 2$$

 AI 파트너와 협업하기 (실행)

웹 인터페이스를 실행하여 코딩 지원을 받습니다.

streamlit run app.py

- 브라우저가 열리면 사이드바에 **프로젝트 이름(DB명)**과 실제 파일 경로를 입력하세요.