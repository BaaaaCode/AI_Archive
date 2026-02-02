import streamlit as st
import os
from dotenv import load_dotenv

# Fix for UnicodeEncodeError on Windows with Gemini/gRPC
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LANG"] = "C.UTF-8"

# Load environment variables
load_dotenv()

from kiwipiepy import Kiwi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pypdf import PdfReader

@st.cache_resource
def get_kiwi():
    return Kiwi()

def tokenize_kiwi(text):
    """
    Kiwi 형태소 분석기로 텍스트를 공백으로 연결된 문자열로 변환
    """
    kiwi = get_kiwi()
    results = kiwi.analyze(text)
    tokens = []
    for result in results:
        for token in result[0]:
            tokens.append(token.form)
    return ' '.join(tokens)

def preprocess_and_chunk(text):
    """
    1. Kiwi 형태소 분석
    2. RecursiveCharacterTextSplitter로 청킹 (size=300, overlap=50)
    """
    processed_text = tokenize_kiwi(text)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(processed_text)

@st.cache_resource
def get_vectorstore(api_key):
    """
    Load existing ChromaDB if available.
    """
    persist_dir = "./antigravity_db"
    
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_key
        )
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="antigravity_docs"
        )
        return vectorstore
    return None

import chromadb

def build_vectorstore(api_key):
    """
    Build new ChromaDB from data folder.
    Refreshes the DB by deleting the collection via client (safer on Windows).
    """
    persist_dir = "./antigravity_db"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    
    if not os.path.exists(data_dir):
        st.error("Data directory not found.")
        return None

    # Gather all texts
    all_chunks = []
    files = os.listdir(data_dir)
    status_text = st.empty()
    
    progress_bar = st.progress(0)
    for i, file in enumerate(files):
        status_text.text(f"Processing {file}...")
        file_path = os.path.join(data_dir, file)
        
        content = ""
        try:
            if file.lower().endswith(".pdf"):
                pdf = PdfReader(file_path)
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        content += text + "\n"
            else: # Default to text file
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            
            if content:
                chunks = preprocess_and_chunk(content)
                all_chunks.extend(chunks)
                
        except Exception as e:
            st.error(f"Error reading {file}: {e}")
            continue
            
        progress_bar.progress((i + 1) / len(files))
    
    status_text.text(f"Generating Embeddings for {len(all_chunks)} chunks...")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=api_key
    )
    
    # Use PersistentClient to handle collection reset
    client = chromadb.PersistentClient(path=persist_dir)
    
    # Try to delete existing collection to avoid duplicates
    try:
        client.delete_collection("antigravity_docs")
    except ValueError:
        pass # Collection might not exist yet
    
    vectorstore = Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        collection_name="antigravity_docs",
        client=client # Pass client directly
    )
    status_text.success("DB successfully built!")
    return vectorstore

def ask_gemini(vectorstore, question, api_key, chat_history):
    # 1. Morphological Analysis of the Question (Using Kiwi as replacement for Okt)
    processed_question = tokenize_kiwi(question)
    
    # 2. Retrieve Top 5 Documents (Increased from 3 to improve recall)
    # Return (doc, score) tuples
    docs_with_scores = vectorstore.similarity_search_with_score(processed_question, k=5)
    
    # Extract docs just for context building
    docs = [doc for doc, score in docs_with_scores]
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Format chat history for context
    # Use last 3 turns to keep prompt size manageable
    recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history
    formatted_history = ""
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n"

    # 3. System Prompt & Generation
    system_prompt = f"""
    너는 법률 전문가야. 아래의 [Context]와 [Chat History]를 바탕으로 질문에 대해 답변해줘.
    만약 [Context]에 없는 내용이라면 "제공된 문서에서 관련 내용을 찾을 수 없습니다."라고 답해줘.
    오직 제공된 맥락 정보만 참고해서 답변해야 해.
    
    [Context]
    {context}
    
    [Chat History]
    {formatted_history}
    
    [Question]
    {question}
    """
    
    # Using gemini-2.5-flash-lite as suggested by user
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0
    )
    
    # Create a generator for streaming
    def stream_func():
        for chunk in llm.stream(system_prompt):
            yield chunk.content

    return stream_func(), docs_with_scores

def summarize_references(docs, api_key):
    """
    References are in tokenized format (e.g. '제 4 조 ...').
    Use AI to reconstruct natural Korean and summarize.
    """
    content = "\n\n".join([doc.page_content for doc in docs])
    
    system_prompt = f"""
    아래 텍스트는 형태소 분석기에 의해 토큰화되어 띄어쓰기가 어색한 한국어 문서들입니다.
    이 내용을 읽고, 자연스러운 한국어 문장으로 다듬어서 핵심 내용을 요약해주세요.
    법률 전문가처럼 명확하고 간결하게 정리해주세요.
    
    [Raw Refereces]
    {content}
    """
    
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0
    )
    
    response = llm.invoke(system_prompt)
    return response.content

# 1. Page Config
st.set_page_config(
    page_title="실험용 챗봇",
    page_icon="🤖",
    layout="wide"
)

def apply_custom_styles():
    st.markdown("""
    <style>
        /* Import Pretendard Font */
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
        
        html, body, [class*="css"] {
            font-family: 'Pretendard', sans-serif;
        }
        
        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #e9ecef;
        }
        
        /* Header Styling */
        h1 {
            background: linear-gradient(to right, #1e3c72, #2a5298);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800 !important;
        }
        
        h2, h3 {
            color: #2c3e50;
            font-weight: 700 !important;
        }
        
        /* Button Styling */
        .stButton > button {
            background: linear-gradient(45deg, #2a5298, #1e3c72);
            color: white !important;
            border: none;
            border-radius: 10px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
            opacity: 0.9;
        }
        
        /* Chat Input Styling */
        .stChatInput {
            border-radius: 15px !important;
        }
        
        /* Message Styling (Optional tweaks) */
        [data-testid="stChatMessage"] {
            padding: 1rem;
            border-radius: 15px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Expander Styling */
        .streamlit-expanderHeader {
            font-weight: 600;
            color: #1e3c72;
        }
    </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# --- Navigation & Page Management ---

def page_chat(api_key, vectorstore):
    st.title("⚖️ AI Chat")
    st.caption("🚀 RAG 기반 법률 상담 챗봇")

    if not api_key:
        st.error("⚠️ API Key가 설정되지 않았습니다. [관리자 페이지]에서 키를 설정해주세요.")
        return
    
    if not vectorstore:
        st.error("⚠️ 학습된 데이터베이스가 없습니다. [관리자 페이지]에서 문서를 업로드하고 DB를 생성해주세요.")
        return

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("법률 관련 궁금한 점을 물어보세요..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("판례와 법령을 분석 중입니다..."):
                try:
                    # Note: ask_gemini returns (stream, docs_with_scores)
                    stream, docs_with_scores = ask_gemini(vectorstore, prompt, api_key, st.session_state.messages)
                    
                    # Streaming response
                    response_text = message_placeholder.write_stream(stream)
                    
                    # Reference Section
                    with st.expander("📚 참조 문서 (AI 요약)"):
                         if "찾을 수 없습니다" in response_text and len(response_text) < 150:
                             st.info("💡 답변을 찾을 수 없어 요약을 생략합니다. 원문 데이터를 확인하세요.")
                             for i, (doc, score) in enumerate(docs_with_scores):
                                st.caption(f"**Ref {i+1}** (유사도: {score:.4f})")
                                st.text(doc.page_content)
                         else:
                             # Extract just docs for summary
                             docs = [doc for doc, score in docs_with_scores]
                             with st.spinner("참조 문서 요약 중..."):
                                summary = summarize_references(docs, api_key)
                                st.markdown(summary)
                                st.caption("---")
                                for i, (doc, score) in enumerate(docs_with_scores):
                                    st.text(f"[Ref {i+1}] (거리: {score:.4f}) {doc.page_content[:100]}...")
                    
                    response = response_text
                except Exception as e:
                    response = f"❌ 오류 발생: {e}"
                    message_placeholder.error(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})


def page_admin(api_key, current_dir, data_dir):
    st.title("🛠️ 관리자 설정")
    
    tab1, tab2 = st.tabs(["🔐 API 및 DB 설정", "📂 문서 데이터 관리"])
    
    with tab1:
        st.subheader("Google API Key 설정")
        current_key = api_key if api_key else ""
        new_key = st.text_input("API Key 입력", value=current_key, type="password", key="admin_api_key")
        
        if st.button("💾 API Key 저장", type="primary"):
            if new_key:
                with open(".env", "w") as f:
                    f.write(f"GOOGLE_API_KEY={new_key}")
                st.success("API Key가 저장되었습니다! (새로고침 후 적용)")
                load_dotenv(override=True)
                st.rerun()
        
        st.markdown("---")
        st.subheader("데이터베이스 관리")
        
        if api_key:
            vectorstore = get_vectorstore(api_key)
            if vectorstore:
                total_count = vectorstore._collection.count()
                col1, col2 = st.columns(2)
                col1.metric("총 학습 청크", f"{total_count}개")
                col2.success("DB 상태: 정상 (antigravity_docs)")
                
                if st.button("🔄 전체 DB 재구축/갱신 (기존 데이터 삭제됨)"):
                    with st.spinner("기존 DB 삭제 및 재학습 중..."):
                        get_vectorstore.clear()
                        import gc
                        gc.collect()
                        build_vectorstore(api_key)
                        st.success("DB가 성공적으로 재구축되었습니다!")
                        st.rerun()
                
                with st.expander("🔍 데이터 샘플링"):
                    docs = vectorstore.get(limit=3)
                    st.json(docs)
            else:
                st.warning("DB가 존재하지 않습니다. 문서를 업로드하고 'DB 생성'을 진행하세요.")
                if st.button("🆕 DB 생성 시작"):
                     with st.spinner("DB 생성 중..."):
                        build_vectorstore(api_key)
                        st.success("완료!")
                        st.rerun()
        else:
            st.error("API Key가 먼저 설정되어야 합니다.")

    with tab2:
        st.subheader("학습 문서 업로드")
        uploaded_files = st.file_uploader("PDF 또는 TXT 파일 업로드", type=["pdf", "txt"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(data_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.success(f"{len(uploaded_files)}개 파일 업로드 완료! 'DB 설정' 탭에서 갱신 버튼을 눌러주세요.")
        
        st.markdown("---")
        st.subheader("현재 저장된 파일 목록")
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            if files:
                st.dataframe({"파일명": files}, use_container_width=True)
            else:
                st.info("저장된 파일이 없습니다.")

# --- Main Execution ---

# Load Env
load_dotenv()
env_api_key = os.getenv("GOOGLE_API_KEY", "")

# Sidebar Navigation
with st.sidebar:
    st.header("🤖 메뉴")
    page = st.radio("이동", ["💬 채팅하기", "🛠️ 관리자 설정"], index=0)
    
    st.markdown("---")
    st.caption("Current Info")
    if env_api_key:
        st.success("API Key: 확인됨")
    else:
        st.error("API Key: 없음")

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Routing
if page == "💬 채팅하기":
    # Need to load vectorstore for chat
    vectorstore = get_vectorstore(env_api_key) if env_api_key else None
    page_chat(env_api_key, vectorstore)
elif page == "🛠️ 관리자 설정":
    page_admin(env_api_key, current_dir, data_dir)
