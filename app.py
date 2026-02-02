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

def build_vectorstore(api_key):
    """
    Build new ChromaDB from data folder.
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
    
    vectorstore = Chroma.from_texts(
        texts=all_chunks,
        embedding=embeddings,
        collection_name="antigravity_docs",
        persist_directory=persist_dir
    )
    status_text.success("DB successfully built!")
    return vectorstore

def ask_gemini(vectorstore, question, api_key, chat_history):
    # 1. Morphological Analysis of the Question (Using Kiwi as replacement for Okt)
    processed_question = tokenize_kiwi(question)
    
    # 2. Retrieve Top 5 Documents (Increased from 3 to improve recall)
    docs = vectorstore.similarity_search(processed_question, k=7)
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

    return stream_func(), docs


# 1. Page Config
st.set_page_config(
    page_title="실험용 챗봇",
    page_icon="🤖",
    layout="wide"
)

# 2. Main Title
st.title("🚀 RAG Chatbot")

# 3. Sidebar
with st.sidebar:
    st.header("⚙️ 설정")
    
    # Define data directory first to avoid NameError
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Load API Key from env if available
    env_api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key = st.text_input("Google API Key", value=env_api_key, type="password")
    
    if st.button("💾 API Key 저장 (로컬 .env)"):
        if api_key:
            with open(".env", "w") as f:
                f.write(f"GOOGLE_API_KEY={api_key}")
            st.success("API Key가 .env 파일에 저장되었습니다!")
            # Reload to apply immediately
            load_dotenv(override=True)
            st.rerun()
        else:
            st.warning("API Key를 입력해주세요.")
    
    # File Upload Section
    st.markdown("---")
    st.header("📂 데이터 업로드")
    uploaded_file = st.file_uploader("학습시킬 파일을 올려주세요 (.txt, .pdf)", type=["txt", "pdf"])
    if uploaded_file:
        file_path = os.path.join(data_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"'{uploaded_file.name}' 저장 완료! 아래 [DB 구축하기]를 눌러 반영해주세요.")
    
    st.markdown("---")
    st.header("🗄️ 데이터 베이스 상태")
    
    if api_key:
        vectorstore = get_vectorstore(api_key)
        
        if vectorstore:
            st.success("✅ DB 로드 완료 (antigravity_docs)")
            
            # DB Inspection Feature
            with st.expander("🔍 DB 내부 데이터 확인"):
                try:
                    collection_data = vectorstore.get(limit=3) 
                    
                    if collection_data and 'documents' in collection_data:
                        docs = collection_data['documents']
                        ids = collection_data['ids']
                        
                        total_count = vectorstore._collection.count()
                        st.write(f"📊 **총 청크 수:** {total_count}개")
                        
                        st.write("🧩 **샘플 데이터 (최대 3개):**")
                        for i, doc in enumerate(docs):
                            st.caption(f"**Chunk {ids[i]}:**")
                            st.text(doc[:100] + "...") 
                    else:
                        st.write("데이터를 가져올 수 없습니다.")
                except Exception as e:
                    st.error(f"데이터 확인 중 오류: {e}")
            
            # Rebuild DB button for updating data
            if st.button("🔄 DB 갱신하기"):
                 with st.spinner("데이터 처리 중..."):
                      vectorstore = build_vectorstore(api_key)
                      get_vectorstore.clear()
                      st.rerun()

        else:
            st.warning("⚠️ DB가 없습니다.")
            if st.button("DB 구축하기"):
                with st.spinner("데이터 처리 중..."):
                     vectorstore = build_vectorstore(api_key)
                     get_vectorstore.clear()
                     st.rerun()
    else:
        st.info("API Key를 입력하면 DB 상태를 확인할 수 있습니다.")


        files = os.listdir(data_dir)
        if files:
            st.markdown("---")
            st.write(f"📂 **소스 파일 ({len(files)}개):**")
            for f in files:
                st.caption(f"- {f}")
        else:
            st.warning("⚠️ data 폴더가 비어있습니다.")

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("궁금한 내용을 물어보세요..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if not api_key:
            response = "⚠️ 사이드바에서 Google API Key를 입력해주세요."
            message_placeholder.warning(response)
        elif not vectorstore:
             response = "⚠️ DB가 로드되지 않았습니다. DB를 먼저 구축해주세요."
             message_placeholder.warning(response)
        else:
            # Change spinner context to allow streaming write
            with st.spinner("답변 생성 중..."):
                try:
                    stream, docs = ask_gemini(vectorstore, prompt, api_key, st.session_state.messages)
                    
                    # Use st.write_stream to simulate typing effect
                    # write_stream returns the full concatenated string
                    response_text = message_placeholder.write_stream(stream)
                    
                    # Optional: Show sources in expander
                    with st.expander("📚 참조 문서"):
                        for i, doc in enumerate(docs):
                            st.caption(f"**Reference {i+1}**")
                            st.text(doc.page_content)
                            
                    response = response_text # For history
                except Exception as e:
                    response = f"❌ 오류 발생: {e}"
                    message_placeholder.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
