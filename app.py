import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="RAG Chatbot Asisten Pribadi", page_icon="🤖")
st.title("🤖 Chatbot Asisten Pribadi")
st.caption("Tanyakan apa saja mengenai CV dan Catatan saya!")

# 2. Ambil API Key dari Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
else:
    st.error("API Key 'GEMINI_API_KEY' belum diatur di Streamlit Secrets!")
    st.stop()

# 3. Inisialisasi Vectorstore (Di-cache agar efisien)
@st.cache_resource
def load_vectorstore():
    documents = []
    
    # Load CV jika ada
    if os.path.exists("cv.pdf"):
        pdf_loader = PyPDFLoader("cv.pdf")
        documents.extend(pdf_loader.load())
        
    # Load Catatan jika ada
    if os.path.exists("catatan.txt"):
        txt_loader = TextLoader("catatan.txt", encoding="utf-8")
        documents.extend(txt_loader.load())
        
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Embedding Model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=gemini_key
    )
    
    # Simpan ke ChromaDB (in-memory di server)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_pribadi"
    )
    return vectorstore

# Jalankan loading vectorstore dengan indikator loading
with st.spinner("Mempersiapkan dokumen..."):
    vectorstore = load_vectorstore()

# 4. Setup RAG Chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=gemini_key,
    temperature=0.3
)

template = """Kamu adalah asisten pribadi AI milik Bayu Aziz yang cerdas dan ramah. 
Jawablah pertanyaan pengguna hanya berdasarkan konteks dokumen (CV dan Catatan) yang diberikan di bawah ini.
Jika informasi tidak tersedia di dalam dokumen, katakan secara jujur bahwa kamu tidak mengetahuinya.

Konteks:
{context}

Pertanyaan: 
{question}

Jawaban:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. UI Interface Chat Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat percakapan
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Chat Pengguna
if user_input := st.chat_input("Tanyakan sesuatu..."):
    # Simpan & tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Respon AI
    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            response = rag_chain.invoke(user_input)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})