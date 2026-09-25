import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_exa import ExaSearchRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="RAG Chatbot Asisten Pribadi", page_icon="🤖")
st.title("🤖 Chatbot Asisten Pribadi Miyo")
st.caption("Tanyakan mengenai CV/Catatan Bayu, atau perkembangan seputar Data & AI!")

# 2. Ambil API Key dari Streamlit Secrets
if "GEMINI_API_KEY" in st.secrets:
    gemini_key = st.secrets["GEMINI_API_KEY"]
else:
    st.error("API Key 'GEMINI_API_KEY' belum diatur di Streamlit Secrets!")
    st.stop()

exa_key = st.secrets.get("EXA_API_KEY", None)
if not exa_key:
    st.warning("⚠️ 'EXA_API_KEY' belum diatur di Streamlit Secrets. Fitur pencarian web Exa tidak akan aktif.")

# 3. Inisialisasi Vectorstore Lokal (Di-cache)
@st.cache_resource
def load_vectorstore():
    documents = []
    
    # Load CV jika ada
    if os.path.exists("CV Bayu Aziz - 090926.pdf"):
        pdf_loader = PyPDFLoader("CV Bayu Aziz - 090926.pdf")
        documents.extend(pdf_loader.load())
        
    # Load Catatan jika ada
    if os.path.exists("QnA about Bayu Aziz.txt"):
        txt_loader = TextLoader("QnA about Bayu Aziz.txt", encoding="utf-8")
        documents.extend(txt_loader.load())
        
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    
    # Embedding Model
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=gemini_key
    )
    
    # Simpan ke ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="rag_pribadi"
    )
    return vectorstore

with st.spinner("Mempersiapkan dokumen lokal..."):
    vectorstore = load_vectorstore()

# 4. Setup Retrievers (Lokal & Exa Search)
local_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

exa_retriever = None
if exa_key:
    exa_retriever = ExaSearchRetriever(
        exa_api_key=exa_key,
        k=2,
        highlights=True
    )

def get_combined_context(query: str) -> str:
    """Mengambil informasi dari retriever lokal (dokumen) dan Exa API (web)."""
    # 1. Cari dokumen lokal
    local_docs = local_retriever.invoke(query)
    local_text = "\n".join([doc.page_content for doc in local_docs])
    
    # 2. Cari via Exa Search jika API Key tersedia
    exa_text = ""
    if exa_retriever:
        try:
            exa_docs = exa_retriever.invoke(query)
            exa_text = "\n".join([doc.page_content for doc in exa_docs])
        except Exception as e:
            exa_text = f"(Gagal mengambil data dari Exa Search: {e})"
            
    exa_display = exa_text if exa_text else "Tidak ada data pencarian web."
    combined_context = f"""--- DOKUMEN LOKAL (CV & CATATAN BAYU) ---
{local_text}

--- HASIL PENCARIAN WEB (EXA SEARCH) ---
{exa_display}"""
    
    return combined_context

# 5. Setup LLM & Prompt dengan Restriksi Domain
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    google_api_key=gemini_key,
    temperature=0.3
)

template = """Kamu adalah Miyo, asisten pribadi AI yang imut, lucu, cerdas dan ramah.
Gaya bahasa Miyo santai tapi tetap formal.

Aturan Penting Menjawab:
1. Utamakan informasi dari **DOKUMEN LOKAL** jika pertanyaan berhubungan dengan Bayu Aziz, CV, latar belakang, atau catatan pribadinya.
2. Gunakan **HASIL PENCARIAN WEB (EXA SEARCH)** HANYA jika pertanyaan berkaitan dengan topik **Data, Artificial Intelligence (AI), Machine Learning, Data Engineering, atau bidang teknologi yang relevan** dengan latar belakang di dokumen lokal.
3. **PENTING (Grounded Scope)**: Jika pertanyaan pengguna melenceng jauh dari konteks (misalnya tentang resep masakan, ramalan zodiak, gosip selebriti, olahraga, atau topik umum di luar Data/AI/Teknologi & profil Bayu Aziz), **TOLAK pertanyaan tersebut secara ramah dan imut**. Jelaskan bahwa Miyo hanya bisa membantu menjawab hal-hal seputar Bayu Aziz, Data, AI, dan teknologi terkait.
4. Jika pertanyaan relevan dengan topik Data/AI/Bayu tetapi jawabannya tidak ditemukan di dokumen lokal maupun web search, katakan secara jujur dan sopan bahwa kamu belum mengetahuinya dan minta user untuk menghubungi Bayu Aziz secara pribadi via whatsapp/email/linkedin

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": get_combined_context, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. UI Interface Chat Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan riwayat percakapan
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input Chat Pengguna
if user_input := st.chat_input("Tanyakan seputar Bayu Aziz, Data, atau AI..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Miyo sedang berpikir..."):
            response = rag_chain.invoke(user_input)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})