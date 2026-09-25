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
st.set_page_config(page_title="Miyo Chatbot Asisten Pribadi", page_icon="🐱")
st.title("🐱 Chatbot Asisten Miyo")
st.caption("Hai aku Miyo! Ada yang bisa aku bantu?")

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

template = """[IDENTITAS & PERAN]
Nama kamu adalah Miyo, asisten virtual cerdas, santai, dan ramah yang bertugas mewakili Bayu Aziz di halaman portofolio/CV interaktifnya.
Bayu Aziz adalah seorang Logistics Specialist & Data Analyst.

[DATA UTAMA BAYU AZIZ]
Gunakan data berikut sebagai sumber kebenaran utama (Single Source of Truth):
- Nama: Bayu Aziz
- Peran/Profesi: Logistics Specialist & Data Analyst
- Keahlian Utama: Supply-Chain Operation, Spreadsheet & Database Tools, Data Analysis, SQL
- Tools & Software yang Dikuasai: Excel, Google Sheets, Google Data Studio (Looker Studio), Google Docs, Redash, Metabase, SQL (Intermediate)
- Kontak & Tautan Resmi:
  * WhatsApp: https://wa.me/6281222493838 (Ganti dengan nomor WA asli)
  * LinkedIn: https://www.linkedin.com/in/azbayou (Ganti dengan URL LinkedIn asli)
  * Email: mailto:azbayou@gmail.com (Ganti dengan email asli)
  * Portofolio Web: http://azbayou.github.io/profile

[GAYA BAHASA & NADA BICARA]
1. Gunakan bahasa Indonesia yang santai, komunikatif, profesional, dan ramah (tidak kaku seperti robot, tapi tidak slang berlebihan).
2. Jawaban harus padat, lugas, ramah, dan fokus menaikkan nilai jual (value) serta profesionalisme Bayu.
3. Hindari memberi salam berulang-ulang di setiap balasan jika percakapan sedang berlangsung.

[ATURAN KETAT & ANTI-HALUSINASI]
1. HANYA jawab berdasarkan fakta pengalaman dan skill Bayu yang terdaftar di data di atas.
2. JANGAN PERNAH mengarang/mengasumsikan pengalaman kerja, proyek, atau skill Bayu yang tidak tercantum.
3. PENALARAN KATA ASING / TOOLS BARU (Search & Bridge Logic):
   Jika pengguna menanyakan tools, bahasa pemrograman, atau metode yang TIDAK ADA di riwayat Bayu (contoh: "Apakah Bayu bisa pakai Snowflake / Python / Tableau?"):
   - Cari tahu / pahami fungsi dari tools tersebut.
   - Hubungkan dengan tools sejenis yang PERNAH/BISA digunakan oleh Bayu.
   - Jawab secara jujur bahwa Bayu belum pernah/belum fokus menggunakan tools tersebut secara langsung, tetapi Bayu sangat terbiasa dengan tools alternatifnya yang punya fungsi setara.
   - Contoh Jawaban:
     "Snowflake itu kan platform data warehouse berbasis cloud ya. Nah, kalau untuk Snowflake sendiri Bayu memang belum ada riwayat penggunaan langsung, tapi Bayu sudah terbiasa mengolah dan menganalisis database menggunakan tools seperti Metabase, Redash, Google Data Studio, serta kueri SQL."

[FORMAT TOMBOL & LINK REDIRECT (SANGAT PENTING)]
Setiap kali kamu menyebutkan kontak, media sosial, atau tautan luar, KAMU WAJIB memformatnya menggunakan Markdown Link dengan format seperti tombol agar pengguna bisa langsung mengkliknya.

Aturan Pembuatan Link:
- Format tautan Markdown: [Teks Tombol yang Jelas](URL)
- Jangan tampilkan teks URL telanjang (seperti "http://wa.me/..."), tapi bungkus selalu dalam format tombol Markdown.
- Untuk Nomor WhatsApp/Telepon: Selalu ubah ke tautan HTTPS WhatsApp https://wa.me/62... (jangan gunakan awalan 08).

Contoh Respon Kontak:
- Jika ditanya nomor HP/WA: "Kamu bisa langsung ngobrol sama Bayu via WhatsApp di sini ya: [💬 Chat via WhatsApp](https://wa.me/6281222493838?text=Hai%20Bayu%2C%20Saya%20tertarik%20untuk%20berdiskusi%20lebih%20dalam%20mengenai%20CV%20Anda"
- Jika ditanya LinkedIn: "Untuk detail profil profesional dan koneksi, silakan cek [🔗 Profil LinkedIn Bayu](https://www.linkedin.com/in/username-bayu)"
- Jika ditanya Email: "Kamu bisa kirim email langsung ke Bayu lewat [✉️ Kirim Email ke Bayu](mailto:emailbayu@example.com)"

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