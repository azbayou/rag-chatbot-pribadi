import os
import streamlit as st
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_exa import ExaSearchRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Chat Miyo", page_icon="🐱")

# --- KUSTOMISASI CSS UI WHATSAPP ---
st.markdown("""
<style>
    /* Styling latar belakang area chat bergaya WhatsApp */
    .stApp {
        # background-color: #efeae2;
    }

    /* Container Dasar Pesan Streamlit */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        margin-bottom: 15px;
    }

    /* ----------------------------------------------------
       PESAN USER (Rata Kanan - Gelembung Hijau Muda)
       ---------------------------------------------------- */
    /* Balik urutan elemen agar avatar ada di kanan */
    div[data-testid="stChatMessage"]:has(.user-msg-hook) {
        flex-direction: row-reverse !important;
    }
    
    /* Styling gelembung chat (target div terakhir yaitu isi chat) */
    div[data-testid="stChatMessage"]:has(.user-msg-hook) > div:last-child {
        background-color: #D3F1FD !important; /* Hijau cerah ala WA */
        color: #111b21 !important;
        padding: 10px 14px;
        border-radius: 12px;
        border-top-right-radius: 0px; /* Ekor gelembung di kanan atas */
        margin-right: 8px; /* Jarak antara teks dan avatar */
        margin-left: 20%; /* Mencegah bubble terlalu lebar */
        box-shadow: 0 1px 1px rgba(0,0,0,0.15);
    }
    
    div[data-testid="stChatMessage"]:has(.user-msg-hook) p,
    div[data-testid="stChatMessage"]:has(.user-msg-hook) li {
        color: #111b21 !important;
    }

    /* ----------------------------------------------------
       PESAN MIYO (Rata Kiri - Gelembung Putih)
       ---------------------------------------------------- */
    /* Urutan normal (avatar di kiri) */
    div[data-testid="stChatMessage"]:has(.bot-msg-hook) {
        flex-direction: row !important;
    }

    /* Styling gelembung chat Miyo */
    div[data-testid="stChatMessage"]:has(.bot-msg-hook) > div:last-child {
        background-color: #E7E7E7 !important;
        color: #111b21 !important;
        padding: 10px 14px;
        border-radius: 12px;
        border-top-left-radius: 0px; /* Ekor gelembung di kiri atas */
        margin-left: 8px;
        margin-right: 20%;
        box-shadow: 0 1px 1px rgba(0,0,0,0.15);
    }
    
    div[data-testid="stChatMessage"]:has(.bot-msg-hook) p,
    div[data-testid="stChatMessage"]:has(.bot-msg-hook) li {
        color: #111b21 !important;
    }

    /* ----------------------------------------------------
       STYLING TOMBOL LINK DALAM CHAT
       ---------------------------------------------------- */
    div[data-testid="stChatMessage"] a {
        display: inline-block;
        padding: 6px 16px;
        margin: 4px 2px;
        background-color: #008CA8; 
        color: #ffffff !important;
        text-decoration: none !important;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
        transition: all 0.2s ease-in-out;
        border: 1px solid rgba(0,0,0,0.05);
    }

    div[data-testid="stChatMessage"] a:hover {
        background-color: #128C7E;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }

    /* ----------------------------------------------------
       DARK MODE (Sesuai tema WA Dark)
       ---------------------------------------------------- */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #2C6885 !important;
        }
        
        /* Chat User Mode Gelap */
        div[data-testid="stChatMessage"]:has(.user-msg-hook) > div:last-child {
            background-color: #005c4b !important;
        }
        div[data-testid="stChatMessage"]:has(.user-msg-hook) p,
        div[data-testid="stChatMessage"]:has(.user-msg-hook) li {
            color: #e9edef !important;
        }
        
        /* Chat Miyo Mode Gelap */
        div[data-testid="stChatMessage"]:has(.bot-msg-hook) > div:last-child {
            background-color: #202c33 !important;
        }
        div[data-testid="stChatMessage"]:has(.bot-msg-hook) p,
        div[data-testid="stChatMessage"]:has(.bot-msg-hook) li {
            color: #e9edef !important;
        }
        
        /* Tombol Link Mode Gelap */
        div[data-testid="stChatMessage"] a {
            background-color: #00a884;
            color: #111b21 !important;
        }
        div[data-testid="stChatMessage"] a:hover {
            background-color: #008f6f;
        }
    }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 8])
with col1:
    try:
        title_logo = Image.open("images/2_20260925_193506_0001.png")
        st.image(title_logo, width=80)
    except FileNotFoundError:
        st.markdown("<h1>🐱</h1>", unsafe_allow_html=True) # Fallback jika gambar tidak ditemukan

with col2:
    st.subheader("Asisten Miyo")
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
    if os.path.exists("knowledges/CV Bayu Aziz - 090926.pdf"):
        pdf_loader = PyPDFLoader("knowledges/CV Bayu Aziz - 090926.pdf")
        documents.extend(pdf_loader.load())
        
    # Load Catatan jika ada
    if os.path.exists("knowledges/QnA about Bayu Aziz.txt"):
        txt_loader = TextLoader("knowledges/QnA about Bayu Aziz.txt", encoding="utf-8")
        documents.extend(txt_loader.load())
        
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
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
    temperature=0.2
)

template = """[IDENTITAS & PERAN]
Kamu adalah Miyo, asisten virtual berbasis AI yang cerdas, ramah, hangat, dan sangat menyenangkan diajak berdiskusi (seperti Gemini).
Tugas utamamu adalah mewakili Bayu Aziz di halaman portofolio/CV interaktifnya. Kamu hadir untuk memberikan informasi sejelas, seseru, dan senyaman mungkin tentang latar belakang, keahlian, serta pengalaman profesional Bayu.

[DATA UTAMA BAYU AZIZ (SINGLE SOURCE OF TRUTH)]
- Nama Lengkap: Bayu Aziz
- Peran/Profesi: Logistics Specialist & Data Analyst
- Keahlian Utama: Supply-Chain Operations, Spreadsheet & Database Tools, Data Analysis, SQL
- Tools & Software yang Dikuasai: Excel, Google Sheets, Google Data Studio (Looker Studio), Google Docs, Redash, Metabase, SQL (Intermediate)
- Kontak & Tautan Resmi:
  * WhatsApp: https://wa.me/6281222493838
  * LinkedIn: https://www.linkedin.com/in/azbayou
  * Email: mailto:azbayou@gmail.com
  * Portofolio Web: http://azbayou.github.io/profile

[GAYA BAHASA & NADA BICARA (ALA GEMINI - CERDAS & FRIENDLY)]
1. Warm, Luwes & Natural: Gunakan bahasa Indonesia yang santai, ramah, responsif, dan manusiawi (seperti teman diskusi yang pintar). Hindari bahasa kaku seperti robot/sistem pendaftaran, tapi tetap jaga batas profesionalisme.
2. Rapi & Mudah Dibaca (Scannable): Jika menjelaskan poin atau analisis, gunakan **teks tebal**, bullet points, atau penomoran yang rapi agar pembaca nyaman menyerap informasi dengan cepat.
3. Interactive & Engaging: Jika relevan, di akhir penjelasan kamu bisa memberikan pertanyaan penutup yang ramah atau opsi topik lanjutan yang membantu pengguna menjelajah profil Bayu lebih jauh.
4. Tanpa Redundansi Salam: Jangan menyapa pengguna berulang-ulang ("Halo!", "Hai!") di setiap balasan jika percakapan sudah berlangsung.
5. Selalu akhiri chat Miyo dengan -Miyo~ dengan nada imut dan lucu.

[ATURAN KETAT & ANTI-HALUSINASI]
1. Faktual: Selalu berpatokan pada data resmi Bayu Aziz di atas serta konteks dokumen yang diberikan.
2. Kejujuran: JANGAN PERNAH mengarang pengalaman, proyek, atau keahlian Bayu yang tidak tercantum.
3. Search & Bridge Logic (Penalaran Tools/Metode Baru):
   Jika pengguna menanyakan tools, bahasa pemrograman, atau metode yang belum ada di riwayat Bayu (misal: Python, Tableau, Snowflake):
   - Jelaskan fungsi tools/metode tersebut secara ringkas dan cerdas.
   - Hubungkan secara jujur dengan tools sejenis yang sudah sangat dikuasai oleh Bayu.
   - Fokusnya adalah skill dan kemampuan Bayu Aziz, jangan bandingkan dengan orang lain!.
   - Contoh Respon: "Snowflake itu platform cloud data warehouse yang canggih banget untuk olah data skala besar. Nah, kalau untuk Snowflake sendiri Bayu memang belum ada riwayat penggunaan langsung, tapi Bayu sudah terbiasa mengolah database menggunakan kueri SQL, Metabase, Redash, dan Google Data Studio. Jadi secara logika alur datanya, Bayu bisa cepat menyesuaikan!"

[FORMAT TOMBOL & LINK REDIRECT (MANDATORI)]
Setiap kali kamu menyebutkan kontak, media sosial, atau tautan luar, KAMU WAJIB memformatnya menggunakan Markdown Link bergaya tombol interaktif.

Aturan Link:
- Format tautan Markdown: [Teks Tombol yang Jelas](URL)
- Jangan pernah tampilkan URL telanjang.
- Untuk Nomor WhatsApp/Telepon: Gunakan link HTTPS WhatsApp https://wa.me/6281222493838...

Contoh Respon Kontak:
- WhatsApp: "Kamu bisa ngobrol langsung sama Bayu lewat WhatsApp di sini ya: [💬 Chat via WhatsApp](https://wa.me/6281222493838?text=Hai%20Bayu%2C%20saya%20tertarik%20berdiskusi%20mengenai%20portofolio%20Anda)"
- LinkedIn: "Untuk terkoneksi secara profesional, yuk cek [🔗 Profil LinkedIn Bayu](https://www.linkedin.com/in/azbayou)"
- Email: "Atau kamu bisa kirim pesan langsung via email ke [✉️ Email Bayu Aziz](mailto:azbayou@gmail.com)"
- Portofolio Web: "Intip juga hasil karya & portofolionya di [🌐 Web Portofolio Bayu](http://azbayou.github.io/profile)"

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

miyo_avatar = "images/1_20260925_193506_0000.png"
user_avatar = "images/ava_user.jpeg"

# Tampilkan riwayat percakapan dengan Hook HTML
for message in st.session_state.messages:
    avatar = miyo_avatar if message["role"] == "assistant" else user_avatar
    with st.chat_message(message["role"], avatar=avatar):
        # Menyisipkan class span tak kasatmata agar CSS bisa mendeteksi role
        hook = "<span class='bot-msg-hook'></span>" if message["role"] == "assistant" else "<span class='user-msg-hook'></span>"
        st.markdown(f"{hook}\n\n{message['content']}", unsafe_allow_html=True)

# Input Chat Pengguna
if user_input := st.chat_input("Tanyakan seputar Bayu Aziz, Data, atau AI..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(f"<span class='user-msg-hook'></span>\n\n{user_input}", unsafe_allow_html=True)

    with st.chat_message("assistant", avatar=miyo_avatar):
        with st.spinner("Miyo sedang berpikir..."):
            response = rag_chain.invoke(user_input)
            st.markdown(f"<span class='bot-msg-hook'></span>\n\n{response}", unsafe_allow_html=True)
            
    st.session_state.messages.append({"role": "assistant", "content": response})