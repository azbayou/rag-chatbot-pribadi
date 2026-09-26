# rag-chatbot-pribadi
🔗 **Live Website:** [https://bay-ragcb.streamlit.app/](https://bay-ragcb.streamlit.app/)

# 🤖 Personal RAG Assistant Chatbot

A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions interactively based on personal documents (CV & Notes). Built using **LangChain**, **Google Gemini**, and **ChromaDB**, and deployed as a web application via **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![LangChain](https://img.shields.io/badge/Framework-LangChain-green)
![Gemini](https://img.shields.io/badge/Model-Gemini_Flash-Lite-orange?logo=google)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit)

---

## 📌 Alur Arsitektur (System Architecture)
```text
.
├──  [ CV (.pdf) & Catatan (.txt) ]
├──  [ Chunking (RecursiveCharacterTextSplitter) ]
├──  [ Vector Embedding (models/gemini-embedding-2) ]
├──  [ Vector Store (ChromaDB) ] ◄─── (User Query via Streamlit UI)
├──  [ Relevant Context Retrieval (k=3) ]
└──  [ Gemini 1.5 Flash LLM ] ───► [ Jawaban Contextual & Akurat ]
```
---

## 🛠️ Tech Stack

* **Language:** Python
* **LLM Engine:** Google Gemini 1.5 Flash (`models/gemini-1.5-flash`)
* **Embedding Model:** Google Gemini Embedding (`models/gemini-embedding-2`)
* **Orchestration:** LangChain (`langchain-google-genai`, `langchain-community`)
* **Vector Store:** ChromaDB
* **Web Framework:** Streamlit
* **Document Loaders:** PyPDFLoader, TextLoader

---

## 📁 Struktur Repository

```text
.
├── app.py           # Script utama aplikasi Streamlit
├── requirements.txt # Dependencies Python
├── cv.pdf           # Document CV pribadi
├── catatan.txt      # Catatan tambahan / portfolio notes
└── README.md        # Dokumentasi proyek
```

---

🚀 Jalankan Secara Lokal (Local Setup)
1. Clone repository ini:
```python
git clone [https://github.com/azbayou/rag-chatbot-pribadi.git](https://github.com/azbayou/rag-chatbot-pribadi.git)
cd rag-chatbot-pribadi
```

2. Buat & aktifkan virtual environment (opsional tapi disarankan):
```python
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau: venv\Scripts\activate  # Windows
```

3. Install dependencies:
```python
pip install -r requirements.txt
```

4. Atur API Key Gemini:
Buat folder .streamlit di root folder, lalu buat file secrets.toml
```python
GEMINI_API_KEY = "API_KEY_GEMINI_ANDA"
```

5. Jalankan aplikasi Streamlit:
```python
streamlit run app.py
```

---

🌐 Deployment (Streamlit Cloud & Integration)
Aplikasi ini di-deploy di Streamlit Community Cloud dan di-embed ke dalam Website Portfolio GitHub Pages menggunakan iframe widget dengan script berikut:
```HTML
<iframe 
  src="https://<YOUR-STREAMLIT-APP-URL>.streamlit.app/?embed=true" 
  width="100%" 
  height="100%" 
  style="border:none;">
</iframe>
```

📝 Lisensi
Proyek ini dibuat untuk keperluan portfolio pribadi dan berlisensi MIT License.
