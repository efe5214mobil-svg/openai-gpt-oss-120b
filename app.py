import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# 🔐 .env yükle
load_dotenv()

# 🔑 API KEY
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=api_key)

# 🎯 Başlık
st.title("MEB Yönetmelik Asistanı")

# 🧠 VECTOR DB
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    db = Chroma(
        persist_directory="okul_asistani_gpt_db",
        embedding_function=embeddings
    )
    return db

vector_db = load_vector_db()

# 🤖 SORGULAMA
def okul_asistani_sorgula(soru):
    docs = vector_db.similarity_search(soru, k=2)  # 🔥 düşürdük

    if not docs:
        return "Veritabanında uygun bilgi bulunamadı."

    # 🔥 çok kısalttık (en kritik fix)
    baglam = "\n\n".join([doc.page_content[:300] for doc in docs])

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Sen MEB yönetmeliği uzmanısın. Sadece verilen bağlama göre kısa ve net cevap ver."
                },
                {
                    "role": "user",
                    "content": f"{baglam}\n\nSoru: {soru}"
                }
            ],
            model="gemma2-9b-it",  # ✅ EN STABİL
            temperature=0,
            max_tokens=500
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"Hata oluştu: {str(e)}"

# ✍️ INPUT
soru = st.text_input("Sorunuzu yazın:")

if soru:
    with st.spinner("Yanıt hazırlanıyor..."):
        cevap = okul_asistani_sorgula(soru)
        st.write(cevap)
