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
    # 🔍 daha iyi arama sorgusu
    arama_sorgusu = f"{soru} meb yönetmelik maddesi devamsızlık şartları"

    # 🔥 gelişmiş arama
    docs = vector_db.similarity_search_with_score(arama_sorgusu, k=5)

    # en iyi sonuçları seç
    docs = sorted(docs, key=lambda x: x[1])[:3]
    docs = [doc[0] for doc in docs]

    if not docs:
        return "Veri bulunamadı."

    # 🔥 bağlam oluştur (kısaltılmış)
    baglam = "\n\n".join([doc.page_content[:500] for doc in docs])

    # 🤖 AI çağrısı
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
Sen MEB yönetmeliği uzmanısın.

Kurallar:
- Sadece verilen bağlama göre cevap ver
- Eğer madde varsa belirt
- En yakın bilgiyi kullan
- "bulunamadı" deme
"""
            },
            {
                "role": "user",
                "content": f"{baglam}\n\nSoru: {soru}"
            }
        ],
        model="gemma2-9b-it",
        temperature=0,
        max_tokens=500
    )

    cevap = chat_completion.choices[0].message.content

    # 📚 kaynak ekleme
    kaynaklar = [doc.page_content[:200] for doc in docs]

    return cevap + "\n\n📚 Kaynak:\n- " + "\n- ".join(kaynaklar)
# ✍️ INPUT
soru = st.text_input("Sorunuzu yazın:")

if soru:
    with st.spinner("Yanıt hazırlanıyor..."):
        cevap = okul_asistani_sorgula(soru)
        st.write(cevap)
