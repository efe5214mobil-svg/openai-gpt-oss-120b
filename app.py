import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings # Güncellendi
from dotenv import load_dotenv
import os
import pandas as pd

# .env yükle
load_dotenv()

# API anahtarı kontrolü
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
if not api_key:
    st.error("Lütfen GROQ_API_KEY anahtarını ayarlayın!")
    st.stop()

client = Groq(api_key=api_key)

# Başlık ve Bilgi
st.set_page_config(page_title="MEB Mevzuat Analisti", layout="wide")
st.title("🏛️ MEB Yönetmelik Uzmanı (GPT-OSS 120B)")
st.markdown("---")

# Vektör Veritabanı Yükleme
@st.cache_resource
def load_vector_db():
    # Python 3.10-3.12 arası en kararlı çalışan model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory="okul_asistani_gpt_db",
        embedding_function=embeddings
    )
    return db

vector_db = load_vector_db()

# Sohbet Hafızası
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sohbet Geçmişini Görüntüle
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sorgulama Fonksiyonu
def analiz_et(soru):
    docs = vector_db.similarity_search(soru, k=4)
    baglam = "\n\n".join([f"Madde İçeriği: {doc.page_content}" for doc in docs])

    system_prompt = """Sen MEB yönetmelikleri ve resmi mevzuat konusunda uzman bir hukuk danışmanısın. 
    Sana verilen bağlamı (yönetmelik maddelerini) dikkatlice analiz et.
    Kurallar:
    1. Sadece bağlamdaki bilgilere sadık kal.
    2. Cevaplarını resmi, net ve maddeler halinde ver.
    3. Eğer soruda geçen durum yönetmelikte açıkça belirtilmemişse, en yakın maddeyi yorumla ama bunun bir yorum olduğunu belirt.
    4. Yanıtını 'MEB Ortaöğretim Kurumları Yönetmeliği uyarınca...' gibi resmi bir girişle başlat."""

    # 3. Model Çağrısı
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"}
            ],
            model="openai/gpt-oss-120b", 
            temperature=0.1, 
            max_tokens=1000
        )
        
        cevap = response.choices[0].message.content
        return cevap, docs
    except Exception as e:
        return f"Model hatası: {str(e)}", []

# Kullanıcı Girişi
if prompt := st.chat_input("Yönetmelik hakkında detaylı sorunuzu yazın..."):
    # Mesajı ekrana bas ve hafızaya al
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mevzuat taranıyor ve analiz ediliyor..."):
            cevap, kaynaklar = analiz_et(prompt)
            st.markdown(cevap)
            
            # Kaynakları gösteren bir expander
            with st.expander("İlgili Yönetmelik Maddeleri (Referans)"):
                for i, doc in enumerate(kaynaklar):
                    st.write(f"**Referans {i+1}:** {doc.page_content[:300]}...")
            
            st.session_state.messages.append({"role": "assistant", "content": cevap})
