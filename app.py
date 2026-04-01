import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
import os
import re
import pandas as pd

# 🔐 .env yükle
load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 🎨 Sayfa Konfigürasyonu ve Modern Stil
st.set_page_config(page_title="MEB Mevzuat Analisti", page_icon="🏛️", layout="wide")

# Modern CSS Tasarımı (Karanlık ve Aydınlık Mod Uyumlu)
st.markdown("""
    <style>
    /* Ana Arka Plan ve Yazı Tipi */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Yan Panel (Sidebar) Tasarımı */
    [data-testid="stSidebar"] {
        background-color: rgba(20, 25, 35, 0.05);
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    /* Hızlı Soru Butonları */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        background-color: transparent;
        transition: all 0.3s ease;
        text-align: left;
        padding: 10px 15px;
    }
    
    .stButton>button:hover {
        border-color: #ff4b4b;
        color: #ff4b4b;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    /* Sohbet Balonları */
    [data-testid="stChatMessage"] {
        border-radius: 20px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
    
    /* Tablo ve Expander Düzeni */
    .stExpander {
        border-radius: 15px !important;
        border: 1px solid rgba(128, 128, 128, 0.2) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 🧠 VECTOR DB Yükleme
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 🤖 SIKÇA SORULAN SORULAR (Sidebar)
def hızlı_sorular():
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/b/b5/Milli_E%C4%9Fitim_Bakanl%C4%B1%C4%9F%C4%B1_Logo.svg", width=100)
        st.title("Hızlı Erişim")
        st.markdown("---")
        sorular = [
            "🔖 Özürsüz devamsızlık sınırı",
            "🎓 Sınıf geçme puanı nedir?",
            "📚 Kaç dersten kalınca sınıf tekrarı olur?",
            "🚭 Sigara içmenin disiplin cezası",
            "🎖️ Takdir ve Teşekkür şartları",
            "🔄 Nakil başvuru zamanları",
            "💍 Evlenen öğrencilerin durumu",
            "⏰ Ders saati süreleri"
        ]
        for s in sorular:
            if st.button(s):
                return s.replace("🔖 ", "").replace("🎓 ", "").replace("📚 ", "").replace("🚭 ", "").replace("🎖️ ", "").replace("🔄 ", "").replace("💍 ", "").replace("⏰ ", "")
    return None

# 🤖 ANALİZ VE SORGULAMA
def okul_asistani_sorgula(soru):
    docs = vector_db.similarity_search(soru, k=4)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        {
            "role": "system", 
            "content": """Sen MEB yönetmeliği konusunda uzmanlaşmış bir analiz modelisin. 
            [SORULAR.PDF VERİLERİNE GÖRE KESİN KURALLAR]:
            - Devamsızlık: Özürsüz 10 günü geçerse başarısız sayılır[cite: 1]. Toplam sınır 30 gündür (Ağır hastalık/organ naklinde 60 gün)[cite: 1].
            - Ders Süresi: Okulda 40, işletmelerde 60 dakikadır[cite: 1].
            - Kayıt: 18 yaşını bitirmemiş olmak gerekir[cite: 1]. Evli olanların kaydı yapılmaz; öğrenciyken evlenenler Açık Lise'ye aktarılır[cite: 1].
            - Başarı: Dersten geçmek için yılsonu puanı en az 50 olmalıdır[cite: 1].
            - Sınıf Geçme: 3 derse kadar sorumlu geçilir[cite: 2]. Toplam başarısızlık 6'yı geçerse sınıf tekrarı yapılır[cite: 2].
            - Belgeler: Teşekkür için 70.00-84.99, Takdir için 85.00 ve üzeri ortalama gerekir[cite: 2].
            - Disiplin: Kopya çekmek veya sigara içmek 'Kınama' cezası gerektirir[cite: 2].
            - Nakil: Aralık ve Mayıs hariç her ayın ilk iş günü başvurulabilir[cite: 2].

            [STRATEJİ]: Cevabını resmi bir dille başlat ve yukarıdaki kesin verileri önceliklendir."""
        }
    ]

    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)

    messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-120b",
            temperature=0.1,
            max_tokens=800
        )
        return completion.choices[0].message.content, docs
    except Exception as e:
        return f"Hata: {str(e)}", None

# --- ARAYÜZ AKIŞI ---
sidebar_soru = hızlı_sorular()

# Başlık Bölümü
st.markdown("### 🏛️ Milli Eğitim Bakanlığı Mevzuat Danışmanı")
st.caption("Yönetmelik maddeleri ve sorular.pdf verilerine dayalı yapay zeka analizi.")

# Chat Ekranı
chat_container = st.container()

with chat_container:
    for msg in st.session_state.conversation:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Giriş Bölümü
prompt = sidebar_soru or st.chat_input("Sorunuzu buraya yazın (Örn: Devamsızlık sınırı nedir?)")

if prompt:
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("⚖️ Mevzuat inceleniyor..."):
        cevap, kaynaklar = okul_asistani_sorgula(prompt)
        st.session_state.conversation.append({"role": "assistant", "content": cevap})
        with st.chat_message("assistant"):
            st.markdown(cevap)
            
            if kaynaklar:
                with st.expander("📄 Dayanak Maddeler ve Tablo Verisi"):
                    # PDF Verisi Özeti
                    st.info("Bu cevap sorular.pdf içerisindeki resmi tablo verileriyle desteklenmiştir.")
                    for doc in kaynaklar:
                        st.caption(doc.page_content[:400] + "...")
                        st.divider()
