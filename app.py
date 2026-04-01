import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
import os
import pandas as pd

# 🔐 API ve Çevre Değişkenleri
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 🎨 Sayfa Yapılandırması
st.set_page_config(page_title="MEB Mevzuat Asistanı", page_icon="🏛️", layout="centered")

# 🖌️ Modern CSS (Butonu Yazı Yazma Yerinin Sağ Üstüne Sabitleme)
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .main-title { font-size: 2.2rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; }
    
    /* 🚀 Sınıf Programı Butonunu Chat Input'un Sağ Üstüne Sabitleme */
    .floating-btn-container {
        position: fixed;
        bottom: 85px; /* Yazı yazma alanının hemen üstü */
        right: 5%;    /* Sağ tarafa yasla */
        z-index: 1000;
    }

    .stLinkButton a {
        background-color: #FF8C00 !important; /* Turuncu */
        color: white !important;
        border-radius: 20px !important;
        border: 2px solid white !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 700 !important;
        text-decoration: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        transition: 0.3s ease !important;
    }
    .stLinkButton a:hover {
        background-color: #E67E22 !important;
        transform: scale(1.05);
    }

    /* Rehber Kartı Stili */
    .category-box {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 15px;
        padding: 18px;
        border-top: 4px solid #FF4B4B;
        height: 100%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    .category-title { font-weight: bold; color: #FF4B4B; margin-bottom: 10px; font-size: 1.15rem; border-bottom: 1px solid rgba(255, 75, 75, 0.1); padding-bottom: 5px; }
    .category-item { font-size: 0.88rem; margin-bottom: 6px; color: #444; line-height: 1.4; }
    
    @media (prefers-color-scheme: dark) {
        .category-item { color: #CCC; }
        .category-box { background-color: rgba(255, 255, 255, 0.05); }
    }

    [data-testid="stChatMessage"] { border-radius: 20px; border: 1px solid rgba(128, 128, 128, 0.15); }
</style>
""", unsafe_allow_html=True)

# 🧠 Vektör Veritabanı
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 🤖 Sorgulama Fonksiyonu (Llama 3.3 70B)
def sorgula(soru):
    docs = vector_db.similarity_search(soru, k=3)
    baglam = "\n\n".join([doc.page_content for doc in docs])
    messages = [{"role": "system", "content": "Sen MEB yönetmelik uzmanısın. Ağırbaşlı ve resmi bir dille cevap ver. Sayısal verilerde sorular.pdf'e sadık kal."}]
    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)
    messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"})
    completion = client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile", temperature=0)
    return completion.choices[0].message.content, docs

# --- ARAYÜZ ---

# Başlık
st.markdown("<div class='main-title'>🏛️ MEB Mevzuat Uzmanı</div>", unsafe_allow_html=True)

# 🚀 TURUNCU BUTON (Yazı yazma yerinin sağ üstünde yüzer)
st.markdown('<div class="floating-btn-container">', unsafe_allow_html=True)
st.link_button("📅 Sınıf Programı", "https://senin-linkin-buraya.com")
st.markdown('</div>', unsafe_allow_html=True)

# 💡 Dashboard Tipi Rehber
if not st.session_state.conversation:
    st.markdown("### 💡 Sorabileceğiniz Konular")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown('<div class="category-box"><div class="category-title">📜 Disiplin</div><div class="category-item">• Kopya cezası?</div><div class="category-item">• Sigara yasağı?</div><div class="category-item">• Kınama nedir?</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="category-box"><div class="category-title">⏳ Devamsızlık</div><div class="category-item">• Özürsüz sınır?</div><div class="category-item">• 30 gün kuralı?</div><div class="category-item">• 60 gün istisnası?</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="category-box"><div class="category-title">🎓 Başarı</div><div class="category-item">• Kaç zayıfla kalınır?</div><div class="category-item">• Takdir puanı?</div><div class="category-item">• Evli öğrenci?</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# 💬 Sohbet Akışı
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ⌨️ Giriş Alanı
if prompt := st.chat_input("Yönetmelik hakkında merak ettiklerinizi buraya yazın..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("⚖️ Mevzuat inceleniyor..."):
            cevap, kaynaklar = sorgula(prompt)
            st.markdown(cevap)
            
            if kaynaklar:
                st.markdown("---")
                st.markdown("📑 **Yönetmelik Referans Tablosu**")
                ref_df = pd.DataFrame({
                    "Kaynak": [f"Kesit {i+1}" for i in range(len(kaynaklar))],
                    "İçerik": [doc.page_content[:300] + "..." for doc in kaynaklar]
                })
                st.table(ref_df)
            
            st.session_state.conversation.append({"role": "assistant", "content": cevap})
