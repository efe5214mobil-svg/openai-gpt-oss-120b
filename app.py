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

# 🖌️ Modern CSS (Turuncu Buton Sağda ve Rehber Kartları)
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .main-title { font-size: 2.2rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; }
    
    /* Turuncu Sınıf Programı Butonu Özelleştirme */
    .stLinkButton a {
        background-color: #FF8C00 !important; /* Turuncu */
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.6rem 1.2rem !important;
        font-weight: 700 !important;
        text-decoration: none !important;
        display: inline-flex !important;
        transition: 0.3s ease !important;
        float: right; /* Sağa yasla */
    }
    .stLinkButton a:hover {
        background-color: #E67E22 !important;
        box-shadow: 0 4px 15px rgba(255, 140, 0, 0.3) !important;
        transform: translateY(-2px);
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

# Üst Başlık ve Sağdaki Buton Düzeni
head_col1, head_col2 = st.columns([3, 1])
with head_col1:
    st.markdown("<div class='main-title'>🏛️ MEB Mevzuat Uzmanı</div>", unsafe_allow_html=True)
with head_col2:
    st.write("") # Boşluk ayarı
    st.link_button("📅 Sınıf Programı", "https://senin-linkin-buraya.com")

# 💡 Dashboard Tipi Rehber (Kategoriler Yan Yana)
if not st.session_state.conversation:
    st.markdown("### 💡 Sorabileceğiniz Konular")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
        <div class="category-box">
            <div class="category-title">📜 Disiplin</div>
            <div class="category-item">• Kopya çekmenin cezası?</div>
            <div class="category-item">• Sigara içmek suç mu?</div>
            <div class="category-item">• Kınama cezası nedir?</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="category-box">
            <div class="category-title">⏳ Devamsızlık</div>
            <div class="category-item">• Özürsüz sınır kaç gün?</div>
            <div class="category-item">• 30 gün kuralı nedir?</div>
            <div class="category-item">• 60 günlük istisnalar?</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="category-box">
            <div class="category-title">🎓 Başarı & Kayıt</div>
            <div class="category-item">• Kaç zayıfla kalınır?</div>
            <div class="category-item">• Takdir puanı kaçtır?</div>
            <div class="category-item">• Evli öğrenci durumu?</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# 💬 Sohbet Akışı
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ⌨️ Giriş Alanı (Buradaki prompt yazma yerinin hemen üstünde buton sağda kalır)
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
                    "İlgili Madde İçeriği": [doc.page_content[:300] + "..." for doc in kaynaklar]
                })
                st.table(ref_df)
            
            st.session_state.conversation.append({"role": "assistant", "content": cevap})
