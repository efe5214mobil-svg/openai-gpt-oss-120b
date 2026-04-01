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

# 🖌️ Modern CSS
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .main-title { font-size: 2.2rem; font-weight: 800; text-align: center; margin-bottom: 1rem; }
    
    /* 🚀 Sınıf Programı Butonu (Sağ Altta) */
    .floating-button-container {
        position: fixed;
        bottom: 90px; 
        right: 5%; 
        z-index: 999999;
    }

    .stLinkButton a {
        background-color: #FF8C00 !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        border: 2px solid white !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        text-decoration: none !important;
        display: inline-flex !important;
    }

    /* Kategori Kartları */
    .category-box {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 15px;
        padding: 15px;
        border-top: 4px solid #FF4B4B;
        margin-bottom: 10px;
    }
    .category-title { font-weight: bold; color: #FF4B4B; margin-bottom: 8px; font-size: 1.1rem; }
    .category-item { font-size: 0.85rem; color: #444; line-height: 1.4; }
    
    @media (prefers-color-scheme: dark) {
        .category-item { color: #CCC; }
        .category-box { background-color: rgba(255, 255, 255, 0.05); }
    }
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

# ❌ Filtre
def uygunsuz_mu(soru):
    yasakli = ["küfür", "argo", "siyaset", "parti", "din", "ırk", "hakaret", "aptal", "salak"]
    return any(k in soru.lower() for k in yasakli)

# 🤖 Sorgulama
def sorgula(soru):
    docs = vector_db.similarity_search(soru, k=3)
    baglam = "\n\n".join([doc.page_content for doc in docs])
    
    messages = [{
        "role": "system",
        "content": """Sen MEB yönetmelik uzmanısın. Resmi bir dille yanıt ver.
        [BİLGİ]: Devamsızlık özürsüz 10, toplam 30 gündür. Geçme notu 50'dir. 
        Eğer soru saçmaysa cevabına mutlaka 'KAPSAM_DISI' ekle."""
    }]
    
    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)
    messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"})

    completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    res = completion.choices[0].message.content
    return (res if res else "Üzgünüm, yanıt oluşturulamadı."), docs

# --- EKRAN ---
st.markdown("<div class='main-title'>🏛️ MEB Mevzuat Uzmanı</div>", unsafe_allow_html=True)

# 🚀 BUTON
st.markdown('<div class="floating-button-container">', unsafe_allow_html=True)
st.link_button("📅 Sınıf Programı", "https://senin-linkin-buraya.com")
st.markdown('</div>', unsafe_allow_html=True)

# 💡 KARTLAR (Sürekli Görünür)
st.markdown("### 💡 Önerilen Konular")
c1, c2, c3 = st.columns(3)
with c1: st.markdown('<div class="category-box"><div class="category-title">📜 Disiplin</div><div class="category-item">• Kopya cezası?<br>• Kınama nedir?</div></div>', unsafe_allow_html=True)
with c2: st.markdown('<div class="category-box"><div class="category-title">⏳ Devamsızlık</div><div class="category-item">• Özürsüz sınır?<br>• 30 gün kuralı?</div></div>', unsafe_allow_html=True)
with c3: st.markdown('<div class="category-box"><div class="category-title">🎓 Başarı</div><div class="category-item">• Kaç zayıf?<br>• Takdir puanı?</div></div>', unsafe_allow_html=True)

st.divider()

# 💬 SOHBET
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("KAPSAM_DISI", ""))

# ⌨️ GİRİŞ
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    if uygunsuz_mu(prompt):
        cevap = "⚠️ Lütfen topluluk kurallarına uygun sorular sorunuz."
        st.session_state.conversation.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"): st.markdown(cevap)
        st.session_state.conversation.append({"role": "assistant", "content": cevap})
    else:
        st.session_state.conversation.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("⚖️ İnceleniyor..."):
                raw_cevap, kaynaklar = sorgula(prompt)
                is_out = "KAPSAM_DISI" in raw_cevap
                clean = raw_cevap.replace("KAPSAM_DISI", "").strip()
                st.markdown(clean)
                
                if kaynaklar and not is_out:
                    st.markdown("---")
                    st.table(pd.DataFrame({
                        "Kaynak": [f"Madde {i+1}" for i in range(len(kaynaklar))],
                        "İçerik": [d.page_content[:200] + "..." for d in kaynaklar]
                    }))
                st.session_state.conversation.append({"role": "assistant", "content": clean})
