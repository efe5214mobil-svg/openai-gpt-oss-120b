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
    .main-title { font-size: 2.2rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; }
    
    .floating-button-container {
        position: fixed;
        bottom: 85px; 
        right: 10%; 
        z-index: 999999;
    }

    .stLinkButton a {
        background-color: #FF8C00 !important;
        color: white !important;
        border-radius: 25px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 700 !important;
        text-decoration: none !important;
        border: 2px solid white !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }

    .category-box {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 15px;
        padding: 18px;
        border-top: 4px solid #FF4B4B;
        height: 100%;
    }
    .category-title { font-weight: bold; color: #FF4B4B; margin-bottom: 10px; font-size: 1.15rem; }
    .category-item { font-size: 0.88rem; margin-bottom: 6px; color: #444; }
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

# 🛡️ GÜVENLİK FİLTRESİ
def icerik_denetimi(soru):
    # Engellenen anahtar kelimeler (Küfür, Siyaset, Din, Irkçılık)
    yasakli_kelimeler = ["siyaset", "din", "ırk", "parti", "küfür_buraya", "hakaret", "aptal", "salak"] # Burayı genişletebilirsin
    soru_low = soru.lower()
    if any(kelime in soru_low for kelime in yasakli_kelimeler):
        return True
    return False

# 🤖 Sorgulama Fonksiyonu
def sorgula(soru):
    # 1. Önce içerik denetimi yap
    if icerik_denetimi(soru):
        return "Üzgünüm, topluluk kurallarımız gereği din, dil, ırk, siyaset veya hakaret içeren sorulara yanıt veremiyorum. Size sadece MEB yönetmeliği hakkında yardımcı olabilirim. [TABLO_YOK]", []

    docs = vector_db.similarity_search(soru, k=3)
    baglam = "\n\n".join([doc.page_content for doc in docs])
    
    messages = [{
        "role": "system", 
        "content": """Sen MEB yönetmelik uzmanısın. 
        - Normal selamlaşmalara (Selam, Merhaba, Nasılsın vb.) samimi ve kısa cevaplar ver.
        - Mevzuat dışı, şaka veya selamlaşma durumlarında cevabın sonuna [TABLO_YOK] etiketini ekle.
        - Yönetmelik sorularında [SABİT PDF VERİLERİ]'ni kullan ve etiket ekleme.
        
        [SABİT PDF VERİLERİ]:
        - Devamsızlık: Özürsüz 10, Toplam 30 gün. (İstisna: 60 gün)
        - Başarı: Geçme sınırı 50 puan.
        - Sınıf Geçme: En fazla 3 dersten sorumlu geçiş, 6+ zayıfta sınıf tekrarı.
        - Ödül: Teşekkür 70+, Takdir 85+.
        - Disiplin: Kopya/Sigara 'Kınama' gerektirir.
        - Kayıt: Evli olanların kayıtları yapılamaz."""
    }]
    
    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": f"BAĞLAM (Mevzuat):\n{baglam}\n\nSORU: {soru}"})
    
    completion = client.chat.completions.create(
        messages=messages, 
        model="llama-3.3-70b-versatile", 
        temperature=0.2 # Selamlaşmalar için biraz daha esnek yaptık
    )
    return completion.choices[0].message.content, docs

# --- ARAYÜZ ---
st.markdown("<div class='main-title'>🏛️ MEB Mevzuat Uzmanı</div>", unsafe_allow_html=True)

st.markdown('<div class="floating-button-container">', unsafe_allow_html=True)
st.link_button("📅 Sınıf Programı", "https://sinifprogrami.streamlit.app/")
st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state.conversation:
    st.markdown("### 💡 Hızlı Öneriler")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown('<div class="category-box"><div class="category-title">📜 Disiplin</div><div class="category-item">"Kopya cezası nedir?"</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="category-box"><div class="category-title">⏳ Devamsızlık</div><div class="category-item">"30 gün kuralı nedir?"</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="category-box"><div class="category-title">🎓 Başarı</div><div class="category-item">"Takdir kaç puan?"</div></div>', unsafe_allow_html=True)

for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("[TABLO_YOK]", ""))

if prompt := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("⚖️ Mevzuat inceleniyor..."):
            cevap, kaynaklar = sorgula(prompt)
            
            tablo_gizle = "[TABLO_YOK]" in cevap
            temiz_cevap = cevap.replace("[TABLO_YOK]", "").strip()
            
            st.markdown(temiz_cevap)
            
            if kaynaklar and not tablo_gizle:
                st.markdown("---")
                st.markdown("📑 **İlgili Mevzuat Referansları**")
                ref_data = []
                for i, doc in enumerate(kaynaklar):
                    sayfa = doc.metadata.get('page', 'Kaynak PDF')
                    ref_data.append({
                        "Kaynak": f"Madde {i+1}",
                        "İçerik Özeti": doc.page_content[:200] + "...",
                        "Konum": sayfa
                    })
                st.table(pd.DataFrame(ref_data))
            
            st.session_state.conversation.append({"role": "assistant", "content": temiz_cevap})
