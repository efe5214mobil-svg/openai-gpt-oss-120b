import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import pandas as pd
from PIL import Image

# 🎨 Sayfa Konfigürasyonu
st.set_page_config(page_title="MEB Mevzuat Uzmanı", page_icon="📜", layout="wide")

# 💅 Gelişmiş Arayüz Tasarımı
st.markdown("""
    <style>
    .stApp { background-color: #f4f7f6; }
    .stChatMessage { border: 1px solid #d1d5db; border-radius: 12px; }
    .assistant-card { 
        background-color: #ffffff; 
        padding: 20px; 
        border-left: 5px solid #1e3a8a; 
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 🛡️ Güvenlik ve Filtreleme (Gelişmiş)
def filtre_kontrol(metin):
    yasakli = ["küfür1", "siyaset", "parti", "din", "inanç", "hakaret", "ülke gündemi", "ekonomi"]
    return any(kelime in metin.lower() for kelime in yasakli)

# 🤖 Detaycı Sorgu Fonksiyonu
def okul_asistani_sorgula(soru):
    if filtre_kontrol(soru):
        return "⚠️ **GÜVENLİK UYARISI:** Sistemimiz sadece MEB yönetmeliğiyle ilgili teknik soruları yanıtlamak üzere programlanmıştır. Lütfen siyaset, din veya kişisel yorum içeren ifadelerden kaçınınız.", None

    # Vektör Araması (Daha fazla bağlam için k=5)
    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    # 🧠 Detaycı ve Açıklayıcı Prompt
    messages = [
        {
            "role": "system", 
            "content": """Sen Milli Eğitim Bakanlığı (MEB) Ortaöğretim Kurumları Yönetmeliği konusunda kıdemli bir hukuk ve mevzuat uzmanısın.
            
            CEVAPLAMA STRATEJİN:
            1. Önce sorunun cevabını net bir şekilde (EVET/HAYIR/BELİRTİLMEMİŞ) ver.
            2. Ardından, ilgili yönetmelik maddesine dayanarak DETAYLI bir açıklama yap. 
            3. Eğer bağlamda madde numarası veya resmi terimler geçiyorsa bunları mutlaka kullan.
            4. Cevabın profesyonel, eğitici ve tamamen dökümana sadık olmalı.
            5. Siyaset, din veya yönetmelik dışı konulara asla girme; gerekirse 'Bu konu mevzuat dışıdır' de.
            6. Asla uydurma bilgi verme. Bağlam yetersizse 'Verilen dökümanlarda bu detay yer almamaktadır' şeklinde belirt."""
        }
    ]
    
    # Hafıza ve Bağlam
    messages.extend(st.session_state.conversation[-2:]) # Hafızayı temiz tutmak için son 2 mesaj
    messages.append({"role": "user", "content": f"Aşağıdaki yönetmelik metnine göre soruyu detaylandırarak açıkla:\n\nBAĞLAM:\n{baglam}\n\nSORU: {soru}"})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.1, # Hafif esneklik ama yüksek sadakat
            max_tokens=800  # Detaylı açıklama için limiti artırdık
        )
        cevap = completion.choices[0].message.content
    except Exception as e:
        cevap = f"Sistem hatası: {e}"

    return cevap, docs

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://www.meb.gov.tr/logo.png", width=100)
    st.header("Yönetim Paneli")
    
    # Ders Programı Bölümü
    klasor = "dersprogram_dosyasi"
    if os.path.exists(klasor):
        dosyalar = [f for f in os.listdir(klasor) if f.lower().endswith(".png")]
        if dosyalar:
            secim = st.selectbox("📌 Sınıf Programı Görüntüle", dosyalar)
            st.image(os.path.join(klasor, secim), use_container_width=True)
    
    if st.button("🗑️ Sohbeti Sıfırla"):
        st.session_state.conversation = []
        st.rerun()

# --- ANA EKRAN ---
st.title("📜 MEB Mevzuat ve Yönetmelik Asistanı")
st.markdown("---")

# Sohbet Akışı
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Örn: Devamsızlık nedeniyle sınıfta kalma kuralı nedir?"):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        status = st.status("🔍 Mevzuat taranıyor ve analiz ediliyor...", expanded=True)
        cevap, kaynak_docs = okul_asistani_sorgula(prompt)
        
        status.update(label="✅ Analiz Tamamlandı", state="complete", expanded=False)
        
        # Ana Cevap
        st.markdown(f'<div class="assistant-card">{cevap}</div>', unsafe_allow_html=True)
        
        # Kaynak Maddeler (Expandable)
        if kaynak_docs:
            with st.expander("📖 Referans Alınan Yönetmelik Maddeleri"):
                for i, doc in enumerate(kaynak_docs):
                    st.info(f"**Kaynak {i+1}:**\n\n{doc.page_content}")
        
        st.session_state.conversation.append({"role": "assistant", "content": cevap})
