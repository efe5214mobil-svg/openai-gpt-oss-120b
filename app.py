import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
import os
import re

# 🔐 API ve Çevre Değişkenleri
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 🎨 Sayfa Yapılandırması
st.set_page_config(page_title="MEB Mevzuat Asistanı", page_icon="🏛️", layout="centered")

# 🖌️ Modern CSS (Görsel yapı korunur)
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .main-title { font-size: 2.2rem; font-weight: 800; text-align: center; margin-bottom: 1rem; }
    
    [data-testid="stChatMessage"] {
        border-radius: 18px;
        padding: 15px;
        border: 1px solid rgba(128, 128, 128, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# 🧠 Vektör Veritabanı
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=embeddings)

vector_db = load_vector_db()

# 🗂️ Sohbet Hafızası
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# ❌ Güvenlik Filtresi
def uygunsuz_mu(soru):
    yasakli = ["küfür", "argo", "siyaset", "parti", "din", "ırk", "hakaret"]
    return any(k in soru.lower() for k in yasakli)

# 🤖 Sorgulama Fonksiyonu (Llama 3.3 70B)
def sorgula(soru):
    docs = vector_db.similarity_search(soru, k=4)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    messages = [{
        "role": "system",
        "content": """Sen T.C. Milli Eğitim Bakanlığı yönetmelikleri konusunda uzman, ağırbaşlı ve resmi bir asistansın.
        
        [KESİN KURALLAR - PDF VERİLERİ]:
        - Devamsızlık: Özürsüz 10 gün sınır (aşılırsa başarısız sayılır), toplam sınır 30 gündür.
        - Başarı: Bir dersten başarılı sayılmak için yılsonu puanı en az 50 olmalıdır.
        - Sınıf Geçme: En fazla 3 dersten başarısız olanlar sorumlu geçer; başarısız ders sayısı 6'yı geçerse sınıf tekrarı yapılır.
        - Disiplin: Kopya çekmek veya tütün mamulü kullanmak 'Kınama' cezası gerektirir.
        - Ödüller: Teşekkür (70.00-84.99), Takdir (85.00+).
        - Kayıt: Evli olanların kaydı yapılmaz; öğrenciyken evlenenler Açık Lise'ye aktarılır.
        - Süre: Ders saati okulda 40 dakikadır.

        TALİMAT: Yanıtlarını resmi bir dille, sadece bu kurallara ve bağlama dayanarak ver."""
    }]

    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)

    messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=800
        )
        return completion.choices[0].message.content, docs
    except Exception as e:
        return f"⚠️ Bir hata oluştu: {str(e)}", None

# --- ARAYÜZ ---
st.markdown("<div class='main-title'>🏛️ MEB Mevzuat Uzmanı</div>", unsafe_allow_html=True)
st.caption("<p style='text-align: center;'>Resmi yönetmelik analiz asistanı</p>", unsafe_allow_html=True)

# 💬 Sohbet Geçmişi
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ⌨️ Tek Giriş Alanı (Gereksiz tüm prompt/oneri yapıları kaldırıldı)
if prompt := st.chat_input("Sorunuzu yazın..."):
    
    if uygunsuz_mu(prompt):
        hata = "❌ Üzgünüm, sorunuz yönetmelik kapsamı dışındadır veya uygunsuz içerik barındırmaktadır."
        st.session_state.conversation.append({"role": "user", "content": prompt})
        st.session_state.conversation.append({"role": "assistant", "content": hata})
        st.rerun()

    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("⚖️ Mevzuat inceleniyor..."):
            cevap, kaynaklar = sorgula(prompt)
            st.markdown(cevap)
            st.session_state.conversation.append({"role": "assistant", "content": cevap})
            
            if kaynaklar:
                with st.expander("📄 İlgili Madde Referansları"):
                    for doc in kaynaklar:
                        st.caption(doc.page_content[:400] + "...")
                        st.divider()
