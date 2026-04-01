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
    .main-title { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; color: #1E1E1E; }
    
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
        margin-bottom: 10px;
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

# 🛡️ Güvenlik Filtresi
def filtre_kontrol(metin):
    yasakli = ["siyaset", "parti", "din", "ırk", "mezhep", "küfür", "argo", "hakaret", "aptal", "salak"]
    metin_low = metin.lower()
    return any(kelime in metin_low for kelime in yasakli)

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# 🤖 Sorgulama Fonksiyonu
def sorgula(soru):
    if filtre_kontrol(soru):
        return "⚠️ Uyarı: Mesajınız topluluk kurallarına aykırı içerik barındırıyor. Siyaset, din veya hakaret içeren sorular yanıtlanmaz. [TABLO_YOK]", []

    # Mevzuatı derinden tara
    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])
    
    messages = [{
        "role": "system", 
        "content": """Sen uzman bir MEB Mevzuat Asistanısın. Aşağıdaki kuralları hafızana al ve cevaplarını bunlara göre ver:

        1. DEVAMSIZLIK: Özürsüz 10 gün, toplam (özürlü+özürsüz) 30 gün sınırı vardır. Bu sınırları 1 saat bile aşan öğrenci sınıf tekrarına kalır. 60 günlük istisna sadece ağır hastalık/nakil durumlarındadır.
        2. YAŞ & EVLİLİK: Kayıt için 18 yaş altı olmak gerekir. Evli olanlar kaydedilmez, öğrenciyken evlenenler Açık Lise'ye gönderilir.
        3. BAŞARI: Ders geçme notu 50'dir. Sorumlu geçme sınırı 3 derstir. Toplam 6 dersten kalan sınıf tekrarı yapar.
        4. SINAVLAR: Her dersten en az 2 yazılı yapılır. Beceri sınavlarında %80 sınav, %20 iş dosyası etkilidir.
        5. DİSİPLİN: Kopya çekmek veya tütün (sigara) kullanmak doğrudan "Kınama" cezasıdır.
        6. ÖDÜLLER: Teşekkür (70.00-84.99), Takdir (85.00+).

        TALİMATLAR:
        - Cevabını verirken "Sabit veri" veya "Bağlam" gibi kelimeler kullanma.
        - Eğer soru MEB kurallarıyla ilgiliyse, cevabı ver ve asla [TABLO_YOK] ekleme (Referans tablosu çıksın).
        - Soru selamlaşma veya okul dışıysa [TABLO_YOK] ekle.
        - Cevaplarını mutlaka MEVZUAT KAYNAKLARI'ndaki döküman içerikleriyle destekle."""
    }]
    
    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": f"MEVZUAT KAYNAKLARI:\n{baglam}\n\nKULLANICI SORUSU: {soru}"})
    
    completion = client.chat.completions.create(
        messages=messages, 
        model="llama-3.3-70b-versatile", 
        temperature=0.1
    )
    return completion.choices[0].message.content, docs

# --- ARAYÜZ ---
st.markdown("<div class='main-title'>🏛️ MEB Mevzuat Uzmanı</div>", unsafe_allow_html=True)

st.markdown('<div class="floating-button-container">', unsafe_allow_html=True)
st.link_button("📅 Sınıf Programı", "https://sinifprogrami.streamlit.app/")
st.markdown('</div>', unsafe_allow_html=True)

# 💡 Öneri Kartları
st.markdown("### 💡 Hızlı Sorular")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="category-box"><div class="category-title">📜 Kayıt & Disiplin</div><div class="category-item">• Evlilik durumu ne olur?<br>• Kopya cezası nedir?</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="category-box"><div class="category-title">⏳ Devamsızlık</div><div class="category-item">• 30 gün kuralı nedir?<br>• Özürsüz kaç gün?</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="category-box"><div class="category-title">🎓 Başarı & Nakil</div><div class="category-item">• Kaç zayıfla kalınır?<br>• Nakil dönemi ne zaman?</div></div>', unsafe_allow_html=True)
st.markdown("---")

# Sohbet Akışı
for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("[TABLO_YOK]", ""))

# Giriş
if prompt := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("⚖️ Mevzuat maddeleri aranıyor..."):
            cevap, kaynaklar = sorgula(prompt)
            
            tablo_gizle = "[TABLO_YOK]" in cevap
            temiz_cevap = cevap.replace("[TABLO_YOK]", "").strip()
            
            st.markdown(temiz_cevap)
            
            # Kaynak maddeler uyuşma varsa görünür
            if kaynaklar and not tablo_gizle:
                st.markdown("📑 **İlgili Mevzuat Referansları**")
                ref_data = []
                for i, doc in enumerate(kaynaklar[:3]):
                    ref_data.append({
                        "Kaynak": f"Madde {i+1}",
                        "İçerik Özeti": doc.page_content[:250] + "..."
                    })
                st.table(pd.DataFrame(ref_data))
            
            st.session_state.conversation.append({"role": "assistant", "content": temiz_cevap})
