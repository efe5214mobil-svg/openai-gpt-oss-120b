import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
import os

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
    .main-title { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; color: #FFFFFF; }
    
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
    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])
    
    messages = [{
        "role": "system", 
        "content": """Sen uzman bir MEB Mevzuat Asistanısın. Aşağıdaki güncel kurallara tam bağlı kalarak, profesyonel ama anlaşılır bir dille cevap ver:

        [TEMEL KURALLAR]:
        - Ders Süresi: Okulda 40 dk, işletmelerde 60 dk.
        - Kayıt & Yaş: Kayıt günü 18 yaşını bitirmemiş olmak şart. Evli olanlar kaydedilmez, öğrenciyken evlenenler Açık Lise'ye nakledilir.
        - Devamsızlık: Özürsüz sınırı 10 gün. Toplam (özürlü+özürsüz) sınırı 30 gün. İstisnai hallerde (organ nakli, ağır hastalık vb.) toplam 60 gün.
        - Geç Gelme: Sadece 1. ders için geçerlidir, sonrası devamsızlık sayılır.
        - Başarı & Sınıf Geçme: Geçme notu 50. Her dersten en az 2 yazılı zorunlu. Max 3 ders zayıfı olan 'Sorumlu' geçer. Toplam 6+ zayıfı olan sınıf tekrarı yapar.
        - Beceri Sınavı: %80 sınav puanı + %20 iş dosyası.
        - Nakil: Aralık ve Mayıs ayları hariç her ayın ilk iş günü başvurulabilir.
        - Disiplin: Kopya ve sigara kullanımı 'Kınama' cezasıdır.
        - Ödüller: Teşekkür (70.00-84.99), Takdir (85.00+).

        Talimat: Teknik referans (madde no vb.) verme, doğrudan cevap ver. Dökümanlardaki (BAĞLAM) bilgilerle yukarıdaki kuralları harmanla."""
    }]
    
    for msg in st.session_state.conversation[-4:]:
        messages.append(msg)
    
    messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nKULLANICI SORUSU: {soru}"})
    
    completion = client.chat.completions.create(
        messages=messages, 
        model="llama-3.3-70b-versatile", 
        temperature=0.1
    )
    return completion.choices[0].message.content

# --- ARAYÜZ ---
st.markdown("<div class='main-title'>🏛️ MEB Yönetmelik Asistanı</div>", unsafe_allow_html=True)

st.markdown('<div class="floating-button-container">', unsafe_allow_html=True)
st.link_button("📅 Sınıf Programı", "https://sinifprogrami.streamlit.app/")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### 💡 Hızlı Sorular")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="category-box"><div class="category-title">📜 Kayıt & Disiplin</div><div class="category-item">• Evlilik durumu ne olur?<br>• Yaş sınırı kaç?</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="category-box"><div class="category-title">⏳ Devamsızlık</div><div class="category-item">• 30 gün kuralı nedir?<br>• Geç gelme durumu?</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="category-box"><div class="category-title">🎓 Başarı & Nakil</div><div class="category-item">• Kaç zayıfla kalınır?<br>• Nakil dönemi ne zaman?</div></div>', unsafe_allow_html=True)
st.markdown("---")

for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    
    if filtre_kontrol(prompt):
        st.warning("⚠️ Uyarı: Mesajınız topluluk kurallarına aykırı içerik (küfür, siyaset, din vb.) barındırdığı için engellenmiştir. Lütfen sadece yönetmelik ile ilgili sorular sorun.")
    else:
        st.session_state.conversation.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("⚖️ İçerik inceleniyor..."):
                cevap = sorgula(prompt)
                st.markdown(cevap)
                st.session_state.conversation.append({"role": "assistant", "content": cevap})
