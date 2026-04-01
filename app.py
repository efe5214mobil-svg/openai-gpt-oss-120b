import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
import os
import re

# 🔐 API ve Çevre Değişkenleri
load_dotenv()
api_anahtari = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
istemci = Groq(api_key=api_anahtari)

# 🎨 Sayfa Yapılandırması
st.set_page_config(page_title="MEB Mevzuat Asistanı", page_icon="🏛️", layout="centered")

# 🖌️ Modern Görünüm (CSS)
st.markdown("""
<style>
    .stApp { font-family: 'Inter', sans-serif; }
    .ana-baslik { font-size: 2.5rem; font-weight: 800; text-align: center; margin-bottom: 1.5rem; color: #FFFFFF; }
    
    .yuzen-buton-alani {
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

    .kategori-kutusu {
        background-color: rgba(128, 128, 128, 0.05);
        border-radius: 15px;
        padding: 18px;
        border-top: 4px solid #FF4B4B;
        height: 100%;
        margin-bottom: 10px;
    }
    .kategori-basligi { font-weight: bold; color: #FF4B4B; margin-bottom: 10px; font-size: 1.15rem; }
    .kategori-maddesi { font-size: 0.88rem; margin-bottom: 6px; color: #444; }
</style>
""", unsafe_allow_html=True)

# 🧠 Veri Tabanı Yükleme
@st.cache_resource
def veri_tabanini_yukle():
    gomme_modeli = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="okul_asistani_gpt_db", embedding_function=gomme_modeli)

vektor_tabani = veri_tabanini_yukle()

# 🛡️ GELİŞMİŞ GÜVENLİK SÜZGECİ
def suzgec_kontrolu(metin):
    # Benzer karakterleri harfe dönüştür
    karakter_haritasi = {
        '1': 'i', '0': 'o', '3': 'e', '4': 'a', '5': 's', '7': 't', '8': 'b', 
        '@': 'a', '$': 's', '€': 'e', '!': 'i'
    }
    
    # 1. Metni küçült
    temiz_metin = metin.lower()
    
    # 2. Benzer karakterleri değiştir
    for eski, yeni in karakter_haritasi.items():
        temiz_metin = temiz_metin.replace(eski, yeni)
    
    # 3. Harf ve rakam dışındaki her şeyi sil (boşluk, nokta, sembol)
    temiz_metin = re.sub(r'[^a-z0-9]', '', temiz_metin)
    
    yasakli_kelimeler = [
        "oc", "aq", "amk", "amq", "pic", "got", "sik", "amc", "yarrak", "fassak", "tassak", "dassak",
        "orospu", "fahise", "pezevenk", "kahpe", "gavat", "meme", "gay", "lezbiyen", "lgbt", 
        "travesti", "seks", "sex", "porno", "vajina", "penis", "siyaset", "parti", "teror", 
        "it", "kopek", "serefsiz", "haysiyetsiz", "beyinsiz", "gerizekali"
    ]
    
    return any(yasakli in temiz_metin for yasakli in yasakli_kelimeler)

if "sohbet_gecmisi" not in st.session_state:
    st.session_state.sohbet_gecmisi = []

# 🤖 Yanıt Oluşturucu
def cevap_olustur(soru):
    ilgili_belgeler = vektor_tabani.similarity_search(soru, k=5)
    kaynak_metin = "\n\n".join([belge.page_content for belge in ilgili_belgeler])
    
    iletiler = [{
        "role": "system", 
        "content": """Sen uzman bir MEB Mevzuat Asistanısın. 
        GÖREVİN: Sadece Milli Eğitim Bakanlığı yönetmelikleri hakkında bilgi vermek.
        ÖNEMLİ: Cevaplarında yabancı kelime kullanma, tamamen Türkçe ifadeler seç.
        KURALLAR: Ders süresi 40/60dk, Devamsızlık 10/30 gün, Geçme 50, Sorumluluk en fazla 3, 6+ zayıf tekrar."""
    }]
    
    for ileti in st.session_state.sohbet_gecmisi[-4:]:
        iletiler.append(ileti)
    
    iletiler.append({"role": "user", "content": f"KAYNAK VERİLER:\n{kaynak_metin}\n\nKULLANICI SORUSU: {soru}"})
    
    yanit = istemci.chat.completions.create(
        messages=iletiler, 
        model="llama-3.3-70b-versatile", 
        temperature=0.1
    )
    return yanit.choices[0].message.content

# --- ARAYÜZ ---
st.markdown("<div class='ana-baslik'>🏛️ MEB Yönetmelik Asistanı</div>", unsafe_allow_html=True)

st.markdown('<div class="yuzen-buton-alani">', unsafe_allow_html=True)
st.link_button("📅 Sınıf Programı", "https://sinifprogrami.streamlit.app/")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("### 💡 Hızlı Sorular")
sutun1, sutun2, sutun3 = st.columns(3)
with sutun1:
    st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">📜 Kayıt & Disiplin</div><div class="kategori-maddesi">• Evlilik durumu ne olur?<br>• Yaş sınırı kaç?</div></div>', unsafe_allow_html=True)
with sutun2:
    st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">⏳ Devamsızlık</div><div class="kategori-maddesi">• 30 gün kuralı nedir?<br>• Özürsüz kaç gün?</div></div>', unsafe_allow_html=True)
with sutun3:
    st.markdown('<div class="kategori-kutusu"><div class="kategori-basligi">🎓 Başarı & Nakil</div><div class="kategori-maddesi">• Kaç zayıfla kalınır?<br>• Nakil dönemi ne zaman?</div></div>', unsafe_allow_html=True)
st.markdown("---")

# Geçmişi Ekrana Yaz
for ileti in st.session_state.sohbet_gecmisi:
    with st.chat_message(ileti["role"]):
        st.markdown(ileti["content"])

# Kullanıcı Girişi
if girdi := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    
    if suzgec_kontrolu(girdi):
        st.error("⚠️ Uyarı: İletiniz topluluk kurallarına aykırı veya uygunsuz içerik barındırdığı için engellenmiştir.")
    else:
        st.session_state.sohbet_gecmisi.append({"role": "user", "content": girdi})
        with st.chat_message("user"):
            st.markdown(girdi)

        with st.chat_message("assistant"):
            with st.spinner("⚖️ İnceleniyor..."):
                cevap = cevap_olustur(girdi)
                st.markdown(cevap)
                st.session_state.sohbet_gecmisi.append({"role": "assistant", "content": cevap})
