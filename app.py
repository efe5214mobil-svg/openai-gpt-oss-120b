import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
import base64
from PIL import Image

# 🎨 Sayfa Ayarları ve Tema
st.set_page_config(page_title="MEB Yönetmelik Asistanı", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #2c3e50 100%);
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTextInput input {
        background-color: #252525 !important;
        color: white !important;
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

# --- SIDEBAR & SINIFLAR (12->9 ve A->Z) ---
st.sidebar.header("📌 Sınıflar")
dersprogram_klasor = "dersprogram_dosyasi"
dosya_haritasi = {}

if os.path.exists(dersprogram_klasor):
    dosyalar = [f for f in os.listdir(dersprogram_klasor) if f.lower().endswith(".png")]
    for d in dosyalar:
        isim_ham = d.lower().replace(".png", "").replace(" ", "").replace("-", "").replace(".", "")
        match = re.search(r"(\d+)([a-z]+)", isim_ham)
        if match:
            sayi = match.group(1)
            harf = match.group(2).upper()
            gosterim_adi = f"{sayi} - {harf}"
            dosya_haritasi[gosterim_adi] = os.path.join(dersprogram_klasor, d)

    def sirala_mantigi(x):
        parcalar = re.search(r"(\d+) - ([A-Z]+)", x)
        if parcalar:
            return (-int(parcalar.group(1)), parcalar.group(2))
        return (0, x)

    sirali_isimler = sorted(dosya_haritasi.keys(), key=sirala_mantigi)
    if sirali_isimler:
        secilen_sinif = st.sidebar.selectbox("Sınıf Seçin:", sirali_isimler)
        st.sidebar.image(Image.open(dosya_haritasi[secilen_sinif]), use_container_width=True)

# 🛡️ GİZLİ FİLTRE (Base64 Koruma)
def uygunsuz_mu(soru):
    # Küfür, +18 ve alakasız kelimeler şifreli haldedir.
    data_enc = "a3VmdXIsYXJnbyxzaXlhc2V0LGRpbixpcmssaGFrYXJldCxwYXJ0aSxzZXgsc2Vrcyxwb3JubyxjaXBsYWssbWVtZSxnb3Qsc2lrLGFtayxwaXBpLHRhY2l6LG11c3RlaGNlbixnYXksbGV6Yml5ZW4sZmV0aXNsdWssdmFnaW5hLHBlbmlzLGVzY29ydCxuYWJlcixuYXNpbHNpbixzZWxhbSxtYWMsaGF2YSxmZW5lcmJhaGNlLGdhbGF0YXNhcmF5"
    yasakli_liste = base64.b64decode(data_enc).decode('utf-8').split(',')
    s = soru.lower()
    if any(k in s for k in yasakli_liste):
        return True, "⚠️ **Uyarı:** Girdiğiniz ifade okul kurallarına veya mevzuat kapsamına uygun değildir."
    return False, ""

# 🤖 SORGULAMA (MEB Yönetmelik Uzmanı)
def okul_asistani_sorgula(soru):
    hata, mesaj = uygunsuz_mu(soru)
    if hata: return mesaj, None

    docs = vector_db.similarity_search(soru, k=5)
    baglam = "\n\n".join([d.page_content for d in docs])

    messages = [
        {
            "role": "system", 
            "content": """Sen resmi bir MEB Yönetmelik Asistanısın. 
            Cevaplarına asla 'Evet' veya 'Hayır' diyerek başlama. Doğrudan döküman verisini paylaş.
            
            ÖNEMLİ KURALLAR:
            - Özürlü devamsızlık sınırı en fazla 20 GÜNDÜR.
            - Özürsüz devamsızlık sınırı 10 GÜNDÜR.
            - Toplam (özürlü+özürsüz) devamsızlık 30 GÜNDÜR. 30.5 olan kalır.
            - Sınıf geçme barajı 50 ortalamadır.
            - Sorumlu geçme şartlarını dökümana göre açıkla.
            - Üslubun ciddi, yardımsever ve tamamen yönetmelik odaklı olmalıdır."""
        }
    ]
    messages.extend(st.session_state.conversation[-2:])
    messages.append({"role": "user", "content": f"BAĞLAM:\n{baglam}\n\nSORU: {soru}"})

    try:
        completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=1000
        )
        return completion.choices[0].message.content, docs
    except:
        return "Şu an yanıt verilemiyor, lütfen sorunuzu yönetmelik çerçevesinde tekrar iletin.", None

# --- ANA EKRAN ---
st.title("🎓 MEB Yönetmelik Asistanı")

# 💡 Kapsamlı SSS Bölümü
with st.expander("❓ Sıkça Sorulan Sorular"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Devamsızlık Bilgileri**")
        st.write("- Özürlü devamsızlık sınırı kaç gün?")
        st.write("- Özürsüz 10 günü geçersem ne olur?")
        st.write("- Heyet raporu devamsızlığı etkiler mi?")
    with c2:
        st.markdown("**Sınıf Geçme ve Disiplin**")
        st.write("- Kaç zayıf ile sınıfta kalınır?")
        st.write("- Ortalamam 50 altındaysa sorumlu geçebilir miyim?")
        st.write("- Okula telefon getirmenin cezası nedir?")

for msg in st.session_state.conversation:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input("Yönetmelik hakkında sorunuzu yazın..."):
    st.session_state.conversation.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mevzuat taranıyor..."):
            cevap, kaynaklar = okul_asistani_sorgula(prompt)
            st.markdown(cevap)
            if kaynaklar:
                with st.expander("📖 Dayanak Yönetmelik Maddeleri"):
                    for k in kaynaklar: st.caption(k.page_content)
        
        st.session_state.conversation.append({"role": "assistant", "content": cevap})
