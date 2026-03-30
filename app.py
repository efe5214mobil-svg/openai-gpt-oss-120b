import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
import pandas as pd
from PIL import Image


load_dotenv()


api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=api_key)


st.title("MEB Yönetmelik Asistanı - Sohbet Hafızalı")
st.info("⚠️ Sadece MEB yönetmeliği ile ilgili sorular sorabilirsiniz. Uygunsuz sorular yanıtlanmayacaktır.")


st.sidebar.header("📌 Sınıf Ders Programı")
dersprogram_klasor = "dersprogram_dosyasi"

siniflar = []
for dosya in os.listdir(dersprogram_klasor):
    if dosya.endswith(".png"):

        sinif = dosya.replace(".png", "")
        if len(sinif) == 2:  
            sinif = f"{sinif[0]}/{sinif[1]}"
        siniflar.append(sinif)


secim = st.sidebar.selectbox("Sınıfı seçin:", siniflar)

dosya_adi = secim.replace("/", "") + ".png"
dosya_yolu = os.path.join(dersprogram_klasor, dosya_adi)


if os.path.exists(dosya_yolu):
    img = Image.open(dosya_yolu)
    st.sidebar.image(img, caption=f"{secim} Ders Programı", use_column_width=True)
else:
    st.sidebar.warning(f"{secim} için görsel bulunamadı!")

# 🧠 VECTOR DB yükle
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory="okul_asistani_gpt_db",
        embedding_function=embeddings
    )
    return db

vector_db = load_vector_db()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

def temizle_cevap(cevap):
    cevap = re.sub(r'(\b\d{2,4}\b)(?:/\s*\1)+', r'\1', cevap)
    cevap = re.sub(r'\s{2,}', ' ', cevap)
    return cevap


def uygunsuz_mu(soru):
    anahtar_kelimeler = ["küfür", "argo", "siyaset", "ülke", "din", "ırk", "cinsiyet"]
    soru_lower = soru.lower()
    return any(anahtar in soru_lower for anahtar in anahtar_kelimeler)


def okul_asistani_sorgula(soru):
    if uygunsuz_mu(soru):
        return "⚠️ Bu soru uygun değil. Lütfen yalnızca MEB yönetmeliği ile ilgili resmi sorular sorun.", None, None

 
    arama_sorgusu = f"{soru} meb yönetmelik maddesi devamsızlık şartları"

    docs = vector_db.similarity_search_with_score(arama_sorgusu, k=3)
    docs = sorted(docs, key=lambda x: x[1])[:3]
    docs = [doc[0] for doc in docs]

    if not docs:
        return "Veri bulunamadı.", None, None


    baglam = "\n\n".join([doc.page_content[:500] for doc in docs])


    messages = [
        {"role": "system", "content": """
Sen MEB yönetmeliği uzmanısın.
Kurallar:
- Sadece verilen bağlama göre cevap ver
- Eğer madde varsa belirt
- En yakın bilgiyi kullan
- "Bulunamadı" deme
- Cevapta küfür, hakaret, siyaset, din, ırk, cinsiyet ile ilgili içerik olamaz
- Cevap sadece resmi MEB yönetmeliği ile ilgili olmalı
- Anlamsız tekrarlar ve saçma ifadeler kullanma
"""}
    ]


    for msg in st.session_state.conversation:
        messages.append(msg)

    messages.append({"role": "user", "content": f"{baglam}\n\nSoru: {soru}"})

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=500
    )

    cevap = chat_completion.choices[0].message.content
    cevap = temizle_cevap(cevap)

    st.session_state.conversation.append({"role": "user", "content": soru})
    st.session_state.conversation.append({"role": "assistant", "content": cevap})

    tablo_data = {"Madde Özeti": [doc.page_content[:100]+"..." for doc in docs]}
    tablo_df = pd.DataFrame(tablo_data)
    kaynaklar = [doc.page_content[:200] for doc in docs]

    return cevap, tablo_df, kaynaklar


for msg in st.session_state.conversation:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

if prompt := st.chat_input("Sorunuzu yazın:"):
    with st.spinner("Yanıt hazırlanıyor..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        cevap, tablo_df, kaynaklar = okul_asistani_sorgula(prompt)
        with st.chat_message("assistant"):
            st.markdown(cevap)
            if tablo_df is not None:
                st.markdown("📊 **İlgili Madde Tablosu:**")
                st.table(tablo_df)
            if kaynaklar is not None:
                st.markdown("📚 **Kaynaklar:**")
                for k in kaynaklar:
                    st.markdown(f"- {k}")
