import streamlit as st
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import re
import pandas as pd
import camelot
import matplotlib.pyplot as plt

# -----------------------------
# 🔐 API ve Streamlit Başlangıç
# -----------------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

st.title("Okul Asistanı - Ders Programı & MEB Yönetmeliği")
st.info("⚠️ 1) MEB yönetmeliği soruları sorabilirsiniz.\n⚠️ 2) Ders programı için sınıf ismi yazabilirsiniz (örn. 9/B, 9B, 9-B, 10C).")

# -----------------------------
# 🧠 VECTOR DB
# -----------------------------
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

# -----------------------------
# MEB Asistanı Fonksiyonları
# -----------------------------
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

    kaynaklar = [doc.page_content[:200] for doc in docs]
    tablo_data = {"Madde Özeti": [doc.page_content[:100]+"..." for doc in docs]}
    tablo_df = pd.DataFrame(tablo_data)

    return cevap, tablo_df, kaynaklar

# -----------------------------
# Ders Programı Fonksiyonları
# -----------------------------
@st.cache_resource
def load_pdf_tables(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages='all')
    if not tables:
        st.warning("PDF’den tablo okunamadı. Lütfen PDF’i kontrol edin.")
        return pd.DataFrame()
    df_list = []
    for t in tables:
        df_temp = t.df
        try:
            class_row = df_temp[df_temp.apply(lambda row: row.astype(str).str.contains(r'\d{1,2}[/-]?[A-D]').any(), axis=1)]
            if not class_row.empty:
                sinif = class_row.iloc[0].astype(str).str.extract(r'(\d{1,2}[/-]?[A-D])')[0].values[0]
            else:
                sinif = "Bilinmiyor"
        except Exception:
            sinif = "Bilinmiyor"
        df_temp['Sınıf'] = sinif
        df_list.append(df_temp)
    if df_list:
        df_all = pd.concat(df_list, ignore_index=True)
    else:
        df_all = pd.DataFrame()
    return df_all

def normalize_class(s):
    s = s.upper().replace("-", "/").replace(" ", "")
    if "/" not in s and len(s) > 1:
        s = s[:-1] + "/" + s[-1]
    return s

pdf_path = "ders_programi.pdf"  # PDF yolu
df_ders = load_pdf_tables(pdf_path)

def ders_programi_goster(sinif_input):
    sinif = normalize_class(sinif_input)
    if df_ders.empty or 'Sınıf' not in df_ders.columns:
        st.warning("PDF’den ders programı okunamadı veya sınıf bilgisi yok.")
        return
    ders_programi = df_ders[df_ders['Sınıf'].apply(normalize_class) == sinif]
    if ders_programi.empty:
        st.warning(f"{sinif} için veri bulunamadı!")
        return

    # Matplotlib ile görsel tablo oluştur
    fig, ax = plt.subplots(figsize=(12, len(ders_programi)*0.5 + 2))
    ax.axis('off')
    tbl = ax.table(cellText=ders_programi.values, colLabels=ders_programi.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    st.pyplot(fig)

# -----------------------------
# Kullanıcı Girdisi
# -----------------------------
user_input = st.text_input("Sorunuzu veya sınıf ismini yazın:")

if user_input:
    if re.search(r'\d{1,2}[/-]?[A-D]', user_input.upper()):
        ders_programi_goster(user_input)
    else:
        cevap, tablo_df, kaynaklar = okul_asistani_sorgula(user_input)
        st.markdown(cevap)
        if tablo_df is not None:
            st.markdown("📊 **İlgili Madde Tablosu:**")
            st.table(tablo_df)
        if kaynaklar is not None:
            st.markdown("📚 **Kaynaklar:**")
            for k in kaynaklar:
                st.markdown(f"- {k}")

# -----------------------------
# Önceki sohbeti göster
# -----------------------------
for msg in st.session_state.conversation:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])
