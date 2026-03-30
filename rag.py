# rag.py
from groq import Groq
import os

# API key kontrolü
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY bulunamadı! .env veya st.secrets kontrol edin.")

client = Groq(api_key=api_key)

def okul_asistani_sorgula(soru, vector_db):
    # Vektör DB’den en yakın 3 dokümanı al
    docs = vector_db.similarity_search(soru, k=3)

    if not docs:
        return "Veri bulunamadı.", []

    # Bağlam boyutunu güvenli sınıra çek (her dokümanın ilk 500 karakteri)
    baglam = "\n\n".join([doc.page_content[:500] for doc in docs])

    # Mesajlar
    messages = [
        {"role": "system", "content": "Cevabını sadece verilen bağlama göre ver."},
        {"role": "user", "content": f"Bağlam:\n{baglam}\n\nSoru: {soru}"}
    ]

    # Groq API çağrısını try/except ile güvenli hale getir
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="openai/gpt-oss-13b",  # Daha güvenli, büyük model yerine
            temperature=0,
            max_tokens=500
        )
        cevap = chat_completion.choices[0].message.content
    except Exception as e:
        return f"Groq API çağrısı başarısız: {e}", []

    # Kaynakları döndür
    kaynaklar = [doc.page_content[:200] for doc in docs]

    return cevap, kaynaklar
