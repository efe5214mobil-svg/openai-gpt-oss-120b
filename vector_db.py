# rag.py
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def okul_asistani_sorgula(soru, vector_db):
    # Vektör DB’den en yakın 3 dokümanı al
    docs = vector_db.similarity_search(soru, k=3)

    # Dokümanları bağlam olarak birleştir
    baglam = "\n\n".join([doc.page_content for doc in docs])

    # AI çağrısı
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Cevabını sadece verilen bağlama göre ver."
            },
            {
                "role": "user",
                "content": f"Bağlam:\n{baglam}\n\nSoru: {soru}"
            }
        ],
        model="openai/gpt-oss-120b",
        temperature=0,
    )

    cevap = chat_completion.choices[0].message.content

    # Kaynakları döndür
    kaynaklar = [doc.page_content[:200] for doc in docs]

    return cevap, kaynaklar
