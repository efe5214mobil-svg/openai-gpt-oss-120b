# rag.py
from groq import Groq
import os
import re
import pandas as pd

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🔄 Tekrarlanan sayıları temizleme
def temizle_cevap(cevap):
    cevap = re.sub(r'(\b\d{2,4}\b)(?:/\s*\1)+', r'\1', cevap)
    cevap = re.sub(r'\s{2,}', ' ', cevap)
    return cevap

# ❌ Soru filtreleme
def soru_filtrele(soru):
    yasak_kelimeler = ["küfür", "hakaret", "orospu", "piç", "siyaset", "din", "ırk", "cinsiyet", "eşcinsel", "terör", "politik"]
    for kelime in yasak_kelimeler:
        if kelime in soru.lower():
            return False
    if len(soru.split()) < 3:
        return False
    return True

# 🤖 RAG fonksiyonu
def okul_asistani_sorgula(soru, vector_db):
    if not soru_filtrele(soru):
        return ("❌ Sorunuz uygun değil veya MEB yönetmeliği ile alakasız.\n"
                "Örnek sorular:\n- Öğrencilerin devamsızlık sınırı nedir?\n"
                "- Okuldan uzaklaştırma kararları hangi maddelerde geçer?\n"
                "- Mazeretli devamsızlık nasıl belgelenir?"), pd.DataFrame()

    docs = vector_db.similarity_search(soru, k=3)
    if not docs:
        return ("❌ İlgili veri bulunamadı.\n"
                "Örnek sorular:\n- Öğrencilerin devamsızlık sınırı nedir?\n"
                "- Okuldan uzaklaştırma kararları hangi maddelerde geçer?\n"
                "- Mazeretli devamsızlık nasıl belgelenir?"), pd.DataFrame()

    baglam = "\n\n".join([doc.page_content for doc in docs])

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """
Sen MEB yönetmeliği uzmanısın.
Kurallar:
- Sadece verilen bağlama göre cevap ver
- Evet/Hayır şeklinde cevap ver
- İlgili maddeleri tablo halinde ver
- Kaynak olarak dokümanı da belirt
- Cevapta küfür, hakaret, siyaset, din, ırk, cinsiyet ile ilgili içerik olamaz
- Cevap sadece resmi MEB yönetmeliği ile ilgili olmalı
- Anlamsız tekrarlar ve saçma ifadeler kullanma
"""
            },
            {
                "role": "user",
                "content": f"Bağlam:\n{baglam}\n\nSoru: {soru}"
            }
        ],
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=500
    )

    cevap = chat_completion.choices[0].message.content
    cevap = temizle_cevap(cevap)

    # Kaynaklar tablo şeklinde
    df = pd.DataFrame({
        "İlgili Madde": [doc.page_content[:50]+"..." for doc in docs],
        "Kaynak": [doc.page_content[:200] for doc in docs]
    })

    return cevap, df
