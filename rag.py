from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def okul_asistani_sorgula(soru, vector_db):
    arama_sorgusu = f"{soru} yönetmelik maddesi şartları ve sınırları"
    docs = vector_db.similarity_search(arama_sorgusu, k=5)

    baglam = "\n\n".join([doc.page_content for doc in docs])

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """Sen MEB yönetmeliği uzmanısın.
Cevabını SADECE verilen bağlama dayandır."""
            },
            {
                "role": "user",
                "content": f"Bağlam: {baglam}\n\nSoru: {soru}"
            }
        ],
        model="openai/gpt-oss-120b",
        temperature=0
    )

    return chat_completion.choices[0].message.content
