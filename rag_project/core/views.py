from django.shortcuts import render
from sentence_transformers import SentenceTransformer
import chromadb
from django.conf import settings
import google.generativeai as genai
import os

# Configure Gemini
genai.configure(api_key=os.getenv("AIzaSyDznK4HR5edmUAyXLkT-FA_t2-9dReP4RU"))

# Load Gemini model
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


def search_view(request):
    if request.method == "GET":
        return render(request, "core/search.html")

    if request.method == "POST":
        query_text = request.POST.get("query")

        # 1. Load embedding model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # 2. Create embedding for query
        query_vector = model.encode(query_text)

        # 3. Connect to ChromaDB
        client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
        collection = client.get_or_create_collection(name="documents")

        # 4. Query database
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=3
        )

        documents = results["documents"][0]

        # 5. Combine context
        context_text = "\n\n".join(documents)

        # 6. Build prompt
        prompt = f"""
Answer the user's question using ONLY the context below.

Context:
{context_text}

Question:
{query_text}

Answer in a clear and concise paragraph.
"""

        # 7. Ask Gemini
        response = gemini_model.generate_content(prompt)
        answer = response.text

        # 8. Send to template
        context = {
            "query": query_text,
            "documents": documents,
            "answer": answer
        }

        return render(request, "core/results.html", context)
