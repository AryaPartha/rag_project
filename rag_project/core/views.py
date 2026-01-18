from django.shortcuts import render
from sentence_transformers import SentenceTransformer
import chromadb
from django.conf import settings
from google import genai
import os
import time

# --- GLOBAL INITIALIZATION (Performance Optimization) ---
# Load the expensive embedding model only once when the server starts.
# This significantly speeds up subsequent searches.
print("--- Loading embedding model... This will happen only once. ---")
start_time = time.time()
# Using a global variable to store the model instance
GLOBAL_EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
end_time = time.time()
print(f"--- Model loaded successfully in {end_time - start_time:.2f} seconds! ---")


def search_view(request):
    """
    Handles the search functionality.
    GET: Renders the search form.
    POST: Processes the query, performs RAG, and displays results.
    """
    if request.method == "GET":
        # Render the simple search form template
        return render(request, "core/search.html")

    if request.method == "POST":
        # 1. Get the user's query from the submitted form
        query_text = request.POST.get("query")

        # --- RAG STEP 1: RETRIEVAL ---

        # 2. Create embedding for the query using the pre-loaded GLOBAL model
        # This is fast because the model is already in memory.
        query_vector = GLOBAL_EMBEDDING_MODEL.encode(query_text)

        # 3. Connect to ChromaDB
        # Use persistent client to read from the saved database file on disk
        chroma_client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))
        # Get the collection where the documents are stored
        collection = chroma_client.get_or_create_collection(name="documents")

        # 4. Query the database for the top 3 most relevant documents
        # The vector must be converted to a list for ChromaDB uses .tolist()
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=3
        )

        # Extract the list of actual document texts from the results
        documents = results["documents"][0]

        # --- RAG STEP 2: GENERATION ---

        # 5. Combine the retrieved documents into a single context string
        # This forms the knowledge base for the LLM for this specific query.
        context_text = "\n\n".join(documents)

        # 6. Build the prompt for the LLM
        # Instructs the model to answer based *only* on the provided context.
        prompt = f"""
Answer the user's question using ONLY the context below.

Context:
{context_text}

Question:
{query_text}

Answer in a clear and concise paragraph.
"""

        # 7. Initialize the Gemini client
        gemini_client = genai.Client(api_key=os.getenv("API_KEY"))

        # --- DEBUG PRINT START ---
        print(f"DEBUG: The prompt string type is: {type(prompt)}")
        print(f"DEBUG: The contents argument being sent is: {[prompt]}")
        print(f"DEBUG: The type of the contents argument is: {type([prompt])}")
        # --- DEBUG PRINT END ---

        # 8. Call the generate_content method
        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[prompt], # Make sure this has square brackets around prompt
        )
        # 9. Prepare the context data to be passed to the results template
        context = {
            "query": query_text,
            "documents": documents,
            "answer": response.text
        }

        # Render the results page with the data
        return render(request, "core/results.html", context)