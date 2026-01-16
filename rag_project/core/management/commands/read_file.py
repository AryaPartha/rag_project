from django.core.management.base import BaseCommand
from core.models import Document
from pypdf import PdfReader
from core.utils import clean_text, chunk_text
from sentence_transformers import SentenceTransformer
import chromadb
from django.conf import settings


class Command(BaseCommand):
    help = "Read latest uploaded PDF, clean, chunk, embed, and store in ChromaDB"

    def handle(self, *args, **kwargs):

        # 1. Get last valid document
        doc = Document.objects.exclude(file="").last()

        if not doc or not doc.file:
            self.stdout.write("❌ No valid document with file found in database.")
            return

        # 2. Get file path
        file_path = doc.file.path
        self.stdout.write(f"📄 Reading file from: {file_path}")

        # 3. Read PDF
        reader = PdfReader(file_path)

        raw_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"

        if not raw_text.strip():
            self.stdout.write("❌ PDF has no readable text.")
            return

        self.stdout.write(f"📜 Extracted {len(raw_text)} characters of text")

        # 4. Clean text
        cleaned_text = clean_text(raw_text)

        # 5. Chunk text
        chunks = chunk_text(cleaned_text)

        if not chunks:
            self.stdout.write("❌ No chunks were created.")
            return

        self.stdout.write(f"✂️ Total chunks created: {len(chunks)}")

        # 6. Load embedding model
        self.stdout.write("🧠 Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # 7. Generate embeddings
        self.stdout.write("⚡ Generating embeddings...")
        embeddings = model.encode(chunks)

        self.stdout.write(f"📐 Embeddings shape: {embeddings.shape}")

        # 8. Initialize ChromaDB
        self.stdout.write("💾 Initializing ChromaDB...")

        client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DB_PATH)
        )

        collection = client.get_or_create_collection(
            name="documents"
        )

        self.stdout.write(f"📂 ChromaDB path: {settings.CHROMA_DB_PATH}")

        # 9. Store in ChromaDB
        self.stdout.write("📥 Storing embeddings in ChromaDB...")

        ids = [f"chunk_{i}" for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=ids
        )

        self.stdout.write("✅ Successfully stored all chunks in ChromaDB!")
# ChromaDB path setting in rag_project/settings.py