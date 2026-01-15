from django.core.management.base import BaseCommand
from core.models import Document
from pypdf import PdfReader
from core.utils import clean_text, chunk_text
from sentence_transformers import SentenceTransformer


class Command(BaseCommand):
    help = "Read latest uploaded PDF, clean, chunk, and embed it"

    def handle(self, *args, **kwargs):
        # 1. Get latest document that actually has a file
        doc = Document.objects.exclude(file="").last()

        if not doc or not doc.file:
            self.stdout.write("No valid document with file found in database.")
            return

        # 2. Get file path
        file_path = doc.file.path
        self.stdout.write(f"Reading file from: {file_path}")

        # 3. Read PDF
        reader = PdfReader(file_path)

        raw_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"

        if not raw_text.strip():
            self.stdout.write("PDF has no readable text.")
            return

        # 4. Clean text
        cleaned_text = clean_text(raw_text)

        # 5. Chunk text
        chunks = chunk_text(cleaned_text)

        self.stdout.write(f"Total chunks created: {len(chunks)}")

        if len(chunks) == 0:
            self.stdout.write("No chunks were created.")
            return

        # 6. Load embedding model
        self.stdout.write("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # 7. Generate embeddings (batch, NOT loop)
        self.stdout.write("Generating embeddings...")
        embeddings = model.encode(chunks)

        # 8. Print result info
        self.stdout.write(f"Embeddings shape: {embeddings.shape}")
        self.stdout.write(f"First embedding vector:\n{embeddings[0]}")
