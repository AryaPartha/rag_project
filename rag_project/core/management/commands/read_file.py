from django.core.management.base import BaseCommand
from core.models import Document
from pypdf import PdfReader
from core.utils import clean_text, chunk_text
from sentence_transformers import SentenceTransformer


class Command(BaseCommand):
    help = "Read latest uploaded PDF, clean, chunk, and embed"

    def handle(self, *args, **kwargs):
        # Get last valid document
        doc = Document.objects.exclude(file="").last()

        if not doc or not doc.file:
            self.stdout.write("No valid document with file found in database.")
            return

        #  Get file path FIRST
        file_path = doc.file.path
        self.stdout.write(f"Reading file from: {file_path}")

        #  Read PDF
        reader = PdfReader(file_path)

        raw_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"

        #  Clean text
        cleaned_text = clean_text(raw_text)

        #  Chunk text
        chunks = chunk_text(cleaned_text)

        self.stdout.write(f"Total chunks created: {len(chunks)}")

        # Load embedding model
        self.stdout.write("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        #  Generate embeddings (batch)
        self.stdout.write("Generating embeddings...")
        embeddings = model.encode(chunks)

        #  Print results
        self.stdout.write(f"Embeddings shape: {embeddings.shape}")
        self.stdout.write("First embedding vector:")
        self.stdout.write(str(embeddings[0]))
