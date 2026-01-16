from django.core.management.base import BaseCommand
from sentence_transformers import SentenceTransformer


class Command(BaseCommand):
    help = "Convert a text query into an embedding vector"

    def add_arguments(self, parser):
        parser.add_argument(
            "query",
            type=str,
            help="The question to ask the database"
        )

    def handle(self, *args, **options):
        # 1. Get query text from terminal
        query_text = options["query"]

        self.stdout.write(f"Received query: {query_text}")

        # 2. Load embedding model
        self.stdout.write("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # 3. Encode query into vector
        self.stdout.write("Encoding query...")
        vector = model.encode(query_text)

        # 4. Print result info
        self.stdout.write(
            f"Query processed. Vector shape: {vector.shape}, Type: {type(vector)}"
        )

        # Optional: print first 10 numbers to confirm
        self.stdout.write(f"First 10 values of vector: {vector[:10]}")
