from django.core.management.base import BaseCommand
from sentence_transformers import SentenceTransformer
from django.conf import settings
import chromadb


class Command(BaseCommand):
    help = "Query ChromaDB using a natural language question"

    def add_arguments(self, parser):
        parser.add_argument('query', type=str, help='The question to ask the database')

    def handle(self, *args, **options):
        # 1. Get query text
        query_text = options['query']
        self.stdout.write(f"Query: {query_text}")

        # 2. Load embedding model
        self.stdout.write("Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # 3. Convert query to vector
        self.stdout.write("Generating query embedding...")
        query_vector = model.encode(query_text)

        self.stdout.write(f"Query vector shape: {query_vector.shape}")

        # 4. Connect to ChromaDB (LOCAL DISK)
        self.stdout.write("Connecting to ChromaDB (Persistent)...")
        client = chromadb.PersistentClient(path=str(settings.CHROMA_DB_PATH))

        # 5. Get collection
        collection = client.get_or_create_collection("knowledge_base")

        # 6. Query ChromaDB
        self.stdout.write("Searching in knowledge base...")
        results = collection.query(
            query_embeddings=[query_vector.tolist()],  # IMPORTANT
            n_results=3
        )

        # 7. Print results
        documents = results["documents"][0]
        distances = results["distances"][0]

        self.stdout.write(self.style.SUCCESS(f"\nFound {len(documents)} matches:\n"))

        for i, doc in enumerate(documents):
            print(f"--- Match {i+1} ---")
            print(f"Content:\n{doc}")
            print(f"Distance: {distances[i]}")
            print()
