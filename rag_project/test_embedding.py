from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Test sentence
text = "This is a test sentence for embedding."

# Convert text to vector
vector = model.encode(text)

# Print results
print("Vector shape:", vector.shape)
print("First 5 values of vector:", vector[:5])
print("Type of vector:", type(vector))
