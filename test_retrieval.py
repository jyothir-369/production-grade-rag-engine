
from dotenv import load_dotenv

load_dotenv(override=True)

from data_loader import embed_texts
from vector_db import get_qdrant_storage


query = "What is Jyothir currently working as?"

print("=" * 80)
print(f"Query: {query}")
print("=" * 80)

# Create query embedding
vector = embed_texts([query])[0]

print(f"Query embedding dimension: {len(vector)}")

# Search Qdrant
qdrant = get_qdrant_storage()

results = qdrant.search(
    vector,
    top_k=3,
)

print("\nSources:")
print(results["sources"])

print("\nRetrieved Records:")

for i, record in enumerate(results["records"], start=1):
    print(f"\n--- Result {i} ---")
    print(f"Score: {record.get('score')}")
    print(f"Source: {record.get('source')}")
    print(record.get("text"))