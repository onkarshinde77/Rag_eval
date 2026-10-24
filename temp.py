import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

load_dotenv()

embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b",
    api_key=os.environ["EMBEDDING_API_KEY"],
)

text = "I am Onkar"

ans = embeddings.embed_query(text)

print("=" * 60)
print("Embedding:", ans)
print("=" * 60)
print("Number of dimensions:", len(ans))