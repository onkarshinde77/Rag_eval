import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from dotenv import load_dotenv
load_dotenv()

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=os.environ["NVIDIA_API_KEY"],
    temperature=0.7,
    max_tokens=2048,
)

class Embeddings:
    def __init__(self):
        self.embedding = NVIDIAEmbeddings(
            model="nvidia/nemotron-3-embed-1b",
            api_key=os.environ["EMBEDDING_API_KEY"],
        )

    def invoke(self, query: str) -> list[float]:
        return self.embedding.embed_query(query)

embeddings = Embeddings()

# def ask_llm(query: str) -> str:
#     response = client.invoke(query)
#     return response.content

# # Example usage
# query = "Explain RAG in simple terms."
# answer = ask_llm(query)
# print(answer)