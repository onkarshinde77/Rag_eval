import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
load_dotenv()

client = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    api_key=os.environ["NVIDIA_API_KEY"],
    temperature=0.7,
    max_tokens=2048,
)

def ask_llm(query: str) -> str:
    response = client.invoke(query)
    return response.content

# Example usage
query = "Explain RAG in simple terms."
answer = ask_llm(query)
print(answer)