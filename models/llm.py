from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

llama_instruct = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
mistral_instruct = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")

vecstore_retriever = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")