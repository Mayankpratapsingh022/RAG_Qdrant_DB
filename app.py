from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

PDF_FILES = [
    "Attention_is_all_you_need.pdf",
    "BERT.pdf",
    "Denosing_diffusion.pdf",
    "Neural_Machine_Translation.pdf",
    "Neural_Turing.pdf"
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Each chunk is 1000 characters max
    chunk_overlap=200    # Each chunk overlaps with the previous one by 200 characters
)

all_chunks = []



for file_name in PDF_FILES:
    pdf_path = Path(__file__).parent / "PDFs" / file_name
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    split_docs = text_splitter.split_documents(docs)
    all_chunks.extend(split_docs)


embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key="your-api-key"  # Replace with your key
)


# vector_store = QdrantVectorStore.from_documents(
#     documents=all_chunks,
#     url="http://localhost:6333",
#     collection_name="learning_langchain",
#     embedding=embedder
# )

# print("Documents injected into Qdrant collection.")

retriever = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_langchain",
    embedding=embedder
)

query = "What is Attention?"
search_result = retriever.similarity_search(query=query)

for i, chunk in enumerate(search_result, 1):
    print(f"\n--- Chunk {i} ---\n{chunk.page_content}")


