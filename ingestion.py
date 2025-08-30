from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

if __name__ == "__main__":
    print("Ingestion started...")
    loader = TextLoader("medium_blog.txt")
    document = loader.load()

    print("Splitting document into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(document)
    print(f"Created {len(chunks)} chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    PineconeVectorStore.from_documents(
        chunks, embeddings, index_name="medium-blog-embeddings-index"
    )

    print("Ingestion completed.")
