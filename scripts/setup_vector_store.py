from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from config.config import Config


def setup_vector_store():
    """One-time setup: Load PDF and create vector store"""
    
    print("📄 Loading PDF...")
    loader = PyPDFLoader(Config.PDF_PATH)
    docs = loader.load()
    print(f"✓ Loaded {len(docs)} pages")
    
    print("\n✂️ Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"✓ Created {len(all_splits)} chunks")
    
    print("\n🔢 Generating embeddings...")
    embed_model = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": Config.DEVICE}
    )
    
    vector_store = FAISS.from_documents(
        documents=all_splits,
        embedding=embed_model
    )
    
    print("\n💾 Saving vector store...")
    vector_store.save_local(Config.VECTOR_STORE_PATH)
    print(f"✓ Saved to {Config.VECTOR_STORE_PATH}")
    
    print("\n✨ Setup complete!")


if __name__ == "__main__":
    setup_vector_store()