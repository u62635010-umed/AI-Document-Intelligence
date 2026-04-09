import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables (Groq_API_KEY)
load_dotenv()

def process_pdf(pdf_path):
    # 1. Load and Parse PDF
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Check if text was extracted (PyPDF sometimes returns empty text for scanned PDFs)
        total_text = sum(len(doc.page_content.strip()) for doc in documents)
        if total_text < 100:
            raise ValueError("Empty or scanned PDF detected, switching to OCR loader")
    except Exception:
        # Fallback to OCR logic
        loader = UnstructuredPDFLoader(
            pdf_path,
            mode="elements",
            strategy="hi_res"
        )
        documents = loader.load()

    # 2. Chunking
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    chunks = recursive_splitter.split_documents(documents)

    # 3. Vector Database
    # Use HuggingFace Embeddings (Free, No API Key needed)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Use in-memory Chroma database to avoid file locking issues on Windows
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    # 4. Retrieval Setup
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4
    
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6]
    )

    # 5. Initialize Groq Chain
    # Note: Ensure Groq_API_KEY is in .env as the user specified
    groq_api_key = os.getenv("Groq_API_KEY")
    if not groq_api_key:
        raise ValueError("Groq_API_KEY not found in .env file")
        
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0,
        groq_api_key=groq_api_key
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=hybrid_retriever,
        return_source_documents=True
    )
    
    return qa_chain

def get_answer(qa_chain, query):
    result = qa_chain.invoke({"query": query})
    return result["result"], result["source_documents"]
