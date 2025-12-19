"""
NeuroQuery: RAG-Based AI Research Assistant
A hybrid search system for querying technical research papers

Architecture:
1. PDF Ingestion → Recursive Chunking
2. Dual Indexing → BM25 (Keyword) + FAISS (Semantic)
3. Hybrid Retrieval → Ensemble with RRF
4. LLM Generation → Llama-3 via Groq (Temperature=0 for factual answers)
"""

import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Groq LLM
from groq import Groq

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Check if Groq API key exists
if not os.getenv("GROQ_API_KEY"):
    st.error(" GROQ_API_KEY not found in .env file!")
    st.stop()

# Chunking strategy: Optimized for technical documents
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap to preserve context across chunks

# Embedding model: Lightweight and efficient for semantic search
EMBEDDING_MODEL = "sentence-transformers/allenai-specter"

# LLM configuration via Groq
LLM_MODEL = "llama-3.3-70b-versatile"  # 70B parameter model with 8K context
LLM_TEMPERATURE = 0  # Deterministic (factual) responses

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_pdf(file_path):
    """
    Load PDF and extract text using PyPDFLoader
    
    Why PyPDFLoader?
    - Handles multi-column layouts correctly
    - Preserves reading order (Column A → Column B)
    - Better than basic string extraction
    
    Args:
        file_path (str): Path to PDF file
    
    Returns:
        list: List of Document objects with page metadata
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        st.success(f" Loaded {len(documents)} pages from PDF")
        return documents
    except Exception as e:
        st.error(f" Error loading PDF: {str(e)}")
        return None


def chunk_documents(documents):
    """
    Split documents into semantically coherent chunks
    
    Why Recursive Splitting?
    - Tries to split at logical boundaries (paragraphs, sentences)
    - Preserves code blocks and mathematical formulas
    - Overlap ensures context isn't lost at chunk boundaries
    
    Separators (in priority order):
    1. Double newline (paragraphs)
    2. Single newline (sentences)
    3. Space (words)
    4. Empty string (characters - last resort)
    
    Args:
        documents (list): List of Document objects
    
    Returns:
        list: List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],  # Logical boundaries
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    st.info(f" Created {len(chunks)} chunks (Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})")
    return chunks


def create_vector_store(chunks):
    """
    Create FAISS vector store for semantic search
    
    FAISS (Facebook AI Similarity Search):
    - Dense vector embeddings capture semantic meaning
    - "Attention mechanism" and "self-attention" are semantically similar
    - Good for conceptual queries
    
    Embedding Model: 
    - 384 dimensions allenai-specter
    - Fast on CPU (important for M4 Mac)
    - Trained on 1B+ sentence pairs
    
    Args:
        chunks (list): List of Document chunks
    
    Returns:
        FAISS: FAISS vector store object
    """
    with st.spinner("Creating embeddings... (This may take 30-60 seconds)"):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # M4 uses CPU for sentence-transformers
            encode_kwargs={'normalize_embeddings': True}  # Cosine similarity optimization
        )
        
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.success("FAISS vector store created!")
        return vectorstore


def create_bm25_retriever(chunks):
    """
    Create BM25 retriever for keyword-based search
    
    BM25 (Best Matching 25):
    - Sparse retrieval based on term frequency
    - Exact match for technical terms (e.g., "ReLU", "BERT")
    - Immune to synonym confusion
    
    Why BM25?
    - "Dropout" (regularization) vs "dropout" (general term) distinction
    - Precise hyperparameter retrieval
    - Complements semantic search
    
    Args:
        chunks (list): List of Document chunks
    
    Returns:
        BM25Retriever: BM25 retriever object
    """
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4  # Retrieve top 4 documents
    st.success("BM25 keyword retriever created!")
    return bm25_retriever


def create_ensemble_retriever(vectorstore, bm25_retriever):
    """
    Combine BM25 and FAISS using Ensemble Retriever
    
    Ensemble Strategy: Reciprocal Rank Fusion (RRF)
    - Takes results from both retrievers
    - Re-ranks using: score = 1 / (rank + k)
    - Merges top results from both methods
    
    Weights:
    - BM25: 0.5 (keyword precision)
    - FAISS: 0.5 (semantic understanding)
    
    Why 50/50?
    - Balanced approach for technical documents
    - Can be tuned based on evaluation
    
    Args:
        vectorstore (FAISS): FAISS vector store
        bm25_retriever (BM25Retriever): BM25 retriever
    
    Returns:
        EnsembleRetriever: Combined retriever
    """
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]  # Equal weighting
    )
    
    st.success("Hybrid ensemble retriever created!")
    return ensemble_retriever


def create_rag_chain(retriever):
    """
    Create RAG chain with Groq LLM
    
    RAG Pipeline:
    1. User question → Retriever finds relevant chunks
    2. Chunks + Question → Prompt Template
    3. Prompt → Groq API (Llama-3)
    4. LLM generates grounded answer
    
    Temperature = 0:
    - Deterministic output
    - Factual extraction (not creative generation)
    - Reduces hallucination
    
    Args:
        retriever (EnsembleRetriever): Hybrid retriever
    
    Returns:
        Custom RAG function
    """
    
    # Define prompt template
    prompt_template = """You are NeuroQuery, an AI assistant specialized in analyzing technical research papers.

Use the following context (retrieved from the research paper) to answer the question accurately.

IMPORTANT RULES:
1. Answer ONLY based on the provided context
2. If the context doesn't contain the answer, say "I cannot find this information in the provided document"
3. Cite specific details from the context
4. Use technical terminology correctly
5. Be concise and precise

Context:
{context}

Question: {question}

Detailed Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    def rag_query(question):
        """
        Execute RAG query
        
        Args:
            question (str): User question
        
        Returns:
            dict: Answer and source documents
        """
        # Retrieve relevant documents
        docs = retriever.get_relevant_documents(question)
        
        # Combine document content as context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format prompt
        formatted_prompt = prompt.format(context=context, question=question)
        
        # Call Groq API
        try:
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a technical research assistant. Provide accurate, well-cited answers based on the given context."
                    },
                    {
                        "role": "user",
                        "content": formatted_prompt
                    }
                ],
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=1024
            )
            
            answer = chat_completion.choices[0].message.content
            
            return {
                "answer": answer,
                "source_documents": docs
            }
        
        except Exception as e:
            st.error(f"Error calling Groq API: {str(e)}")
            return None
    
    return rag_query


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """
    Main Streamlit application
    """
    
    # Page configuration
    st.set_page_config(
        page_title="NeuroQuery - Your Personal AI Research Assistant",
        page_icon="",
        layout="wide"
    )
    
    # Title and description
    st.title("NeuroQuery: RAG-Based AI Research Assistant")
    st.markdown("""
    **Chat with Technical Research Papers** using Hybrid Search (BM25 + Vector Embeddings)
    
     Upload a PDF →  Ask Questions →  Get Accurate Answers with Citations
    """)
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Document Upload")
        
        uploaded_file = st.file_uploader(
            "Upload Research Paper (PDF)",
            type=["pdf"],
            help="Upload technical papers like 'Attention Is All You Need' or 'ResNet'"
        )
        
        st.markdown("---")
        st.markdown("""
        ### 🔧 System Configuration
        - **Chunk Size**: 1000 chars
        - **Overlap**: 200 chars
        - **Embedding**: allenai-specter
        - **LLM**: Llama-3 70B via Groq
        - **Search**: Hybrid (BM25 + FAISS)
        """)
    
    # Initialize session state
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # PDF Processing
    if uploaded_file is not None:
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Process button
        if st.sidebar.button("Process Document", type="primary"):
            
            with st.spinner("Processing document..."):
                
                # Step 1: Load PDF
                st.subheader("Step 1: Loading PDF")
                documents = load_pdf(tmp_path)
                
                if documents is None:
                    st.stop()
                
                # Step 2: Chunk documents
                st.subheader("Step 2: Chunking Document")
                chunks = chunk_documents(documents)
                
                # Step 3: Create vector store
                st.subheader("Step 3: Creating Vector Store (FAISS)")
                vectorstore = create_vector_store(chunks)
                
                # Step 4: Create BM25 retriever
                st.subheader("Step 4: Creating Keyword Retriever (BM25)")
                bm25_retriever = create_bm25_retriever(chunks)
                
                # Step 5: Create ensemble retriever
                st.subheader("Step 5: Creating Hybrid Ensemble Retriever")
                ensemble_retriever = create_ensemble_retriever(vectorstore, bm25_retriever)
                
                # Step 6: Create RAG chain
                st.subheader("Step 6: Creating RAG Chain with Groq LLM")
                st.session_state.rag_chain = create_rag_chain(ensemble_retriever)
                
                st.success("Document processed successfully! You can now ask questions.")
        
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    
    # Chat Interface
    if st.session_state.rag_chain is not None:
        
        st.markdown("---")
        st.subheader("Ask Questions")
        
        # Question input
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the attention mechanism? How does it work?"
        )
        
        # Submit button
        if st.button("🔍 Search & Generate Answer"):
            
            if question:
                
                with st.spinner("Searching and generating answer..."):
                    
                    # Execute RAG query
                    result = st.session_state.rag_chain(question)
                    
                    if result:
                        
                        # Display answer
                        st.markdown("### Answer:")
                        st.markdown(result["answer"], unsafe_allow_html=False)
                        
                        # Display source documents
                        with st.expander("View Source Documents"):
                            for i, doc in enumerate(result["source_documents"]):
                                st.markdown(f"**Source {i+1}:**")
                                st.text(doc.page_content)
                                st.markdown(f"*Metadata: {doc.metadata}*")
                                st.markdown("---")
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": result["answer"]
                        })
            
            else:
                st.warning("Please enter a question")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.subheader("Chat History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q: {chat['question'][:50]}..."):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
    
    else:
        st.info("Upload a PDF document to get started!")


if __name__ == "__main__":
    main()
