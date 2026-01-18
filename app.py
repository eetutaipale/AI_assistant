import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import shutil

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Q&A Assistant",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "document_text" not in st.session_state:
    st.session_state.document_text = ""

# Title and description
st.title("📄 PDF Q&A Assistant")
st.markdown("Upload a PDF document and ask questions about its content using AI.")

# Sidebar for API key configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Google Gemini API Key input
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=os.getenv("GOOGLE_API_KEY", ""),
        help="Enter your Google Gemini API key or set it in a .env file"
    )
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Enter your Google Gemini API key
    2. Upload a PDF file
    3. Wait for document processing
    4. Ask questions about the document
    """)

# PDF Upload Section
st.header("📤 Upload PDF Document")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf",
    help="Upload a PDF file to extract and analyze"
)

# Process PDF when uploaded
if uploaded_file is not None:
    if not api_key:
        st.warning("⚠️ Please enter your Google Gemini API key in the sidebar to proceed.")
    else:
        # Extract text from PDF
        with st.spinner("📖 Extracting text from PDF..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    shutil.copyfileobj(uploaded_file, tmp_file)
                    tmp_path = tmp_file.name
                
                # Read PDF
                pdf_reader = PdfReader(tmp_path)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                if not text.strip():
                    st.error("❌ Could not extract text from PDF. The file might be scanned or corrupted.")
                else:
                    st.session_state.document_text = text
                    st.success(f"✅ Successfully extracted {len(text)} characters from PDF")
                    
                    # Show document preview
                    with st.expander("📄 Document Preview", expanded=False):
                        st.text(text[:1000] + "..." if len(text) > 1000 else text)
                    
                    # Process document for RAG
                    with st.spinner("🔄 Processing document for Q&A..."):
                        try:
                            # Split text into chunks
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200,
                                length_function=len
                            )
                            chunks = text_splitter.split_text(text)
                            
                            # Create embeddings and vector store
                            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                            vector_store = Chroma.from_texts(
                                texts=chunks,
                                embedding=embeddings,
                                persist_directory=None  # In-memory for this app
                            )
                            
                            st.session_state.vector_store = vector_store
                            st.session_state.document_processed = True
                            
                            st.success(f"✅ Document processed! Created {len(chunks)} text chunks.")
                            
                        except Exception as e:
                            st.error(f"❌ Error processing document: {str(e)}")
                            
            except Exception as e:
                st.error(f"❌ Error reading PDF: {str(e)}")

# Q&A Section
st.header("💬 Ask Questions")

if st.session_state.document_processed and st.session_state.vector_store:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Question input
    question = st.chat_input("Ask a question about the document...")
    
    if question:
        # Add user question to chat
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Create QA chain with Google Gemini
                    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
                    
                    # Create custom prompt
                    prompt_template = """Use the following pieces of context to answer the question at the end. 
                    If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    Use three sentences maximum and keep the answer concise.

                    Context: {context}

                    Question: {question}

                    Answer:"""
                    
                    PROMPT = PromptTemplate(
                        template=prompt_template,
                        input_variables=["context", "question"]
                    )
                    
                    # Create retrieval QA chain
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=st.session_state.vector_store.as_retriever(
                            search_kwargs={"k": 3}
                        ),
                        chain_type_kwargs={"prompt": PROMPT},
                        return_source_documents=True
                    )
                    
                    # Get answer (handle both old and new LangChain API)
                    try:
                        # Try new API (invoke method)
                        result = qa_chain.invoke({"query": question})
                    except (TypeError, AttributeError):
                        # Fall back to old API (direct call)
                        result = qa_chain({"query": question})
                    
                    answer = result.get("result", result.get("answer", "Sorry, I couldn't generate an answer."))
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show source documents
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(result.get("source_documents", []), 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.markdown("---")
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"❌ Error generating answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
        
elif uploaded_file is None:
    st.info("👆 Please upload a PDF file to get started.")
elif not st.session_state.document_processed:
    st.info("⏳ Please wait for the document to be processed...")
else:
    st.warning("⚠️ Please upload a PDF file and ensure your Google Gemini API key is configured.")
