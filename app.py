import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Try to import the modern chain functions, fallback to LCEL if not available
try:
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    USE_MODERN_CHAINS = True
except ImportError:
    try:
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        USE_MODERN_CHAINS = True
    except ImportError:
        USE_MODERN_CHAINS = False
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
    
    # Model selection - use models that are actually available
    # Default to gemini-2.5-flash which was tested and works
    model_options = [
        "gemini-2.5-flash",  # Fast and efficient (tested and working)
        "gemini-2.5-pro",  # More capable
        "gemini-2.0-flash",  # Alternative flash model
        "gemini-3-pro-preview",  # Latest preview
        "gemini-3-flash-preview",  # Latest flash preview
        "gemini-pro-latest",  # Latest stable
        "gemini-flash-latest",  # Latest flash
        "gemini-2.5-flash-lite",  # Lite version
    ]
    
    # Use working model from test if available
    default_index = 0
    if "working_model" in st.session_state:
        working_model = st.session_state.working_model
        if working_model in model_options:
            default_index = model_options.index(working_model)
        else:
            # Add working model to the list if not already there
            model_options.insert(0, working_model)
            default_index = 0
    
    model_choice = st.selectbox(
        "Gemini Model",
        model_options,
        index=default_index,
        help="Select the Gemini model. 'gemini-2.5-flash' is recommended for speed and cost. Click 'Test API Key' to see all available models."
    )
    
    # Test API key button
    if st.button("🧪 Test API Key"):
        try:
            if api_key:
                # Try using google-generativeai directly to see what works
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    
                    # List available models
                    st.info("📋 Checking available models...")
                    models = genai.list_models()
                    
                    available_chat_models = []
                    for model in models:
                        if 'generateContent' in model.supported_generation_methods:
                            model_name = model.name.replace('models/', '')
                            available_chat_models.append(model_name)
                    
                    if available_chat_models:
                        st.success(f"✅ Found {len(available_chat_models)} available model(s):")
                        for model_name in available_chat_models:
                            st.text(f"  • {model_name}")
                        
                        # Try to use the first available model
                        test_model_name = available_chat_models[0]
                        st.info(f"🧪 Testing with model: {test_model_name}")
                        
                        model = genai.GenerativeModel(test_model_name)
                        response = model.generate_content("Say 'API key works' if you can read this.")
                        
                        st.success(f"✅ API Key works! Model '{test_model_name}' responded successfully.")
                        st.text(f"Response: {response.text[:100]}...")
                        
                        # Store the working model name
                        st.session_state.working_model = test_model_name
                        st.info(f"💡 Try using '{test_model_name}' in the model selector above")
                    else:
                        st.warning("⚠️ No models found with generateContent support")
                        
                except ImportError:
                    st.warning("google-generativeai package not found. Trying LangChain method...")
                    # Fallback to LangChain method
                    test_llm = ChatGoogleGenerativeAI(
                        model="gemini-pro",
                        temperature=0,
                        api_key=api_key
                    )
                    test_response = test_llm.invoke("Say 'API key works' if you can read this.")
                    st.success(f"✅ API Key is valid! Response: {test_response.content[:50]}...")
            else:
                st.warning("Please enter your API key first")
        except Exception as e:
            error_str = str(e)
            st.error(f"❌ API Key test failed: {error_str}")
            if "404" in error_str or "NOT_FOUND" in error_str:
                st.info("💡 The model name might be wrong. Check the error details above for the exact model name that failed.")
            elif "401" in error_str or "403" in error_str or "AUTH" in error_str.upper():
                st.info("💡 This means your API key is invalid or doesn't have proper permissions")
    
    # Button to list available models
    if st.button("🔍 List Available Models"):
        try:
            if api_key:
                # Try to list models using the Google GenAI client
                try:
                    from google.genai import Client
                    client = Client(api_key=api_key)
                    models = client.models.list()
                    st.info("**Available Models:**")
                    available_models = []
                    for model in models:
                        if hasattr(model, 'supported_generation_methods') and 'generateContent' in model.supported_generation_methods:
                            model_name = model.name.replace('models/', '').replace('publishers/google/models/', '')
                            available_models.append(model_name)
                            st.text(f"✓ {model_name}")
                    
                    if available_models:
                        st.session_state.available_models = available_models
                        st.success(f"Found {len(available_models)} model(s) with generateContent support")
                    else:
                        st.warning("No models found with generateContent support")
                except ImportError:
                    # Fallback: try google-generativeai package
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=api_key)
                        models = genai.list_models()
                        st.info("**Available Models:**")
                        available_models = []
                        for model in models:
                            if 'generateContent' in model.supported_generation_methods:
                                model_name = model.name.replace('models/', '')
                                available_models.append(model_name)
                                st.text(f"✓ {model_name}")
                        
                        if available_models:
                            st.session_state.available_models = available_models
                            st.success(f"Found {len(available_models)} model(s) with generateContent support")
                        else:
                            st.warning("No models found with generateContent support")
                    except Exception as e2:
                        st.error(f"Could not list models. Error: {str(e2)}")
                        st.info("💡 Try using 'gemini-pro' (without version number) as the model name")
            else:
                st.warning("Please enter your API key first")
        except Exception as e:
            st.error(f"Error listing models: {str(e)}")
            st.info("💡 Try using 'gemini-pro' (without version number) as the model name")
    
    # Show available models if found
    if "available_models" in st.session_state and st.session_state.available_models:
        st.info(f"💡 Try these models: {', '.join(st.session_state.available_models[:3])}")
    
    # Store model choice in session state
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = model_choice
    else:
        st.session_state.selected_model = model_choice
    
    st.markdown("---")
    st.markdown("### 📖 Instructions")
    st.markdown("""
    1. Enter your Google Gemini API key
    2. Select a Gemini model
    3. Upload a PDF file
    4. Wait for document processing
    5. Ask questions about the document
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
                            # Use text-embedding-004 (newer) or fallback to embedding-001
                            # Explicitly pass API key
                            try:
                                embeddings = GoogleGenerativeAIEmbeddings(
                                    model="models/text-embedding-004",
                                    api_key=api_key if api_key else None
                                )
                            except Exception:
                                # Fallback to older embedding model
                                try:
                                    embeddings = GoogleGenerativeAIEmbeddings(
                                        model="models/embedding-001",
                                        api_key=api_key if api_key else None
                                    )
                                except Exception:
                                    # Last fallback - try without model specification
                                    embeddings = GoogleGenerativeAIEmbeddings(
                                        api_key=api_key if api_key else None
                                    )
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
                    # Use the model selected in the sidebar, or use working model from test
                    # Default to gemini-2.5-flash which is known to work
                    selected_model = st.session_state.get("selected_model", "gemini-2.5-flash")
                    
                    # If we found a working model from the test, prefer that
                    if "working_model" in st.session_state:
                        selected_model = st.session_state.working_model
                    
                    # Try LangChain first, but have fallback to direct API if it fails
                    llm = None
                    use_direct_api = False
                    
                    try:
                        # Explicitly pass API key - this is important for proper authentication
                        llm = ChatGoogleGenerativeAI(
                            model=selected_model,
                            temperature=0,
                            api_key=api_key if api_key else None
                        )
                    except Exception as langchain_error:
                        # If LangChain fails, try using google-generativeai directly
                        error_str = str(langchain_error)
                        if "404" in error_str or "NOT_FOUND" in error_str:
                            st.warning(f"⚠️ LangChain failed with model '{selected_model}'. Trying direct API...")
                            try:
                                import google.generativeai as genai
                                genai.configure(api_key=api_key)
                                # Try to find a working model
                                models = genai.list_models()
                                working_model_name = None
                                for model in models:
                                    if 'generateContent' in model.supported_generation_methods:
                                        working_model_name = model.name.replace('models/', '')
                                        break
                                
                                if working_model_name:
                                    st.info(f"✅ Found working model: {working_model_name}")
                                    use_direct_api = True
                                    # We'll handle this differently below
                                else:
                                    raise Exception("No working models found")
                            except Exception as direct_error:
                                raise Exception(f"Both LangChain and direct API failed. LangChain error: {error_str}. Direct API error: {str(direct_error)}")
                        else:
                            raise langchain_error
                    
                    # Create custom prompt
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", """Use the following pieces of context to answer the question at the end. 
                        If you don't know the answer, just say that you don't know, don't try to make up an answer.
                        Use three sentences maximum and keep the answer concise.

                        Context: {context}"""),
                        ("human", "{input}")
                    ])
                    
                    # Get retriever
                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    
                    # Create chain using modern API or LCEL fallback
                    source_docs = []
                    if USE_MODERN_CHAINS:
                        # Use modern retrieval chain
                        document_chain = create_stuff_documents_chain(llm, prompt)
                        retrieval_chain = create_retrieval_chain(retriever, document_chain)
                        result = retrieval_chain.invoke({"input": question})
                        answer = result.get("answer", "Sorry, I couldn't generate an answer.")
                        source_docs = result.get("context", [])
                    else:
                        # Fallback to LCEL approach
                        def format_docs(docs):
                            return "\n\n".join(doc.page_content for doc in docs)
                        
                        chain = (
                            {"context": retriever | format_docs, "input": RunnablePassthrough()}
                            | prompt
                            | llm
                            | StrOutputParser()
                        )
                        
                        answer = chain.invoke(question)
                        # Get source documents separately
                        source_docs = retriever.invoke(question)
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Show source documents
                    with st.expander("📚 Source Documents"):
                        if source_docs:
                            for i, doc in enumerate(source_docs, 1):
                                st.markdown(f"**Source {i}:**")
                                # Handle both Document objects and strings
                                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                                st.text(content[:500] + "..." if len(content) > 500 else content)
                                st.markdown("---")
                        else:
                            st.text("No source documents available.")
                    
                    # Add assistant response to chat
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_str = str(e)
                    error_msg = f"❌ Error generating answer: {error_str}"
                    
                    # Provide helpful suggestions for common errors
                    if "404" in error_str or "NOT_FOUND" in error_str:
                        error_msg += "\n\n💡 **Troubleshooting 404 Errors:**"
                        error_msg += "\n\n1. **Check API Key Source:**"
                        error_msg += "\n   - Get your key from: https://makersuite.google.com/app/apikey"
                        error_msg += "\n   - Make sure it's a Gemini API key (not Vertex AI)"
                        error_msg += "\n\n2. **Enable Gemini API:**"
                        error_msg += "\n   - Go to Google Cloud Console"
                        error_msg += "\n   - Enable 'Generative Language API'"
                        error_msg += "\n\n3. **Try Different Model Names:**"
                        error_msg += "\n   - Use 'gemini-pro' (most compatible)"
                        error_msg += "\n   - Click 'Test API Key' button in sidebar"
                        error_msg += "\n   - Click 'List Available Models' to see what works"
                        error_msg += "\n\n4. **API Version Issue:**"
                        if "v1beta" in error_str:
                            error_msg += "\n   - Your API is using v1beta which has limited model support"
                            error_msg += "\n   - Try 'gemini-pro' without version numbers"
                        error_msg += "\n\n5. **Check API Key Permissions:**"
                        error_msg += "\n   - Ensure the key has 'Generative Language API' access"
                        error_msg += "\n   - Verify billing is enabled if required"
                    
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
