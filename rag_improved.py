# -*- coding: utf-8 -*-
"""
Created on Wed May 21 23:02:17 2025

@author: AKC
"""

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama

import tempfile
import os
from datetime import datetime
import pandas as pd
from fpdf import FPDF


# Page configuration
st.set_page_config(page_title="RAG Document Chat", layout="wide")
st.title("Document RAG Chat Assistant")


# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "chain" not in st.session_state:
    st.session_state.chain = None


def process_documents(uploaded_files):
    """
    Process uploaded document files and create text chunks for retrieval.
    
    Parameters:
        uploaded_files (list): List of uploaded file objects from Streamlit
        
    Returns:
        list: List of document chunks processed and split for retrieval
        
    Raises:
        Exception: If document processing fails
    """
    documents = []
    temp_dir = tempfile.mkdtemp()
    
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Save the uploaded file to a temporary location
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load the file based on its extension
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_path)
                documents.extend(loader.load())
            elif file_extension in [".txt", ".md"]:
                loader = TextLoader(temp_path)
                documents.extend(loader.load())
            else:
                st.warning(f"Unsupported file type: {file_extension}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    document_chunks = text_splitter.split_documents(documents)
    
    return document_chunks


def create_vectorstore(document_chunks):
    """
    Create a vectorstore from document chunks.
    
    Parameters:
        document_chunks (list): List of document chunks to embed
        
    Returns:
        Chroma: Vector database with embedded documents
        
    Raises:
        Exception: If vectorstore creation fails
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=document_chunks, 
        embedding=embeddings
    )
    return vectorstore


def setup_rag_chain(vectorstore):
    """
    Set up the RAG chain with the language model and vectorstore.
    
    Parameters:
        vectorstore (Chroma): Vector database with embedded documents
        
    Returns:
        ConversationalRetrievalChain: Chain object for question answering
        
    Raises:
        Exception: If RAG chain setup fails
    """
    try:
        llm = Ollama(model="mistral", base_url="http://localhost:11434")
        
        # Create the conversational retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        return chain
    except Exception as e:
        st.error(f"Error setting up RAG chain: {str(e)}")
        return None


def get_rag_response(query, chain):
    """
    Get response from the RAG chain.
    
    Parameters:
        query (str): User question to process
        chain (ConversationalRetrievalChain): RAG chain for question answering
        
    Returns:
        tuple: (answer string, list of source documents)
        
    Raises:
        Exception: If response generation fails
    """
    # Extract chat history in the format the chain expects
    chat_history = [(msg["content"], msg["response"]) 
                   for msg in st.session_state.messages 
                   if "response" in msg]
    
    # Get response
    result = chain({"question": query, "chat_history": chat_history})
    
    return result["answer"], result["source_documents"]


def save_chat_history(format_type="txt"):
    """
    Save the current chat conversation to a file.
    
    Parameters:
        format_type (str): File format to save ("txt", "pdf", or "xlsx")
        
    Returns:
        str: Path to the saved file or error message
    """
    if not st.session_state.messages:
        return "No conversation to save"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}"
    
    try:
        if format_type == "txt":
            # Save as TXT file
            with open(f"{filename}.txt", "w") as f:
                for msg in st.session_state.messages:
                    f.write(f"{msg['role'].upper()}: {msg['content']}\n\n")
            return f"{filename}.txt"
        
        elif format_type == "xlsx":
            # Save as XLSX file
            data = {
                "Role": [msg["role"] for msg in st.session_state.messages],
                "Content": [msg["content"] for msg in st.session_state.messages]
            }
            df = pd.DataFrame(data)
            df.to_excel(f"{filename}.xlsx", index=False)
            return f"{filename}.xlsx"
        
        elif format_type == "pdf":
            # Save as PDF file
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            for msg in st.session_state.messages:
                role = msg["role"].upper()
                # Handle long content by splitting it into multiple lines
                content = msg["content"]
                pdf.multi_cell(0, 10, txt=f"{role}: {content}")
                pdf.ln(5)  # Add spacing between messages
            
            pdf.output(f"{filename}.pdf")
            return f"{filename}.pdf"
    
    except Exception as e:
        return f"Error saving file: {str(e)}"


# Sidebar for document upload and conversation saving
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files to upload (PDF, TXT)",
        accept_multiple_files=True,
        type=["pdf", "txt", "md"]
    )
    
    process_btn = st.button("Process Documents")
    if uploaded_files and process_btn:
        with st.spinner("Processing documents..."):
            # Process documents
            document_chunks = process_documents(uploaded_files)
            st.session_state.documents = document_chunks
            
            if document_chunks:
                # Create vectorstore
                vectorstore = create_vectorstore(document_chunks)
                st.session_state.vectorstore = vectorstore
                
                # Setup RAG chain
                st.session_state.chain = setup_rag_chain(vectorstore)
                
                st.success(f"Processed {len(uploaded_files)} documents into {len(document_chunks)} chunks")
            else:
                st.error("No document content was extracted. Please check your files.")
    
    # Display number of documents processed
    if st.session_state.documents:
        st.write(f"Documents processed: {len(st.session_state.documents)} chunks")
    
    # Save conversation section
    st.header("Save Conversation")
    col1, col2 = st.columns(2)
    with col1:
        file_format = st.selectbox("Format", ["txt", "xlsx", "pdf"])
    with col2:
        if st.button("Save Chat"):
            result = save_chat_history(file_format)
            if result.startswith("Error") or result.startswith("No conversation"):
                st.error(result)
            else:
                st.success(f"Saved to {result}")
                # Create download button for the file
                with open(result, "rb") as file:
                    st.download_button(
                        label="Download File",
                        data=file,
                        file_name=os.path.basename(result),
                        mime="application/octet-stream"
                    )


# Main interface with two columns
col1, col2 = st.columns([2, 1])

with col1:
    # Chat interface
    st.subheader("Chat with your documents")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Accept user input if documents are loaded
    if st.session_state.vectorstore is not None:
        user_query = st.chat_input("Ask a question about your documents")
        
        if user_query:
            # Display user message
            with st.chat_message("user"):
                st.write(user_query)
            
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, source_docs = get_rag_response(user_query, st.session_state.chain)
                    st.write(response)
                    
            # Save assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "response": response,
                "source_docs": source_docs
            })
    else:
        st.info("Please upload and process documents to start chatting.")

with col2:
    # Display source documents for the most recent response
    st.subheader("Sources")
    if st.session_state.messages and len(st.session_state.messages) > 0:
        last_message = st.session_state.messages[-1]
        if "source_docs" in last_message and last_message["source_docs"]:
            for i, doc in enumerate(last_message["source_docs"]):
                with st.expander(f"Source {i+1}"):
                    st.write(doc.page_content)
                    st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
        elif st.session_state.messages[-1]["role"] == "assistant":
            st.write("No specific sources for this response.")
