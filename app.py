import streamlit as st
import subprocess
import sys
import os
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct  # Add PointStruct here
# Check if google.generativeai is installed, if not install it

def install_packages():
    packages = ['google-generativeai']
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            st.success(f"{package} is already installed.")
        except ImportError:
            st.info(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            st.success(f"{package} has been installed successfully.")

# Install required packages
install_packages()

# Now import the rest of the packages
import PyPDF2
import json
import requests
from typing import Optional, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Import Google Generative AI after installation
import google.generativeai as genai

if "client" not in st.session_state:
    st.session_state.client = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None


def get_all_urls(base_url):
    urls = set()
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for link in soup.find_all("a", href=True):
                url = link["href"]
                full_url = urljoin(base_url, url)
                parsed_url = urlparse(full_url)
                if parsed_url.netloc == urlparse(base_url).netloc:
                    urls.add(
                        parsed_url.scheme + "://" + parsed_url.netloc + parsed_url.path
                    )
    except Exception as e:
        st.error(f"An error occurred while crawling {base_url}: {e}")
    return urls


def extract_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()

            lines = (line.strip() for line in text.splitlines())

            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

            text = " ".join(chunk for chunk in chunks if chunk)

            return text
        else:
            st.warning(
                f"Failed to fetch content from {url}: Status code {response.status_code}"
            )
            return None
    except Exception as e:
        st.warning(f"Error extracting text from {url}: {e}")
        return None


def fetch_url_content(url: str) -> Optional[str]:
    try:
        return extract_text_from_url(url)
    except Exception as e:
        st.error(f"Error: Failed to fetch URL {url}. Exception: {e}")
        return None


def get_gemini_embeddings(texts: List[str], api_key: str):
    try:
        # Configure the API key
        genai.configure(api_key=api_key)
        
        # Generate embeddings
        embeddings = []
        for text in texts:
            # Use the embedding_model to get embeddings
            embedding_result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_query"
            )
            # Access the embedding from the response
            embeddings.append(embedding_result["embedding"])
        return embeddings

    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None




def process_uploaded_pdfs(uploaded_files):
    pdf_list = []
    for uploaded_file in uploaded_files:
        content = ""
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                content += page.extract_text()
            pdf_list.append({"content": content, "filename": uploaded_file.name})
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    return pdf_list


def process_and_index_documents(
    uploaded_files, web_urls=None, chunk_size=150, crawl_website=False
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
    )

    all_chunks = []
    doc_metadata = []

    if uploaded_files:
        all_documents = process_uploaded_pdfs(uploaded_files)
        for doc in all_documents:
            chunks = text_splitter.split_text(doc["content"])
            all_chunks.extend(chunks)
            for _ in chunks:
                doc_metadata.append(
                    {"filename": doc["filename"], "source": "pdf_dataset"}
                )

    if web_urls:
        urls = [url.strip() for url in web_urls.split(",")]

        if crawl_website:
            all_urls = set()
            progress_bar = st.progress(0)
            progress_text = st.empty()

            for i, base_url in enumerate(urls):
                progress_text.text(f"Crawling website: {base_url}")
                site_urls = get_all_urls(base_url)
                all_urls.update(site_urls)
                progress_bar.progress((i + 1) / len(urls))

            urls = list(all_urls)
            progress_text.text(f"Found {len(urls)} unique URLs")

        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, url in enumerate(urls):
            progress_text.text(f"Processing URL {i+1}/{len(urls)}: {url}")
            content = fetch_url_content(url)

            if content is not None:
                chunks = text_splitter.split_text(content)
                all_chunks.extend(chunks)
                for _ in chunks:
                    doc_metadata.append({"url": url, "source": "web_content"})

            progress_bar.progress((i + 1) / len(urls))
            time.sleep(0.5)

        progress_text.empty()
        progress_bar.empty()

    if not all_chunks:
        st.error("No content to process. Please provide valid PDFs or web URLs.")
        return None, None

    api_key = st.session_state.gemini_api_key

    with st.spinner("Generating embeddings with Gemini..."):
        embeddings = get_gemini_embeddings(all_chunks, api_key=api_key)
        if not embeddings:
            return None, None

    client = QdrantClient(":memory:")  # Use in-memory storage
    collection_name = "agent_rag_index"
    VECTOR_SIZE = len(embeddings[0])  # Get dimension from first embedding

    with st.spinner("Creating vector database..."):
        try:
            client.delete_collection(collection_name)
        except:
            # Collection might not exist yet
            pass
            
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        ids = list(range(len(all_chunks)))
        payload = [
            {"content": chunk, "metadata": metadata}
            for chunk, metadata in zip(all_chunks, doc_metadata)
        ]

        # Upload in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            end_idx = min(i + batch_size, len(all_chunks))
            client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                id=ids[j],
                vector=embeddings[j],
                payload=payload[j]
            )
                    for j in range(i, end_idx)
                ]
            )

    st.success(
        f"Indexed {len(all_chunks)} chunks from {len(set(m['source'] for m in doc_metadata))} different sources"
    )
    return client, collection_name


def gemini_check_context_relevance(context, question, api_key):
    """
    Use Gemini to determine if the context can answer the question
    
    Args:
        context: The retrieved context text
        question: The user's question
        api_key: Gemini API key
        
    Returns:
        Boolean indicating if context is relevant
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""
        Your task is to determine if the following context contains information that can answer the question.
        
        Context:
        {context}
        
        Question: {question}
        
        If the context contains information to answer the question, respond with exactly "YES".
        If the context does not contain information to answer the question, respond with exactly "NO".
        Respond with only YES or NO.
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip().upper()
        
        return result == "YES"
    except Exception as e:
        st.error(f"Error checking context relevance: {str(e)}")
        return False


def answer_question_with_gemini(question, context, api_key):
    """
    Use Gemini to answer a question based on provided context
    
    Args:
        question: The question to answer
        context: The context to base the answer on
        api_key: Gemini API key
        
    Returns:
        Gemini's response as a string
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""
        You are an expert in answering questions. Provide answers based **exclusively** on the given context.
        
        **Rules:**
        1. If the question cannot be answered using the context, respond only with: "I don't know."
        2. Do **not** infer, assume, or add information not explicitly provided in the context.
        3. Your answers must be:
        - **Concise**: Avoid unnecessary details.
        - **Informative**: Focus on actionable and precise responses.
        4. Format your response in **Markdown**.
        
        **Context:** {context}
        
        **Question:** {question}
        
        **Answer:**
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Gemini API Error: {str(e)}")
        return "I encountered an error while trying to generate a response. Please check your Gemini API key and try again."


def answer_question(question, client, collection_name, top_k=3):
    if not question.strip():
        st.warning("Please enter a question.")
        return

    api_key = st.session_state.gemini_api_key
    
    def search(text: str):
        # Get embedding for the query
        query_embedding = get_gemini_embeddings([text], api_key=api_key)[0]
        
        # Search the vector database
        return client.search(
            collection_name=collection_name, 
            query_vector=query_embedding, 
            limit=top_k
        )

    def format_docs(docs):
        formatted_chunks = []
        for doc in docs:
            source_info = ""
            if doc.payload["metadata"]["source"] == "pdf_dataset":
                source_info = (
                    f"\nSource: PDF file {doc.payload['metadata']['filename']}"
                )
            else:
                source_info = f"\nSource: Web article {doc.payload['metadata']['url']}"
            formatted_chunks.append(doc.payload["content"] + source_info)
        return "\n\n".join(formatted_chunks)

    with st.spinner("Searching for relevant information..."):
        results = search(question)
        context = format_docs(results)

        is_relevant = gemini_check_context_relevance(context, question, api_key)

        if is_relevant:
            st.info("Found relevant information in the indexed content")
            answer = answer_question_with_gemini(question, context, api_key)
            st.markdown(answer)
        else:
            st.info("Relevant information found in indexed content. Using Gemini for a general answer...")
            
            # Ask Gemini without specific context
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-pro')
            
            general_prompt = f"""
            Please answer the following question:
            {question}
            
            Guidelines:
            - Provide a concise and informative answer
            - When appropriate, include relevant facts and context
            - Format your response in Markdown
            - If you don't know the answer with certainty, acknowledge this
            """
            
            response = model.generate_content(general_prompt)
            st.markdown(response.text)


st.title("RAG System with PDF and Website Crawling Support")

with st.expander("API Settings"):
    gemini_api_key = st.text_input("Enter your Gemini API Key:", type="password", 
                                   help="Required for embeddings and completions")
    if gemini_api_key:
        st.session_state.gemini_api_key = gemini_api_key

uploaded_files = st.file_uploader(
    "Upload PDF files:", accept_multiple_files=True, type=["pdf"]
)

st.subheader("Website Input")
web_urls = st.text_input(
    "Enter website URLs (comma-separated):", placeholder="https://example.com"
)
crawl_website = st.checkbox(
    "Crawl entire website(s)",
    help="Enable this to extract content from all pages of the specified website(s)",
)

if st.button("Process and Index Documents"):
    if not st.session_state.get("gemini_api_key"):
        st.error("Please enter your Gemini API key first.")
    else:
        st.session_state.client, st.session_state.collection_name = (
            process_and_index_documents(
                uploaded_files, web_urls, crawl_website=crawl_website
            )
        )

if st.session_state.client and st.session_state.collection_name:
    question = st.text_input("Ask a question about the documents:")
    if st.button("Get Answer"):
        if not st.session_state.get("gemini_api_key"):
            st.error("Please enter your Gemini API key first.")
        else:
            answer_question(
                question, st.session_state.client, st.session_state.collection_name
            )
elif uploaded_files or web_urls:
    st.warning("Please process and index the documents first.")
else:
    st.info("Upload PDFs or provide web URLs to get started.")

