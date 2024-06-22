from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Access the API keys
google_api_key = os.getenv('GOOGLE_API_KEY')

# Initialize Streamlit app title and description
st.title("DocQuery AI")
st.write("Instant answers from multiple PDFs. Transforming document research with AI. Unlock insights effortlessly.")

# File uploader for PDF files
uploaded_files = st.file_uploader("Upload your PDF files", type="pdf", accept_multiple_files=True)

# Initialize session state variables
if "processed" not in st.session_state:
    st.session_state["processed"] = False

# Function to extract text from PDF file
def extract_text_from_PDF(pdf_file):
    reader = PdfReader(pdf_file)
    extracted_text = ""
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        extracted_text += page.extract_text() + "\n"  # Extract text from page
    return extracted_text

# Function to chunk text into manageable parts
def chunk_text(text, chunk_size=400):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Process files if uploaded and not already processed
if uploaded_files and not st.session_state["processed"]:
    with st.spinner("Processing Files...."):
        text_from_pdfs = ""
        for file in uploaded_files:
            text = extract_text_from_PDF(file)
            if text:
                text_from_pdfs += text
            else:
                st.error(f"Failed to extract text from {file.name}")

        pdf_chunks = chunk_text(text_from_pdfs)

        # Initialize embeddings with Google Generative AI
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Store embeddings in FAISS
        if pdf_chunks:
            library = FAISS.from_texts(pdf_chunks, embedding_function)
            st.session_state["library"] = library
            st.session_state["processed"] = True
            st.success("Processed", icon="✅")

# Create a prompt template for user queries
template = """
Question: {question}

Context for answering question: {context}

Instructions:
1. Read the question carefully and use the information provided in the context to generate a relevant answer.
2. Ensure that your answer addresses the question asked by the user.
3. If the question is not related to the provided context, respond with "The question is not related to the provided context."
"""

# Create a PromptTemplate object
prompt_template = PromptTemplate(input_variables=["question", "context"], template=template)

# Initialize the ChatGoogleGenerativeAI instance
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)

# Display user query input and answer generation button if files are processed
if st.session_state["processed"]:
    user_query = st.text_input(
        "Ask any question about the PDF file",
        placeholder="How many people were given free ration?"
    )
    if st.button("Get Answer"):
        with st.spinner("Generating Answer...."):
            retrieved_chunks = st.session_state["library"].similarity_search(query=user_query, k=5)

            context = ""
            for chunk in retrieved_chunks:
                context += chunk.page_content

            prompt = prompt_template.format(question=user_query, context=context)
            result = llm.invoke(prompt)
            print("Prompt = ", prompt)

            st.write("Question: ", user_query)
            st.write("Answer:")
            st.write(result.content)
