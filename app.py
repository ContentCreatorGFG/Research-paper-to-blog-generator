from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from huggingface_hub import login
from langchain.prompts import PromptTemplate
import os

# Hardcoded API keys (not recommended for production)
HUGGINGFACEHUB_API_TOKEN = "API_TOKEN"
GROQ_API_KEY = "API_KEY"

# Login to Hugging Face
login(HUGGINGFACEHUB_API_TOKEN)

# Set your GROQ API key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Specify your local PDF file paths here
pdf_paths = ['research_paper_academic.pdf']

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text  # Return the combined text from all PDF pages

# Call the function with the list of PDF paths
raw_text = get_pdf_text(pdf_paths)
print("Extracted text length:", len(raw_text))

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

text_chunks = get_text_chunks(raw_text)
print("Number of text chunks:", len(text_chunks))

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-xl",
        model_kwargs={"device": "cuda"}
    )
    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )
    return vectorstore

vectorstore = get_vectorstore(text_chunks)
print("Vectorstore created")

# Define a prompt template
template = """
You are an expert assistant. Use the following context to answer the question.

Context: {context}
Question: {question}
"""

def get_conversation_chain(vectorstore):
    llm = ChatGroq(
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.7,
        request_timeout=30
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

qa_chain = get_conversation_chain(vectorstore)

query = "Write a blog-style summary of this research paper for non-expert readers."
response = qa_chain.run(query)
print("✅ Blog :\n", response)