#dotenv is a Python library used for loading environment variables from a .env file into the environment.
#streamlit is a popular Python library used for creating web applications with minimal effort.
#PyPDF2 is a library for reading and manipulating PDF files in Python.
#langchain is a library for natural language processing tasks, and CharacterTextSplitter is a class used for splitting text into chunks.
#OpenAIEmbeddings is a class used for generating word embeddings using OpenAI's language model.
#FAISS is a library for efficient similarity search and clustering of dense vectors.
#This line imports the load_qa_chain function from the langchain.chains.question_answering module.

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def main():
    load_dotenv('.env_temp')
    st.set_page_config(page_title="Chat PDF")
    st.header("Travel Bot💬")
    
    # creates a file uploader widget in the Streamlit app
    pdf = st.file_uploader("Upload your PDF file", type="pdf")
    
    # extracts the text content from each page of the PDF file and appends it to the text variable.
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # split into chunks
        char_text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        text_chunks = char_text_splitter.split_text(text)
      
        # create embeddings
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(text_chunks, embeddings) 
        llm = OpenAI() 
        chain = load_qa_chain(llm, chain_type="stuff")
      
        # show user input
        num_questions = st.number_input("How many questions do you want to ask?", min_value=1, max_value=10, step=1)
        
        for i in range(num_questions):
            st.subheader(f"Question {i+1}")
            query = st.text_input(f"Type your question {i+1}:")
            if query:
                docs = docsearch.similarity_search(query)
                response = chain.run(input_documents=docs, question=query)
                st.write(response)
    

if __name__ == '__main__':
    main()
