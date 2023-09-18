from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import streamlit as st
import tempfile
from langchain.callbacks.base import BaseCallbackHandler
import os
import openai

# Streamlit 웹앱에서 Vector DB를 사용하기 위함. 로컬환경에서 사용할 시 비활성화.
# This code is for using Vector DB in Streamlit web app. Inactivate it when you want in a local environment.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
###########################################################################

openai.api_key = st.secrets["api_key"]
# 글자 하나씩 실시간으로 띄우기
# Printing letters one by one in real time
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token
        self.container.markdown(self.text)

def main():
    st.title('🦜🔗Welcome to GPTools :scroll:')
    st.sidebar.success("👆 Select a page above.")
    
    uploaded_files = st.file_uploader("Upload one or multiple files. Supported formats are **pdf, docx, doc** and **txt**. Your files don't be stored here.", accept_multiple_files=True)
    if uploaded_files:
        text = []
        # Uploaded_file을 객체로 반환. 하지만 경로를 문자열이나 바이트 형태로 제공하지 않기 때문에
        # PyPDFLoader와 같은 몇몇 라이브러리에선 오류. 직접적으로 바이트 데이터를 처리하는 기능이 없기때문이다.
        # 따라서 임시경로를 생성하고 해당 경로를 각각Loader들로 전달한다.
    
        # Returns Uploaded_file as an object, but errors in some libraries, such as PyPDFLoader, because it does not provide paths in string or byte form.
        # This is because there is no function to directly process byte data. Therefore, a temporary file is created and the corresponding path is delivered to the loaders respectively.
    
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
            # loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)
    
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap = 100)
        texts = text_splitter.split_documents(text)
    
        # Database, Embedding
        with st.spinner("Processing"):
            # persist_directory = 'db' # Save to directory named 'db' (if you want to use in local env)
            persist_directory = tempfile.mkdtemp() # temporary directory
            embedding = OpenAIEmbeddings(openai_api_key=st.secrets["api_key"])
            vectordb = Chroma.from_documents(
                documents=texts,
                embedding = embedding, # Which embedding
                persist_directory = persist_directory)
        
        # Initialization
        vectordb.persist()
        vectordb = None
    
        vectordb = Chroma(
            persist_directory = persist_directory,
            embedding_function = embedding
        )
        retriever = vectordb.as_retriever()
    
        with st.form("form", clear_on_submit=True):
            st.header("Chat with GPT 😄")
            question = st.text_input("Ask anything about your files", placeholder="Enter to submit")
            submit = st.form_submit_button("Submit")
            if submit and question:
                with st.spinner("Wait for it..."):
                    chat_box = st.empty()
                    stream_hander = StreamHandler(chat_box)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True, callbacks=[stream_hander], openai_api_key=st.secrets["api_key"]),
                        retriever=retriever,
                        return_source_documents = True, 
                        chain_type = "stuff")
                    qa_chain({"query": question})
                    
if __name__ == '__main__' :
    main()
