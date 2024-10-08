##########################################################
## 세션 기억 구현하기 ##
##########################################################

import streamlit as st
from langchain_community.document_loaders import ( 
    PyPDFLoader, UnstructuredFileLoader, CSVLoader, UnstructuredExcelLoader 
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
import os
import tempfile
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import Document  # Document 객체 추가
import chardet

# OpenAI API Key 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# openai.api_key = st.secrets["api_key"]

# 실시간으로 글자 출력하는 핸들러
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# 메인 함수
def main():
    st.title('🦜🔗Welcome to GPTools 📜')
    st.sidebar.success("👆 Select a page above.")

    uploaded_files = st.file_uploader(
        "Upload one or multiple files. Supported formats are **pdf, docx, doc**, **csv**, and **txt**.",
        accept_multiple_files=True
    )

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None

            # 파일 확장자에 따른 로더 선택
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
                
#####################################################################################
# pdf 이미지, 테이블 등등 ocr 같은거 필요함.
#####################################################################################
                
                
            elif file_extension in [".docx", ".doc"]:
                loader = UnstructuredFileLoader(temp_file_path, file_type="docx")
            elif file_extension == ".csv":
                loader = CSVLoader(temp_file_path)
            elif file_extension in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(temp_file_path)
            elif file_extension == ".txt":
                # 텍스트 파일의 인코딩 자동 감지 후 읽기
                with open(temp_file_path, 'rb') as f:
                    raw_data = f.read()
                    encoding = chardet.detect(raw_data)['encoding']

                # 감지된 인코딩으로 파일 읽기
                try:
                    with open(temp_file_path, 'r', encoding=encoding) as f:
                        text.append(f.read())
                except Exception as e:
                    st.error(f"Failed to load txt file: {e}")

            if loader:
                try:
                    loaded_documents = loader.load()
                    # 각 문서의 page_content를 text 리스트에 추가
                    text.extend(doc.page_content for doc in loaded_documents)
                except Exception as e:
                    st.error(f"Failed to load file: {e}")

            os.remove(temp_file_path)

        # Document 객체로 변환
        documents = [Document(page_content=txt) for txt in text]

        # 텍스트 분할 처리
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # 임베딩 및 벡터 저장소 처리
        with st.spinner("Processing"):
            persist_directory = tempfile.mkdtemp()  # 임시 디렉토리 생성
            embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
            vectordb = Chroma.from_documents(
                documents=texts,
                embedding=embedding,
                persist_directory=persist_directory
            )

        # 벡터 DB 저장
        vectordb.persist()
        vectordb = None

        # 벡터 DB 초기화
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )

        retriever = vectordb.as_retriever()

        # 질문 폼 생성
        with st.form("form", clear_on_submit=True):
            st.header("Chat with GPT 😄")
            question = st.text_input("Ask anything about your files", placeholder="Enter to submit")
            submit = st.form_submit_button("Submit")

            if submit and question:
                with st.spinner("Wait for it..."):
                    chat_box = st.empty()
                    stream_handler = StreamHandler(chat_box)
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=ChatOpenAI(
                            model_name="gpt-4o-mini",
                            temperature=0,
                            streaming=True,
                            callbacks=[stream_handler],
                            openai_api_key=os.getenv("OPENAI_API_KEY") # # openai.api_key = st.secrets["api_key"]
                        ),
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type="stuff"
                    )
                    qa_chain({"query": question})

# 메인 함수 실행
if __name__ == '__main__':
    main()