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
from tools import *
from langchain.schema import Document  # Document 객체 추가
import time  # 실시간 문자 출력에 사용


# OpenAI API Key 설정
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 메인 함수
def main():
    st.title('🦜🔗Welcome to GPTools 📜')
    st.sidebar.success("👆 Select a page above.")

    # 세션 상태 초기화
    if 'messages' not in st.session_state:
        reset_chat()

    # 파일 업로드
    uploaded_files = st.file_uploader(
        "Upload one or multiple files. Supported formats are **pdf, docx, doc, xlsx, xls, csv**, and **txt**. Your files won't be stored here.",
        accept_multiple_files=True
    )

    # 파일이 업로드되지 않은 경우 세션을 초기화
    if not uploaded_files and st.session_state.get('file_uploaded', False):
        reset_chat()
        st.session_state.file_uploaded = False

    if uploaded_files:
        st.session_state.file_uploaded = True  # 파일 업로드 상태 저장
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
            elif file_extension in [".docx", ".doc"]:
                loader = UnstructuredFileLoader(temp_file_path, file_type="docx")
            elif file_extension == ".csv":
                loader = CSVLoader(temp_file_path)
            elif file_extension in [".xlsx", ".xls"]:
                loader = UnstructuredExcelLoader(temp_file_path)
            elif file_extension == ".txt":
                try:
                    text_content = load_txt_file(temp_file_path)
                    text.append(text_content)
                except Exception as e:
                    st.error(f"Failed to load txt file: {e}")

            if loader:
                try:
                    loaded_documents = loader.load()
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

        # 채팅 인터페이스 구현
        for message in st.session_state.messages:
            if message['role'] == 'user':
                st.chat_message("user").markdown(message['content'])
            else:
                st.chat_message("assistant").markdown(message['content'])

        # 유저의 질문 입력
        user_input = st.chat_input("Ask anything about your files")

        if user_input:
            # 유저 메시지를 세션에 저장
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user").markdown(user_input)

            # GPT 응답 처리
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(
                    model_name="gpt-4o-mini",
                    temperature=0,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                ),
                retriever=retriever,
                return_source_documents=True,
                chain_type="stuff"
            )
            result = qa_chain({"query": user_input})
            assistant_reply = result['result']

            # GPT의 응답을 세션에 저장
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

            # GPT 응답을 한 글자씩 출력 (아이콘과 함께)
            with st.chat_message("assistant"):
                message_container = st.empty()  # 비어있는 컨테이너를 생성
                display_typing_effect(message_container, assistant_reply)

# 메인 함수 실행
if __name__ == '__main__':
    main()