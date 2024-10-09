import streamlit as st
import tempfile
import re
import pandas as pd
from langchain_community.document_loaders import TextLoader

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_chat import message

from tools import *

def main():

    st.title('🦜🔗Make your own Chat Bot :tongue:')
    st.sidebar.success("👆 Select a page above.")
    
    uploaded_file = st.file_uploader("Upload your **Kakao Talk Message file.txt	📃** Your file don't be stored here.")
    if uploaded_file:
        text = []
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        # loader = TextLoader(temp_file_path)
        loader = TextLoader(temp_file_path, encoding = 'utf-8')
        
        all_text_list = loader.load()
        for i in range(len(all_text_list)):
            text_lines = all_text_list[i].page_content.split('\n')[3:]
            all_text_list[i].page_content = '\n'.join(text_lines)
    
        names = []
        times = []
        texts = []
    
        for line in text_lines:
            # \[(.*?)\]: 대괄호([]) 안의 어떤 문자든지(0개 이상)를 비탐욕적(non-greedy)으로 매칭
            # (.*): 어떤 문자든지(0개 이상)를 탐욕적(greedy)으로 매칭하여 나머지 모든 내용을 추출

            # \[(.*?)\]: Matches any character in brackets ([]) as non-greedy
            # (.*): Match any character (more than 0) as greedy and extract all the rest
            
            match = re.search(r'\[(.*?)\] \[(.*?)\] (.*)', line)
            if match:
                names.append(match.group(1))
                times.append(match.group(2))
                texts.append(match.group(3))
    
        df = pd.DataFrame({'Name': names, 'Time': times, 'Text': texts})
        merged_df = format_conversations(df)
    
        # 단톡방 인원 정리
        # Check Name list in message file
        name_list = list()
        for i in merged_df['Name'].unique():
            name_list.append(i)
        name_list = ["-- Select one below --"] + name_list
    
        placeholder = st.empty()
        with placeholder.container():
            version = st.radio('Who to make into a Chatbot?', name_list)
            # 챗봇으로 만들어질 대상 지정
            # Select the target for making a Chatbot
            if version == "-- Select one below --":
                st.write("Please select a target to make into a Chatbot.")
            else:
                for i in range(1, len(name_list)):
                    if version == name_list[i]:
                        final_data = question_answer(merged_df,name_list[i])
        if version != "-- Select one below --":
            placeholder.empty()
    
        if version != "-- Select one below --":
        # 빈칸이나 결측치 제거
        # Delete blank or missing values
            final_data = final_data[(final_data['질문'] != '') & (final_data['대답'] != '')].dropna()
    
            # Load Bert model
            # Streamlit은 작동할때마다 스크립트를 처음부터 다시 돌리기 때문에 캐쉬에 저장해서 사용하면 된다.
            # Streamlit turns the script back to the beginning every time it works, so you can save it to cache and use it.
            @st.cache_resource
            def cached_model():
                model = SentenceTransformer('jhgan/ko-sroberta-multitask')
                return model
            model = cached_model()
    
            # 로딩창 추가
            # Load embedding data (add progress bar)
            @st.cache_resource
            def get_dataset():
                # Create a progress bar and a status text 
                progress_bar = st.progress(0)
                status_text = st.empty()
    
                num_questions = len(final_data['질문'])
    
                embeddings = []
                for i, question in enumerate(final_data['질문']):
                    embedding = list(model.encode(question))
                    embeddings.append(embedding)
    
                    # Update the progress bar and the status text
                    progress_bar.progress((i + 1) / num_questions)
                    status_text.text(f'Processing question {i + 1} of {num_questions}...')
    
                final_data['embedding'] = embeddings
    
                # Clear the status text
                progress_bar.success("Processing is complete. 👈 See a left sidebar for chatting with your bot!")
                status_text.text("")
                
                return final_data
    
            final_data = get_dataset()
            
            # Chat form
            if 'generated' not in st.session_state:
                st.session_state['generated'] = []
            if 'past' not in st.session_state:
                st.session_state['past'] = []
    
            with st.sidebar.form('form', clear_on_submit=True):
                st.header(f"Chat with {version} bot 😄")
                user_input = st.text_input('User :', placeholder="Enter to submit")
                submitted = st.form_submit_button('Submit')
            if submitted and user_input:
                embedding = model.encode(user_input)
                final_data['similarity'] = final_data['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
                answer = final_data.loc[final_data['similarity'].idxmax()]
                st.session_state.past.append(user_input)
                st.session_state.generated.append(answer['대답'])
    
            for i in range(len(st.session_state['past'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
                if len(st.session_state['generated']) > i:
                    message(st.session_state['generated'][i], key=str(i) + '_bot')
    
    # Cache initialize
    if uploaded_file is None:
        st.cache_resource.clear()
        st.session_state.clear()

if __name__ == '__main__' :
    main()
