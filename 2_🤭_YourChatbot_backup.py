import streamlit as st
import tempfile
import re
import pandas as pd
from langchain.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_chat import message

def format_conversations(df):
    '''
    여러명일경우 하나로 일단 압축시킨다.
    그리고 챗봇 대상의 "answer"이전 사람의 대사만 "questions"으로 할당한다.
    If there are multiple people, compress it into one.
    And only the lines of the person before "answer" to be heard are assigned as "questions."
    
    e.g
    
    C <- target to make into a Chatbot
    
    A : Hi guys
    B : Hey~
    B : What are you doing?
    C : I'm having a meal with ma mates.
    
    questions = Hey~ What are you doing?
    answer = I'm having a meal with ma mates.
    '''
    current_name = ''
    # conversations = []
    names = []
    texts = []

    for index, row in df.iterrows():
        name = row['Name']
        text = row['Text']

        if name != current_name:
            names.append(name)
            texts.append(text)
            current_name = name
        else:
            texts[-1] += ' ' + text # On what basis will the data be divided?

    result_df = pd.DataFrame({'Name': names, 'Text': texts})
    return result_df

# "answer_user" 기준으로 질문과 답변이 나뉜다.
# Questions and answers are divided based on "answer_user".
def question_answer(df, answer_user):
    questions = []
    answers = []
    current_question = ''

    for index, row in df.iterrows():
        question = current_question
        answer = ''

        text = row['Text']

        if row['Name'] == answer_user:
            answer = text
        else:
            question = text

        if answer.strip() != '':
            questions.append(question)
            answers.append(answer)

        current_question = question

    result_df = pd.DataFrame({'질문': questions, '대답': answers})
    return result_df

st.title('🦜🔗Make your own Chat Bot :tongue:')
st.sidebar.success("👆 Select a page above.")

uploaded_file = st.file_uploader("Upload your **Kakao Talk Message file.txt	📃** Your file don't be stored here.")
if uploaded_file:
    text = []
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    loader = TextLoader(temp_file_path)
    
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


        # Embedding (no progress bar)
        # @st.cache(allow_output_mutation=True)
        # @st.cache_resource
        # def get_dataset():
        #     final_data['embedding'] = final_data['질문'].map(lambda x: list(model.encode(x)))
        #     return final_data
        # final_data = get_dataset()

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
    
    
    
    
    
    
        # Chat form
        # with st.sidebar.form("form", clear_on_submit=True):
        #     st.header(f"Chat with {version} bot 😄")
        #     question = st.text_input("Question :", placeholder="Enter to submit")
        #     submit = st.form_submit_button("Submit")
        #     if submit and question:
        #         with st.spinner("입력중.."):
        #             embedding = model.encode(question)
        #             final_data['similarity'] = final_data['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
        #             answer = final_data.loc[final_data['similarity'].idxmax()]
        #             st.write(answer['대답'])