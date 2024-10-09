import pandas as pd
import chardet

def detect_encoding(file_path):
    """파일의 인코딩을 자동으로 감지하여 반환합니다."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        return result['encoding']

def load_txt_file(file_path):
    """텍스트 파일을 로드하는 함수로, UTF-8 및 다른 인코딩을 처리합니다."""
    try:
        # 기본적으로 UTF-8로 시도
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # UTF-8이 실패하면 다른 인코딩으로 시도
        detected_encoding = detect_encoding(file_path)
        try:
            with open(file_path, 'r', encoding=detected_encoding) as file:
                return file.read()
        except Exception as e:
            raise RuntimeError(f"Failed to load txt file: {e}")
        
        
# your chat bot tools
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