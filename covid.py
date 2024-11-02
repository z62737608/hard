import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EXCEL_FILE_PATH = 'qna.xlsx'

@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path)
    df['Q'] = df['Q'].fillna('')
    df['A'] = df['A'].fillna('')
    return df

df = load_data(EXCEL_FILE_PATH)

@st.cache_resource
def train_vectorizer(data):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(data['Q'].tolist())
    return vectorizer, vectors

question_vectorizer, question_vector = train_vectorizer(df)

def get_most_similar_question(user_question, threshold):
    new_sen_vector = question_vectorizer.transform([user_question])
    simil_score = cosine_similarity(new_sen_vector, question_vector)
    if simil_score.max() < threshold:
        return None, "이것은 코로나19와 관련없는 내용입니다."
    else:
        max_index = simil_score.argmax()
        most_similar_question = df['Q'].tolist()[max_index]
        most_similar_answer = df['A'].tolist()[max_index]
        return most_similar_question, most_similar_answer

st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Nanum+Gothic:wght@400;700&display=swap');
        body {{
            background-color: #ffffff;
            font-family: 'Nanum Gothic', sans-serif;
            color: #333333;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2em;
            margin: 0 auto;
            width: 90%;
            max-width: 1200px;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
        }}
        .content {{
            width: 100%;
            border-radius: 10px;
            padding: 2em;
            text-align: left;
            background-color: rgba(255, 255, 255, 0.3);
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }}
        .stButton button {{
            margin: 10px 0;
            width: 100%;
            border-radius: 10px;
            border: 1px solid #04751a;
            padding: 15px 30px;
            color: #ffffff;
            background-color: #04751a;
            font-size: 1.2em;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }}
        .stButton button:hover {{
            background-color: #ff0000;
            color: #ffffff;
            cursor: pointer;
        }}
        h1 {{
            color: #000000;
            text-align: center;
            font-weight: 700;
            margin-bottom: 0.5em;
        }}
        .header {{
            font-size: 2em;
            font-weight: bold;
            color: #0ABAB5;
            text-align: center;
            margin-bottom: 1em;
        }}
        .section {{
            background-color: rgba(255, 255, 0, 100);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-size: 1.2em;
            color: #000000;
            box-shadow: 0 0 10px rgba(0, 255, 0, 100);
            animation: fadeIn 0.5s ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}
        .small-button {{
            margin: 10px 5px;
            width: auto;
            border-radius: 5px;
            border: 1px solid #0ABAB5;
            padding: 8px 15px;
            color: #0ABAB5;
            background-color: #ffffff;
            font-size: 0.8em;
            font-weight: 600;
            transition: all 0.3s ease-in-out;
        }}
        .small-button:hover {{
            background-color: #0ABAB5;
            color: #ffffff;
            cursor: pointer;
        }}
    </style>
""", unsafe_allow_html=True)

if 'active_button' not in st.session_state:
    st.session_state.active_button = None

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]

st.title("코로나19 예측 모듈")

st.sidebar.header("세부 내용")

large_buttons = {
    "코로나19 위험성": "A.mp4",
    "코로나19 대비 방법": "B.mp3"
}

small_buttons = {
    "코로나19란?": "x",
    "코로나19가 발생한 시기": "y.",
    "코로나19와 감기의 차이": "z.",
    "코로나19가 불러온 악영향": "aa.",
    "코로나19의 특징": "bb.",
    "코로나19 백신 종류": "cc.",
    "코로나19로 인한 자가격리 설명": "dd.",
    "코로나19의 전망": "ee."
}

for button, content in large_buttons.items():
    if st.sidebar.button(button, key=button):
        st.session_state.active_button = button if st.session_state.active_button != button else None
    if st.session_state.active_button == button:
        if button == "코로나19 위험성":
            st.sidebar.video(content)
        elif button == "코로나19 대비 방법":
            st.sidebar.audio(content)

for button, content in small_buttons.items():
    if st.sidebar.button(button, key=button):
        st.session_state.active_button = button if st.session_state.active_button != button else None
    if st.session_state.active_button == button:
        st.markdown(f"<div class='section'>{content}</div>", unsafe_allow_html=True)

threshold = st.slider("예측 오차 정도", 0.0, 1.0, 0.43)

user_input = st.text_input("내용을 입력하세요:")

if st.button("검색", key="search", help="small"):
    if user_input:
        similar_question, answer = get_most_similar_question(user_input, threshold)
        
        if similar_question:
            st.session_state.conversation_history.append({"role": "assistant", "content": f"내용: {similar_question}"})
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
            st.write(f"**내용:** {similar_question}")
            st.write(f"**답변:** {answer}")
        else:
            st.write("이것은 코로나19와 관련없는 내용입니다.")

if st.button("이전 내용 보기", key="view_history", help="small"):
    st.write("### 기록")
    for msg in st.session_state.conversation_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        st.write(f"**{role}:** {msg['content']}")

if st.button("검색 초기화", key="new_search", help="small"):
    st.session_state.conversation_history = [{"role": "system", "content": "You are a helpful assistant."}]
    st.experimental_rerun()

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
