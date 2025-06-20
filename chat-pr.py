import uuid

import streamlit as st
from llm import stream_ai_message

st.set_page_config(page_title='📑MIT 리포트')
st.title('📑MIT가 선정한 미래 기술 리포트')

## URL의 parameter에 session id 가져오기/저장
query_params = st.query_params

if 'session_id' in query_params:
    session_id = query_params['session_id']

else:
    session_id = str(uuid.uuid4())
    st.query_params.update({'session_id': session_id})


## Streamlit 내부 세션: session id 저장
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = session_id

if 'message_list' not in st.session_state:
    st.session_state.message_list=[]

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

## 예시 질문 제안
st.markdown("""
### ✨ 이런 질문을 해보세요!

- 📋 **MIT에서 선정한 기술**이 뭐야?
- 💡 **소형언어모델**이 뭐야?
- 🏭 **녹색철강**에 대해 알려줘
- 🚕 **로보택시**가 뭔가요?
- ✈️ **청정 제트연료**는 어떻게 만들어져?
""")

## 채팅 메시지
placeholder= '질문을 작성해 주세요.'

if user_question := st.chat_input(placeholder=placeholder):
        ## 사용자 메시지 화면 출력
    with st.chat_message('user'):
        st.write(user_question)
    st.session_state.message_list.append({'role':'user','content':user_question})

    with st.spinner('대답을 생성 중입니다.'):
        session_id = st.session_state.session_id
        ai_message = stream_ai_message(user_question, session_id=session_id)

        with st.chat_message('ai'):
            st.write(ai_message)
        st.session_state.message_list.append({'role':'ai','content':ai_message})

