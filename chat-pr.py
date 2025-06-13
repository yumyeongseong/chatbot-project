import streamlit as st
from llm import get_ai_message
st.set_page_config(page_title='📑MIT 리포트')
st.title('📑MIT가 선정한 미래 기술 리포트')

if 'message_list' not in st.session_state:
    st.session_state.message_list=[]

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

## prompt 창(채팅 창) ##############################

placeholder= '질문을 작성해 주세요.'
if user_question := st.chat_input(placeholder=placeholder):
    with st.chat_message('user'):
        ## 사용자 메시지 화면 출력
        st.write(user_question)
    st.session_state.message_list.append({'role':'user','content':user_question})

    with st.spinner('대답을 생성 중입니다.'):

        session_id = 'user-session'
        ai_message = get_ai_message(user_question, session_id=session_id)

        with st.chat_message('ai'):
            st.write(ai_message)
        st.session_state.message_list.append({'role':'ai','content':ai_message})

