import streamlit as st
from llm import get_ai_message
st.set_page_config(page_title='📑MIT 과학이야기')
st.title('📑MIT 읽어주는 Chabot')

if 'message_list' not in st.session_state:
    st.session_state.message_list=[]

print(f'before: {st.session_state.message_list}')

## 이전 채팅 내용 화면에 출력(표시)

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

## prompt 창(채팅 창) ##############################

placeholder= '어떤 레포트를 정리해 드릴까요?'
if user_question := st.chat_input(placeholder=placeholder):
    with st.chat_message('user'):
        ## 사용자 메시지 화면 출력
        st.write(user_question)
    st.session_state.message_list.append({'role':'user','content':user_question})

    with st.spinner('이야기 보따리에서 이야기를 꺼내는 중입니다.'):

        session_id = 'user-session'
        ai_message = get_ai_message(user_question, session_id=session_id)

        with st.chat_message('ai'):
            st.write(ai_message)
        st.session_state.message_list.append({'role':'ai','content':ai_message})

