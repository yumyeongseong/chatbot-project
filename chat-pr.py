import streamlit as st

st.set_page_config(page_title='📑MIT 리포트 챗봇')
st.title('📑MIT 리포트 챗봇')

if 'message_list' not in st.session_state:
    st.session_state.message_list=[]

print(f'before: {st.session_state.message_list}')

## 이전 채팅 내용 화면에 출력(표시)

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

## prompt 창(채팅 창) ##############################

placeholder= '어떤 리포트를 정리해 드릴까요?'
if user_question := st.chat_input(placeholder=placeholder):
    with st.chat_message('user'):
        ## 사용자 메시지 화면 출력
        st.write(user_question)
    st.session_state.message_list.append({'role':'user','content':user_question})

with st.chat_message('ai'):
    st.write('리포트 업데이트 중입니다. 추후 방문 부탁드립니다.')
st.session_state.message_list.append({'role':'ai','content':'리포트 업데이트 중입니다. 추후 방문 부탁드립니다.'})

print(f'after: {st.session_state.message_list}')

print(user_question)