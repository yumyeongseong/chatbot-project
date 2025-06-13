import os

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

## 환경변수 읽어오기 ===================================================================
load_dotenv()


## LLM 함수 정의
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


## Database 함수 정의 =================================================================
def get_database():
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    
    ## 임베딩 모델 지정
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = 'mit10th'

    ## 저장된 인덱스 가져오기 ========================================================
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding,
    )
    
    return database

### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

## retrievalQA 함수 정의 ==============================================================
def get_retrievalQA():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    database = get_database()
    llm = get_llm()
    retriever = database.as_retriever(search_kwargs={'k':2})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # ===================================================
    system_prompt = (
        '''
         [identity]
        - 당신은 과학이야기를 아주 재밌게 이야기하는 과학 커뮤니에이터입니다.
        - [context]를 참고하여 사용자의 질문에 답변하세요.
        - 재밌는 이야기 하듯이 레포트를 풀어서 얘기해 주세요
        - 문단 마지막에 <'레포트 제목', '레포트 발행일자'>를 답변하세요.
        - 업데이트된 레포트 이외 질문을 하면 '레포트를 업데이트 중입니다 추후 질문 부탁드립니다.'라고 답변하세요.
        - 대답 할 수 없는 정보에 대해서는 현재 읽을 수 있는 [context]에 대해서 설명해 줄수 있다고 덧 붙여주세요.
        {context}
        '''
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    # ===================================================
   
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key='answer',
    ).pick('answer')


    return conversational_rag_chain

## [AI message 함수 정의] ==================================
def get_ai_message(user_message, session_id='default'):
    qa_chain = get_retrievalQA()

    
    # ai_message = qa_chain.invoke(user_message)

    ai_message = qa_chain.stream(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}},
    )
    
    return ai_message


