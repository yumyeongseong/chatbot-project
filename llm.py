import json
import os

from dotenv import load_dotenv
from langchain.chains import (create_history_aware_retriever,
                              create_retrieval_chain)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (ChatPromptTemplate, FewShotPromptTemplate,
                                    MessagesPlaceholder, PromptTemplate)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from config import answer_examples


## 환경변수 읽어오기 ===================================================================
load_dotenv()


## LLM 함수 정의
def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm


## Embedding 설정 + Vector Store Index 가져오기==========================================
def load_vectorstore():
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

## 히스토리 기반 리트리버 ==============================================================
def build_history_aware_retriever(llm, retriever):
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever

    
def build_few_shot_examples() -> str:
    example_prompt = PromptTemplate.from_template("질문: {input}\n\n답변:{answer}")

    few_shot_prompt = FewShotPromptTemplate(
        examples=answer_examples, ## 질문/답변 예시들 (전체 type: list, 각 질문/답변: dict)
        example_prompt=example_prompt, ## 단일 예시 포맷
        prefix='다음 질문에 답변하세요 : ',  ## 예시들 위로 추가되는 텍스트(도입부)
        suffix="질문: {input}", ## 예시들 뒤에 추가되는 텍스트(실제 사용자 질문 변수)
        input_variables=["input"],  ## suffix에서 사용할 변수
    )

    formated_few_shot_prompt = few_shot_prompt.format(input='{input}')

    return formated_few_shot_prompt

## [외부 사전 로드]
def load_dictionary_from_file(path= 'keyword_dictionary.json'):
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)

def buld_dictionary_text(dictionary: dict) -> str:
     return '\n'.join([
        f'{k} ({", ".join(v["tags"])}): {v["definition"]} [출처: {v["source"]}]'
        for k, v in dictionary.items()
    ])

def build_qa_prompt():

    keyword_dictionary = load_dictionary_from_file()

    dictionary_text = buld_dictionary_text(keyword_dictionary)

    system_prompt = (
        '''
            [identity]
        - 당신은 과학이야기를 아주 재밌게 이야기하는 과학 커뮤니에이터입니다.
        - 문단 마지막에 <레포트 제목>의 정보를 남겨주세요.
        - 업데이트된 레포트 이외 질문을 하면 '레포트를 업데이트 중입니다 추후 질문 부탁드립니다.'라고 답변하세요.
        - 사용자의 질문이 [context]에 없으면 [keyword_dictionary]를 참고해서 질문에 답하세요.
        - 여러 주제를 응답 할 때는 무조건 항목별로 응답하세요.
        [context]
        {context}

        [keyword_dictionary]
        {dictionary_text}
        '''
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    ).partial(dictionary_text=dictionary_text)

    return qa_prompt
   
## 전체 chain 구성 ===============================================================
def build_conversational_chain():
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

    ## LLM 모델 지정
    llm = get_llm()

    ## vector store에서 index 정보
    database = load_vectorstore()
    retriever = database.as_retriever(search_kwargs={'k':2})

    history_aware_retriever= build_history_aware_retriever(llm,retriever)
    
    qa_prompt = build_qa_prompt()
    
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
def stream_ai_message(user_message, session_id='default'):
    qa_chain =  build_conversational_chain()

    ai_message_stream = qa_chain.stream(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}},
    )

    ai_message = ''.join([chunk for chunk in ai_message_stream])

    # vector store에서 검색된 문서 출력
    retriever = load_vectorstore().as_retriever(search_kwargs={'k':2})
    search_results = retriever.invoke(user_message)
    
    print(f'\nPinecone 검색 결과 >> \n{search_results[0].page_content[:100]}')
    return ai_message
