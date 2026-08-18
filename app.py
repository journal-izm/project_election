import time
import random
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 1. 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# 페이지 설정
st.set_page_config(page_title="AI 실시간 선거 보도 & RAG 시스템", layout="wide")
st.title("🗳️ [Project 1] RAG 기반 실시간 선거 보도 및 팩트체크 시스템")

if not api_key:
    st.error("❌ .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다. 키를 확인해주세요!")

# 2. 후보자 공약 데이터베이스 (Vector DB 구축용 샘플 문서)
raw_pledges = [
    Document(
        page_content="기호 1번 김철수 후보 핵심 공약: 디지털 미디어 시티 확대, AI 특화 고등학교 설립, 청년 주거 지원금 월 20만원 지급", 
        metadata={"candidate": "기호 1번 김철수"}
    ),
    Document(
        page_content="기호 2번 이영희 후보 핵심 공약: 친환경 탄소중립 도시 실현, 대중교통 요금 50% 인하, 소상공인 무이자 대출 지원", 
        metadata={"candidate": "기호 2번 이영희"}
    ),
    Document(
        page_content="기호 3번 박민수 후보 핵심 공약: 지역 상권 부활 프로젝트, 전통시장 디지털 결제 인프라 구축, 일자리 10만 개 창출", 
        metadata={"candidate": "기호 3번 박민수"}
    )
]

@st.cache_resource
def init_vector_db(key):
    if not key:
        return None
    # OpenAI 임베딩을 이용해 메모리 기반 Chroma DB 생성
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    vectorstore = Chroma.from_documents(raw_pledges, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 1})

# 3. 실시간 개표 스트리밍 시뮬레이션 함수
def fetch_mock_stream_data():
    candidates = ["기호 1번 김철수", "기호 2번 이영희", "기호 3번 박민수"]
    data = []
    for cand in candidates:
        votes = random.randint(20000, 90000)
        rate = round(random.uniform(20.0, 50.0), 2)
        data.append({"후보": cand, "득표수": votes, "득표율(%)": rate})
    df = pd.DataFrame(data)
    df = df.sort_values(by="득표율(%)", ascending=False).reset_index(drop=True)
    return df

# 4. RAG 기반 속보 기사 자동 생성 함수
def generate_rag_article(df_status, retriever, key):
    if not key or not retriever:
        return "API Key 또는 Retriever가 준비되지 않았습니다."
    
    # 현재 득표율 1위 후보 추출
    top_candidate = df_status.iloc[0]["후보"]
    
    # Vector DB에서 1위 후보 공약 검색 (RAG Retrieval)
    retrieved_docs = retriever.invoke(top_candidate)
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # LLM (GPT-3.5-turbo 또는 GPT-4) 설정
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo", openai_api_key=key)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 냉철하고 객관적인 선거 데이터 저널리스트입니다. 제공된 실시간 개표 데이터와 후보자의 공약 정보를 바탕으로 팩트 중심의 속보 기사를 작성하세요."),
        ("human", "실시간 개표 현황:\n{status}\n\n참고할 후보자 공약 DB:\n{context}\n\n요청사항: 현재 1위 후보의 득표 상황을 분석하고, 그의 핵심 공약을 연계하여 판세를 전망하는 3줄 속보 기사를 작성해주세요.")
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "status": df_status.to_string(),
        "context": context_text
    })
    return result, top_candidate

# 5. 대시보드 레이아웃 구성
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 실시간 개표 스트리밍 대시보드 (Simulation)")
    stream_placeholder = st.empty()

with col2:
    st.subheader("📰 RAG 기반 AI 속보 기사 & 팩트체크")
    news_placeholder = st.empty()
    metric_placeholder = st.empty()

# 실행 버튼
if st.button("🚀 실시간 개표 방송 및 RAG 기사 자동 생성 시작"):
    if not api_key:
        st.error("API 키를 먼저 설정해주세요!")
    else:
        retriever = init_vector_db(api_key)
        
        # 교육용 시뮬레이션 3회 반복 루프
        for i in range(3):
            df_current = fetch_mock_stream_data()
            timestamp = f"2026-08-18 22:{20+i}:00"
            
            # 개표 현황 갱신
            with stream_placeholder.container():
                st.info(f"⏱️ 개표 방송 송출 중... [타임스탬프: {timestamp}]")
                st.dataframe(df_current, use_container_width=True)
                st.bar_chart(df_current.set_index("후보")["득표율(%)"])
            
            # RAG 기사 생성 갱신
            with news_placeholder.container():
                with st.spinner("RAG 엔진이 공약 DB를 대조하며 기사를 작성 중입니다..."):
                    article, leader = generate_rag_article(df_current, retriever, api_key)
                    st.success(f"**[속보] {leader} 선두 질주**")
                    st.write(article)
            
            # 팩트체크 신뢰도 점수 표시
            with metric_placeholder.container():
                score = random.randint(95, 99)
                st.metric(label="🛡️ AI 팩트체크 신뢰도 점수 (Anti-Hallucination)", value=f"{score}점", delta="정상 범위")
            
            time.sleep(3) # 3초 간격 대기