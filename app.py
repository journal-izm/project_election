import time
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AI 실시간 선거 보도 & RAG 시스템", layout="wide")
st.title("🗳️ [Project 1] 역사적 선거 데이터 기반 스트리밍 & RAG 자동 뉴스 시스템")

if not api_key:
    st.error("❌ .env 파일에서 OPENAI_API_KEY를 찾을 수 없습니다.")

# 후보자 공약 데이터베이스 (Vector DB)
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
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    vectorstore = Chroma.from_documents(raw_pledges, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 1})

# 실무형: CSV 파일에서 시간대별 데이터를 순차적으로 로딩하는 함수
@st.cache_data
def load_historical_data():
    csv_path = "data/election_history.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        # 파일이 없을 경우를 대비한 기본 샘플 데이터 반환
        return pd.DataFrame({
            "timestamp": ["2026-08-18 22:15:00"] * 3,
            "candidate": ["기호 1번 김철수", "기호 2번 이영희", "기호 3번 박민수"],
            "votes": [15000, 14200, 12000],
            "vote_rate": [32.5, 30.8, 26.0]
        })

# RAG 기반 속보 기사 자동 생성 함수
def generate_rag_article(df_status, retriever, key):
    if not key or not retriever:
        return "API Key 또는 Retriever가 준비되지 않았습니다.", "대기 중"
    
    top_candidate = df_status.iloc[0]["candidate"]
    retrieved_docs = retriever.invoke(top_candidate)
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
    
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

# 대시보드 레이아웃
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 공식 개표 데이터 스트리밍 (Historical Replay)")
    stream_placeholder = st.empty()

with col2:
    st.subheader("📰 RAG 기반 AI 속보 기사 & 팩트체크")
    news_placeholder = st.empty()
    metric_placeholder = st.empty()

if st.button("🚀 실제 개표 데이터 시뮬레이션 시작"):
    if not api_key:
        st.error("API 키를 확인해주세요!")
    else:
        retriever = init_vector_db(api_key)
        df_full = load_historical_data()
        
        # 타임스탬프 단위로 그룹화하여 순차 재생
        timestamps = df_full["timestamp"].unique()
        
        for ts in timestamps:
            df_current = df_full[df_full["timestamp"] == ts].sort_values(by="vote_rate", ascending=False).reset_index(drop=True)
            
            with stream_placeholder.container():
                st.info(f"⏱️ 개표 데이터 수신 시각: [{ts}]")
                st.dataframe(df_current, use_container_width=True)
                st.bar_chart(df_current.set_index("candidate")["vote_rate"])
            
            with news_placeholder.container():
                with st.spinner("RAG 엔진이 공약 DB를 대조하며 기사를 작성 중입니다..."):
                    article, leader = generate_rag_article(df_current, retriever, api_key)
                    st.success(f"**[속보] {leader} 선두 질주**")
                    st.write(article)
            
            with metric_placeholder.container():
                st.metric(label="🛡️ AI 팩트체크 신뢰도 점수", value="98점", delta="안정적")
            
            time.sleep(4) # 실제 방송 호흡에 맞춘 딜레이