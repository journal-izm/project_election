import os
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="선거 개표 실시간 스트리밍 서버", version="1.0")

DATA_PATH = "data/중앙선거관리위원회_대통령선거 개표결과_20250603.csv"
processed_data_cache = []

def load_and_prep_election_data():
    global processed_data_cache
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ 경고: '{DATA_PATH}' 파일을 찾을 수 없습니다. data/ 폴더를 확인해주세요.")
        return
    
    # 실제 선관위 CSV 로드
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
    
    # 비후보 항목(선거인수, 투표수 등) 제외
    exclude_items = ['선거인수', '투표수', '무효 투표수', '기권자수']
    df_clean = df[~df['후보자'].isin(exclude_items)].copy()
    
    # 구시군명 단위로 누적 스트림 데이터 시뮬레이션 구성
    regions = df_clean['구시군명'].unique()
    
    for i in range(1, min(len(regions), 20) + 1):
        subset_regions = regions[:i]
        sub_df = df_clean[df_clean['구시군명'].isin(subset_regions)]
        agg = sub_df.groupby('후보자')['득표수'].sum().reset_index()
        total_votes = agg['득표수'].sum()
        
        if total_votes > 0:
            agg['vote_rate'] = round((agg['득표수'] / total_votes) * 100, 2)
        else:
            agg['vote_rate'] = 0.0
        
        agg = agg.sort_values(by='vote_rate', ascending=False).reset_index(drop=True)
        agg['step_id'] = i
        agg['current_region'] = regions[i-1]
        processed_data_cache.append(agg)

load_and_prep_election_data()

@app.get("/")
def home():
    return {"status": "online", "message": "선거 개표 모의 서버 정상 작동 중", "total_steps": len(processed_data_cache)}

@app.get("/api/stream-votes/{step}")
def get_stream_data(step: int):
    if not processed_data_cache:
        raise HTTPException(status_code=404, detail="개표 데이터가 로드되지 않았습니다.")
    
    if step > len(processed_data_cache):
        return {"status": "finished", "message": "모든 개표 데이터 스트리밍이 완료되었습니다."}
    
    current_df = processed_data_cache[step - 1]
    return {
        "status": "streaming",
        "step": step,
        "target_region": current_df['current_region'].iloc[0],
        "data": current_df.to_dict(orient="records")
    }