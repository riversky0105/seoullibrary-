import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt

# -------------------
# 데이터 불러오기
# -------------------
@st.cache_data
def load_data():
    df = pd.read_excel("서울시 공공도서관 서울도서관 이용자 현황.xlsx", sheet_name="최신 이용자", skiprows=2)
    df.columns = ['구', '이용자수'] + list(df.columns[2:])  # 첫 두 열 이름 설정
    df = df[['구', '이용자수']].dropna()
    df['이용자수'] = pd.to_numeric(df['이용자수'], errors='coerce')
    return df.dropna()

df = load_data()

# -------------------
# 앱 제목
# -------------------
st.set_page_config(page_title="서울시 도서관 이용자 수 분석", layout="wide")
st.title("📚 서울시 도서관 이용자 수 분석")
st.markdown("**2018~2023년 서울시 공공도서관 이용자 데이터를 시각화한 대시보드입니다.**")

# -------------------
# 지표 요약
# -------------------
col1, col2 = st.columns(2)
col1.metric("총 자치구 수", df['구'].nunique())
col2.metric("총 이용자 수", f"{int(df['이용자수'].sum()):,}명")

# -------------------
# 자치구별 바 차트
# -------------------
st.subheader("📈 자치구별 도서관 이용자 수")
fig, ax = plt.subplots(figsize=(12, 6))
df_sorted = df.sort_values(by="이용자수", ascending=False)
ax.bar(df_sorted['구'], df_sorted['이용자수'], color='skyblue')
ax.set_ylabel("이용자 수")
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------
# 지도 시각화
# -------------------
st.subheader("🗺️ 자치구별 이용자 수 지도")

# 서울시 각 구의 대표 좌표 (위도, 경도)
sample_locations = {
    "강남구": [37.5172, 127.0473],
    "강동구": [37.5301, 127.1238],
    "강북구": [37.6396, 127.0256],
    "강서구": [37.5509, 126.8495],
    "관악구": [37.4784, 126.9516],
    "광진구": [37.5385, 127.0823],
    "구로구": [37.4954, 126.8874],
    "금천구": [37.4569, 126.8958],
    "노원구": [37.6542, 127.0568],
    "도봉구": [37.6688, 127.0470],
    "동대문구": [37.5744, 127.0396],
    "동작구": [37.5124, 126.9392],
    "마포구": [37.5663, 126.9014],
    "서대문구": [37.5791, 126.9368],
    "서초구": [37.4836, 127.0326],
    "성동구": [37.5633, 127.0367],
    "성북구": [37.5894, 127.0167],
    "송파구": [37.5145, 127.1056],
    "양천구": [37.5169, 126.8664],
    "영등포구": [37.5264, 126.8963],
    "용산구": [37.5326, 126.9903],
    "은평구": [37.6176, 126.9227],
    "종로구": [37.5731, 126.9795],
    "중구": [37.5636, 126.9976],
    "중랑구": [37.6063, 127.0927]
}

# folium 지도 생성
m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

# 정규화 함수 (반지름 계산용)
def normalize(val, min_val, max_val):
    return 5 + 15 * ((val - min_val) / (max_val - min_val))  # 반지름: 5~20

min_users, max_users = df['이용자수'].min(), df['이용자수'].max()

for _, row in df.iterrows():
    gu = row['구']
    users = row['이용자수']
    if gu in sample_locations:
        folium.CircleMarker(
            location=sample_locations[gu],
            radius=normalize(users, min_users, max_users),
            popup=f"{gu}: {int(users):,}명",
            color='blue',
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

folium_static(m, width=1000)

# -------------------
# 요약 메시지
# -------------------
top_gu = df_sorted.iloc[0]
st.success(f"✅ **가장 도서관 이용자 수가 많은 구는 `{top_gu['구']}`이며, 총 `{int(top_gu['이용자수']):,}명`이 이용했습니다.**")

# -------------------
# 마무리 안내
# -------------------
st.markdown("---")
st.caption("🔗 더 많은 AI 프로젝트 도구는 [https://gptonline.ai/ko/](https://gptonline.ai/ko/) 에서 확인하세요.")




