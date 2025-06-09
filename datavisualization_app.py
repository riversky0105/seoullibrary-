import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt

# 파일 불러오기
@st.cache_data
def load_data():
    df = pd.read_excel("서울시 공공도서관 서울도서관 이용자 현황.xlsx", sheet_name="최신 이용자", skiprows=2)
    df.columns = ['구', '이용자수'] + list(df.columns[2:])  # 간단하게 열 이름 정리
    df = df[['구', '이용자수']].dropna()
    df['이용자수'] = pd.to_numeric(df['이용자수'], errors='coerce')
    return df.dropna()

df = load_data()

st.title("📚 서울시 도서관 이용자 수 분석")
st.markdown("**2018~2023년 서울시 공공도서관 이용자 데이터를 시각화한 대시보드입니다.**")

# 상단 요약
st.metric("총 구 수", df['구'].nunique())
st.metric("총 이용자 수", int(df['이용자수'].sum()))

# 바 차트
st.subheader("📈 자치구별 도서관 이용자 수")
fig, ax = plt.subplots(figsize=(10, 5))
df_sorted = df.sort_values(by="이용자수", ascending=False)
ax.bar(df_sorted['구'], df_sorted['이용자수'], color='skyblue')
plt.xticks(rotation=45)
st.pyplot(fig)

# 지도 시각화 (기본 중심 좌표는 서울)
st.subheader("🗺️ 자치구별 이용자 수 지도")
m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

# 샘플 좌표 (실제 데이터에는 좌표 매핑 필요)
sample_locations = {
    "강남구": [37.5172, 127.0473],
    "강동구": [37.5301, 127.1238],
    "강북구": [37.6396, 127.0256],
    # ...
}

for _, row in df.iterrows():
    gu = row['구']
    if gu in sample_locations:
        folium.CircleMarker(
            location=sample_locations[gu],
            radius=row['이용자수'] / 10000,  # 사용자 수에 따라 크기 조절
            popup=f"{gu}: {int(row['이용자수'])}명",
            color='blue',
            fill=True,
            fill_opacity=0.6
        ).add_to(m)

folium_static(m)

# 가장 이용자 수 많은 구
top_gu = df_sorted.iloc[0]
st.success(f"✅ **가장 이용자 수가 많은 구: {top_gu['구']} ({int(top_gu['이용자수'])}명)**")




