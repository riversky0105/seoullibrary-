# streamlit_app.py
import streamlit as st
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans
import plotly.express as px

# 데이터 불러오기
@st.cache
def load_data():
    df = pd.read_excel("서울시 공공도서관 서울도서관 이용자 현황.xlsx", sheet_name="최신 이용자", skiprows=2)
    df = df[pd.to_numeric(df['이용자수'], errors='coerce').notnull()]
    df = df[['실거주', '이용자수']].rename(columns={'실거주': '구명'})
    df['이용자수'] = pd.to_numeric(df['이용자수'])
    return df

# 지도 정보 (서울시 행정구역 중심 좌표가 필요)
@st.cache
def load_geo_data():
    # 서울시 각 구의 중심 좌표 예시 데이터 (사용자 데이터로 대체 가능)
    return pd.DataFrame({
        '구명': ['강남구', '서초구', '마포구', '송파구', '노원구', '중구'],
        'lat': [37.5172, 37.4836, 37.5663, 37.5145, 37.6543, 37.5636],
        'lon': [127.0473, 127.0324, 126.9015, 127.1066, 127.0568, 126.9976]
    })

st.title("서울시 도서관 이용자 수 분석")
st.markdown("출처: 서울시 공공도서관 서울도서관 (2018~2023)")

df = load_data()
geo = load_geo_data()

# 병합
merged = pd.merge(df, geo, on="구명", how="inner")

# KMeans 클러스터링
kmeans = KMeans(n_clusters=5, random_state=0)
merged["cluster"] = kmeans.fit_predict(merged[["이용자수"]])

# 시각화
fig = px.scatter_mapbox(
    merged,
    lat="lat",
    lon="lon",
    color="이용자수",
    size="이용자수",
    hover_name="구명",
    mapbox_style="carto-positron",
    zoom=10,
    color_continuous_scale="Viridis",
    title="서울시 도서관 이용자 수 (군집화)"
)

st.plotly_chart(fig)

