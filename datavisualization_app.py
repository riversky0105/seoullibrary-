import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
import json

st.set_page_config(layout="wide")
st.title("📚 서울시 자치구별 도서관 이용자 수 분석")

@st.cache_data
def load_data():
    df_raw = pd.read_excel("서울시 공공도서관 서울도서관 이용자 현황.xlsx", sheet_name="최신 이용자")
    df = df_raw.iloc[1:].copy()
    df.columns = df_raw.iloc[0]
    df = df.reset_index(drop=True)

    # 컬럼명 통일
    df = df.rename(columns={"지역": "도서관명", "자치구": "구", "이용자수": "총이용자수"})
    df["총이용자수"] = pd.to_numeric(df["총이용자수"], errors="coerce")
    df = df.dropna(subset=["총이용자수", "구"])
    return df

@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/gisangy/Seoul-Goo-GeoJSON/main/seoul_municipalities_geo_simple.json"
    response = requests.get(url)
    return response.json()

df = load_data()
geojson = load_geojson()

# 자치구별 총합 계산
df_gu = df.groupby("구")["총이용자수"].sum().reset_index()
df_gu.columns = ["name", "총이용자수"]

# 지도 그리기
m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

folium.Choropleth(
    geo_data=geojson,
    data=df_gu,
    columns=["name", "총이용자수"],
    key_on="feature.properties.name",
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="도서관 이용자 수",
).add_to(m)

# 툴팁 추가
folium.GeoJson(
    geojson,
    name="자치구 경계",
    tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["자치구:"]),
).add_to(m)

st.subheader("🗺️ 자치구별 이용자 수 지도")
st_data = st_folium(m, width=1000, height=700)

