import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="도서관 이용자 수 분석", layout="wide")

st.title("📚 도서관 이용자 수 분석 및 지도 시각화")
st.markdown("서울시뿐 아니라 다른 지역도 포함하여 분석합니다.")

# 📂 1. 데이터 로딩 (엑셀 파일 고정)
@st.cache_data
def load_data():
    file_path = "서울시 공공도서관 서울도서관 이용자 현황.xlsx"
    df_raw = pd.read_excel(file_path, sheet_name="최신 이용자")
    df_clean = df_raw.iloc[1:].copy()
    df_clean.columns = df_raw.iloc[0]
    df_clean = df_clean.reset_index(drop=True)
    df_clean = df_clean.rename(columns={"지역": "도서관명", "이용자수": "총이용자수"})
    df_clean = df_clean[["도서관명", "총이용자수"]]
    df_clean["총이용자수"] = pd.to_numeric(df_clean["총이용자수"], errors="coerce")
    df_clean = df_clean.dropna(subset=["총이용자수"])
    return df_clean

df = load_data()
st.subheader("📋 데이터 미리보기")
st.write(df.head())

# 📍 2. 위도 경도 추가 (임시로 랜덤 생성 또는 사용자 지도 데이터 통합 필요)
# 실제 사용 시: 도서관 위치 데이터셋과 병합 필요
if "위도" not in df.columns or "경도" not in df.columns:
    np.random.seed(42)
    df["위도"] = np.random.uniform(37.4, 37.7, len(df))
    df["경도"] = np.random.uniform(126.8, 127.2, len(df))

# 📊 3. KMeans 클러스터링 (상위 5개 도서관 중심)
df_sorted = df.sort_values("총이용자수", ascending=False).reset_index(drop=True)
top5_coords = df_sorted.head(5)[["위도", "경도"]].values
coords = df[["위도", "경도"]].values

kmeans = KMeans(n_clusters=5, init=top5_coords, n_init=1)
df["cluster"] = kmeans.fit_predict(coords)

cluster_colors = ['red', 'blue', 'green', 'orange', 'purple']
df["color"] = df["cluster"].apply(lambda x: cluster_colors[x])

# 🌍 4. 지도 시각화
m = folium.Map(location=[df["위도"].mean(), df["경도"].mean()], zoom_start=11)

for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row["위도"], row["경도"]],
        radius=5 + (row["총이용자수"] / df["총이용자수"].max()) * 10,
        popup=f"{row['도서관명']}<br>이용자 수: {int(row['총이용자수']):,}",
        color=row["color"],
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

st.subheader("🗺️ 도서관 이용자 수 시각화")
st_folium(m, width=1000, height=700)
