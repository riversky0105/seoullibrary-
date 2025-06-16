import streamlit as st
st.set_page_config(page_title="서울시 도서관 분석 및 예측", layout="wide")

st.title("📚 서울특별시 자치구별 도서관 이용자 수 분석 및 예측")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import folium
import requests
import json
from shapely.geometry import shape
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 한글 폰트 설정
font_path = os.path.join(os.getcwd(), "fonts", "NanumGothicCoding.ttf")
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams['axes.unicode_minus'] = False
else:
    font_prop = None

@st.cache_data
def load_ml_data():
    df = pd.read_csv("공공도서관 자치구별 통계 파일.csv", encoding='cp949', header=1)
    df = df[df.iloc[:, 0] != '소계']
    df.columns = [
        '자치구명','개소수','좌석수','자료수_도서','자료수_비도서','자료수_연속간행물',
        '도서관 방문자수','연간대출책수','직원수','직원수_남','직원수_여','예산'
    ]
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    return df

df_stat = load_ml_data()

df_users = df_stat[['자치구명','도서관 방문자수']].copy()
df_users.columns = ['구','이용자수']
df_users['이용자수'] = df_users['이용자수'].astype(int)
df_users_sorted = df_users.sort_values(by='이용자수', ascending=False).reset_index(drop=True)

# 📊 자치구별 도서관 이용자 수 시각화
st.subheader("📊 자치구별 도서관 이용자 수 그래프 시각화")
st.markdown("서울특별시 각 자치구의 도서관 방문자 수를 나타낸 막대그래프입니다. 그래프는 방문자 수가 많은 자치구부터 순서대로 정렬되어 있으며, 막대의 길이가 방문자 수 규모를 나타냅니다.")

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(df_users_sorted['구'], df_users_sorted['이용자수'], color='skyblue')
ax.set_title("📌 자치구별 이용자 수", fontproperties=font_prop)
ax.set_xlabel("자치구", fontproperties=font_prop)
ax.set_ylabel("이용자 수", fontproperties=font_prop)
ax.set_xticks(range(len(df_users_sorted)))
ax.set_xticklabels(df_users_sorted['구'], rotation=45, fontproperties=font_prop)
yticks = ax.get_yticks()
ax.set_yticklabels([f"{int(t):,}" for t in yticks], fontproperties=font_prop)
st.pyplot(fig)

# 🗺️ 지도 시각화
st.subheader("🗺️ 자치구별 도서관 이용자 수 지도 시각화")
st.markdown("서울특별시 각 자치구의 경계 안에 위치한 마커는 해당 구의 도서관 방문자 수 규모를 나타냅니다. 마커 크기가 클수록 방문자 수가 많습니다.")

geo_url = "https://raw.githubusercontent.com/southkorea/seoul-maps/master/kostat/2013/json/seoul_municipalities_geo_simple.json"
res = requests.get(geo_url)
seoul_geo = res.json()

m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
folium.GeoJson(seoul_geo, name="경계", style_function=lambda f: {
    'fillColor': '#dddddd',
    'color': 'black',
    'weight': 3,
    'fillOpacity': 0.2
}).add_to(m)

min_v, max_v = df_users['이용자수'].min(), df_users['이용자수'].max()
for feature in seoul_geo['features']:
    gu = feature['properties']['name']
    if gu in df_users['구'].values:
        center = shape(feature['geometry']).centroid
        val = df_users[df_users['구'] == gu]['이용자수'].values[0]
        norm = (val - min_v) / (max_v - min_v)
        folium.CircleMarker(
            location=[center.y, center.x],
            radius=10 + 30 * norm,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f"{gu}: {val:,}명"
        ).add_to(m)

folium.LayerControl().add_to(m)
folium_static(m)

top = df_users_sorted.iloc[0]
st.success(f"✅ 도서관을 가장 많이 이용한 구는 **{top['구']}**입니다, 이용자 수는 총 **{top['이용자수']:,}명**입니다.")

# 🔍 변수 중요도 분석
st.subheader("🔍 변수 중요도 분석")
st.markdown("도서관 이용자 수에 어떤 변수들이 영향을 주는지를 분석했습니다. 중요도가 높을수록 해당 변수가 방문자 수에 더 큰 영향을 준다는 뜻입니다.")

X = df_stat.drop(columns=['자치구명', '도서관 방문자수'])
y = df_stat['도서관 방문자수']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.markdown(f"✅ **예측 오차(MSE)**: `{mse:,.0f}` | **정확도(R²)**: `{r2:.4f}`")

importance = pd.Series(model.feature_importances_, index=X.columns)
fig2, ax2 = plt.subplots(figsize=(10, 6))
importance.sort_values().plot.barh(ax=ax2, color='skyblue')
ax2.set_title("📌 RandomForest 변수 중요도", fontproperties=font_prop)
ax2.set_xlabel("중요도", fontproperties=font_prop)
ax2.set_ylabel("변수", fontproperties=font_prop)
ax2.set_yticklabels(importance.sort_values().index, fontproperties=font_prop)
xticks = ax2.get_xticks()
ax2.set_xticklabels([f"{x:.1f}" for x in xticks], fontproperties=font_prop)
st.pyplot(fig2)

# 📄 자치구별 통계 데이터
st.subheader("📄 자치구별 통계 데이터")
st.markdown("각 자치구의 도서관 관련 운영 지표를 확인할 수 있는 표입니다. 분석 및 시각화에 사용된 모든 원천 데이터입니다.")
st.dataframe(df_stat)
