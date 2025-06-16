import streamlit as st

# ✅ 반드시 가장 먼저 위치해야 함!
st.set_page_config(page_title="서울시 도서관 분석 및 예측", layout="wide")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import folium
from streamlit_folium import folium_static
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------
# 1. 한글 폰트 설정
# -----------------------
font_path = os.path.join(os.getcwd(), "fonts", "NanumGothicCoding.ttf")
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    mpl.rcParams['axes.unicode_minus'] = False
    st.write(f"✅ 한글 폰트 적용 완료: {font_prop.get_name()}")
else:
    font_prop = None
    st.warning("⚠️ NanumGothicCoding.ttf 폰트 파일이 없습니다. 기본폰트 사용 중.")

# -----------------------
# 2. Streamlit 제목
# -----------------------
st.title("📚 서울시 도서관 이용자 수 분석 및 예측")

# -----------------------
# 3. 자치구별 이용자 수 데이터 로드
# -----------------------
@st.cache_data
def load_user_data():
    df = pd.read_excel("서울시 공공도서관 서울도서관 이용자 현황 전처리 완료 파일.xlsx", sheet_name="최신 이용자")
    df = df[['실거주', '이용자수']].copy()
    df.columns = ['구', '이용자수']
    df['이용자수'] = pd.to_numeric(df['이용자수'], errors='coerce')
    df.dropna(inplace=True)
    df = df[df['구'].str.endswith('구')]
    return df

df_users = load_user_data()

# -----------------------
# 4. 바 차트 시각화
# -----------------------
st.subheader("📊 자치구별 도서관 이용자 수")
df_sorted = df_users.sort_values(by="이용자수", ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(df_sorted['구'], df_sorted['이용자수'], color='skyblue')

ax.set_title("📌 자치구별 이용자 수", fontsize=16, fontproperties=font_prop)
ax.set_xlabel("자치구", fontproperties=font_prop)
ax.set_ylabel("이용자 수", fontproperties=font_prop)

ax.set_xticks(range(len(df_sorted)))
ax.set_xticklabels(df_sorted['구'], rotation=45, fontproperties=font_prop)
ax.set_yticklabels(ax.get_yticks(), fontproperties=font_prop)

st.pyplot(fig)

# -----------------------
# 5. 지도 시각화
# -----------------------
st.subheader("🗺️ 자치구별 도서관 이용자 수 지도")

sample_locations = {
    "강남구": [37.5172, 127.0473], "강동구": [37.5301, 127.1238], "강북구": [37.6396, 127.0256],
    "강서구": [37.5509, 126.8495], "관악구": [37.4784, 126.9516], "광진구": [37.5385, 127.0823],
    "구로구": [37.4954, 126.8874], "금천구": [37.4569, 126.8958], "노원구": [37.6542, 127.0568],
    "도봉구": [37.6688, 127.0470], "동대문구": [37.5744, 127.0396], "동작구": [37.5124, 126.9392],
    "마포구": [37.5663, 126.9014], "서대문구": [37.5791, 126.9368], "서초구": [37.4836, 127.0326],
    "성동구": [37.5633, 127.0367], "성북구": [37.5894, 127.0167], "송파구": [37.5145, 127.1056],
    "양천구": [37.5169, 126.8664], "영등포구": [37.5264, 126.8963], "용산구": [37.5326, 126.9903],
    "은평구": [37.6176, 126.9227], "종로구": [37.5731, 126.9795], "중구": [37.5636, 126.9976],
    "중랑구": [37.6063, 127.0927]
}
m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
min_val, max_val = df_users['이용자수'].min(), df_users['이용자수'].max()

for _, row in df_users.iterrows():
    gu = row['구']
    if gu in sample_locations:
        val = row['이용자수']
        folium.CircleMarker(
            location=sample_locations[gu],
            radius=5 + 15 * (val - min_val) / (max_val - min_val),
            popup=f"{gu}: {int(val):,}명",
            color='blue', fill=True, fill_opacity=0.6
        ).add_to(m)

folium_static(m)

# -----------------------
# 6. 최다 이용 구 출력
# -----------------------
top_gu = df_sorted.iloc[0]
st.success(f"✅ 가장 도서관 이용자 수가 많은 구는 **`{top_gu['구']}`**, 총 **`{int(top_gu['이용자수']):,}명`** 입니다.")

# -----------------------
# 7. 머신러닝 준비 및 결과
# -----------------------
@st.cache_data
def load_ml_data():
    df = pd.read_csv("공공도서관 자치구별 통계 파일.csv", encoding='cp949', header=1)
    df = df[df.iloc[:,0] != '소계']
    df.columns = [
        '자치구명','개소수','좌석수','자료수_도서','자료수_비도서','자료수_연속간행물',
        '도서관 방문자수','연간대출책수','직원수','직원수_남','직원수_여','예산'
    ]
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    return df

try:
    df_stat = load_ml_data()
    st.subheader("📄 통계 데이터 미리보기")
    st.dataframe(df_stat)

    X = df_stat.drop(columns=['자치구명','도서관 방문자수'])
    y = df_stat['도서관 방문자수']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.markdown(f"✅ **MSE**: `{mse:,.0f}`  |  **R²**: `{r2:.4f}`")

    st.subheader("🔍 변수 중요도 분석")
    importance = pd.Series(model.feature_importances_, index=X.columns)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot.barh(ax=ax2, color='skyblue')

    ax2.set_title("📌 RandomForest 변수 중요도", fontsize=16, fontproperties=font_prop)
    ax2.set_xlabel("중요도", fontproperties=font_prop)
    ax2.set_ylabel("변수 이름", fontproperties=font_prop)
    ax2.set_yticklabels(importance.sort_values().index, fontproperties=font_prop)
    ax2.set_xticklabels(ax2.get_xticks(), fontproperties=font_prop)

    st.pyplot(fig2)

except Exception as e:
    st.error(f"❌ 오류 발생: {e}")
