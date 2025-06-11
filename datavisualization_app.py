import pandas as pd
import numpy as np
import platform
import matplotlib.pyplot as plt
import folium

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import streamlit as st
from streamlit_folium import folium_static

# -------------------
# 한글 폰트 설정
# -------------------
def set_korean_font():
    if platform.system() == 'Windows':
        plt.rc('font', family='Malgun Gothic')
    elif platform.system() == 'Darwin':
        plt.rc('font', family='AppleGothic')
    else:
        plt.rc('font', family='NanumGothic')
    plt.rc('axes', unicode_minus=False)

set_korean_font()

# -------------------
# 데이터 로드
# -------------------
@st.cache_data
def load_data():
    df = pd.read_excel("서울시 공공도서관 서울도서관 이용자 현황 전처리 완료 파일.xlsx", sheet_name="최신 이용자")
    df = df[['실거주', '이용자수']].copy()
    df.columns = ['구', '이용자수']
    df.dropna(inplace=True)
    df['이용자수'] = pd.to_numeric(df['이용자수'], errors='coerce')
    df = df[df['구'].str.endswith('구')]
    return df

df = load_data()

# -------------------
# Streamlit UI 시작
# -------------------
st.set_page_config(page_title="서울시 도서관 이용자 수 분석", layout="wide")
st.title("📚 서울시 도서관 이용자 수 분석")
st.markdown("**전처리된 데이터를 기반으로 한 자치구별 도서관 이용자 현황입니다.**")

# -------------------
# 요약 정보
# -------------------
col1, col2 = st.columns(2)
col1.metric("총 자치구 수", df['구'].nunique())
col2.metric("총 이용자 수", f"{int(df['이용자수'].sum()):,}명")

# -------------------
# 바 차트 시각화
# -------------------
st.subheader("📊 자치구별 도서관 이용자 수")
df_sorted = df.sort_values(by="이용자수", ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(df_sorted['구'], df_sorted['이용자수'], color='skyblue')
ax.set_ylabel("이용자 수")
plt.xticks(rotation=45)
st.pyplot(fig)

# -------------------
# 지도 시각화
# -------------------
st.subheader("🗺️ 자치구별 이용자 수 지도")

# 서울시 자치구 좌표
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

m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

def normalize(val, min_val, max_val):
    return 5 + 15 * ((val - min_val) / (max_val - min_val))

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
# 최다 이용 구 요약
# -------------------
top_gu = df_sorted.iloc[0]
st.success(f"✅ **가장 도서관 이용자 수가 많은 구는 `{top_gu['구']}`이며, 총 `{int(top_gu['이용자수']):,}명`이 이용했습니다.**")



# -------------------
#머신러닝 코드

# 1. CSV 파일 읽기
file_path = '공공도서관_20250611101301.csv'  # 실제 파일 경로로 수정 필요
df = pd.read_csv(file_path, encoding='utf-8')  # 인코딩 문제 있으면 'cp949' 또는 'utf-8-sig' 사용

# 2. 열 이름 수동 지정
df.columns = [
    '자치구명', '개소 수(계)', '좌석수(계)', '좌석수(도서)', '좌석수(자료열람)', '좌석수(기타)',
    '자료수(도서)', '자료수(비도서)', '도서관 방문자수',
    '직원수(계)', '직원수(남)', '직원수(여)', '예산(백만원)'
]

# 3. 숫자형 컬럼 처리 (쉼표 제거 + 숫자 변환)
for col in df.columns[1:]:
    df[col] = df[col].astype(str).str.replace(',', '').str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 4. 결측값 처리 (NaN → 0)
df.fillna(0, inplace=True)

# 5. 입력 변수(X), 타겟 변수(y) 정의
X = df.drop(columns=['자치구명', '도서관 방문자수'])  # 자치구명은 학습에 사용하지 않음
y = df['도서관 방문자수']

# 6. 학습용/테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. 랜덤 포레스트 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8. 예측 및 성능 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 평균 제곱 오차(MSE): {mse:.2f}")
print(f"📈 결정계수(R²): {r2:.4f}")

# 9. 중요 변수 확인
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
print("\n🔍 변수 중요도:")
print(feature_importance.sort_values(ascending=False))

