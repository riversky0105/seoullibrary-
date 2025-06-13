import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 🔄 캐시 초기화
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("data.csv")  # 데이터 파일 경로 수정
    return df

# ✅ 한글 폰트 적용 함수
def set_korean_font():
    font_path = os.path.join(os.getcwd(), "NanumGothicCoding.ttf")
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()

        mpl.rcParams['font.family'] = font_name
        mpl.rcParams['axes.unicode_minus'] = False
        plt.rc('font', family=font_name)
        plt.rcParams['font.family'] = font_name

        fm._rebuild()
        fm.fontManager.addfont(font_path)

        return font_prop  # ⬅️ 반환하여 그래프에 직접 적용 가능
    else:
        st.warning("⚠️ 'NanumGothicCoding.ttf' 파일을 찾을 수 없습니다.")
        return None

# ✅ 페이지 초기화
st.set_page_config(layout="wide")
st.title("📊 서울시 도서관 분석 대시보드")

# ✅ 데이터 로드
df = load_data()
st.dataframe(df)

# ✅ 한글 폰트 적용
font_prop = set_korean_font()

# ✅ 모델 학습
X = df.drop(columns=["대출건수"])
y = df["대출건수"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ✅ 모델 성능 지표 출력
st.markdown(f"✅ **MSE**: <span style='color:green'>{mse:,.0f}</span> | **R²**: <span style='color:#2E8B57'>{r2:.4f}</span>", unsafe_allow_html=True)

# ✅ 변수 중요도 시각화
st.subheader("🔍 변수 중요도 분석")
importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
ax.invert_yaxis()

# ✅ 그래프 한글 설정 적용
if font_prop:
    ax.set_title("📌 RandomForest 변수 중요도", fontsize=16, fontproperties=font_prop)
    ax.set_xlabel("중요도", fontsize=12, fontproperties=font_prop)
    ax.set_ylabel("변수 이름", fontsize=12, fontproperties=font_prop)

st.pyplot(fig)

# ✅ 캐시 초기화 버튼
if st.button("🔄 캐시 초기화 및 새로고침"):
    st.cache_data.clear()
    st.experimental_rerun()



