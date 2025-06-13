import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows 환경 기준, 다른 환경 시 폰트 경로 수정)
font_path = "C:/Windows/Fonts/malgun.ttf"
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# 페이지 제목
st.set_page_config(page_title="도서관 방문자 수 예측 시스템", layout="wide")
st.title("📚 머신러닝 기반 도서관 방문자 수 예측")

# CSV 업로드
st.sidebar.header("📂 데이터 파일 업로드")
uploaded_file = st.sidebar.file_uploader("CSV 파일을 선택하세요.", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📖 데이터 미리보기")
    st.dataframe(df)

    # 도서관 방문자수 컬럼 존재 여부 확인
    if '도서관 방문자수 (명)' not in df.columns:
        st.error("❌ '도서관 방문자수 (명)' 컬럼이 데이터에 없습니다. CSV 파일을 확인하세요.")
    else:
        # 📊 서울시 구별 도서관 방문자 수 시각화
        st.header("🏙️ 서울시 구별 도서관 방문자 수")

        if '자치구별(2)' in df.columns:
            gu_visit = df.groupby('자치구별(2)')['도서관 방문자수 (명)'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = sns.barplot(x=gu_visit.index, y=gu_visit.values, palette="viridis", ax=ax)
            ax.set_title("서울시 구별 도서관 방문자 수", fontsize=16)
            ax.set_xlabel("자치구", fontsize=12)
            ax.set_ylabel("방문자 수 (명)", fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
        else:
            st.warning("'자치구별(2)' 컬럼이 없어 구별 방문자 수 시각화를 진행할 수 없습니다.")

        # 🎯 머신러닝 예측 모델 학습
        st.header("🤖 도서관 방문자 수 예측")

        feature_cols = ['개소 (개)', '좌석수 (개)', '자료수 (권)', '자료수 (권).1', '자료수 (권).2',
                        '연간대출 책수 (권)', '직원수 (명)', '직원수 (명).1', '직원수 (명).2', '예산 (백만원)']

        available_features = [col for col in feature_cols if col in df.columns]

        if available_features:
            X = df[available_features]
            y = df['도서관 방문자수 (명)']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)

            st.write(f"✅ **RMSE:** {rmse:,.0f}")
            st.write(f"✅ **R² Score:** {r2:.3f}")

            # 📌 변수 중요도 시각화
            st.subheader("🔎 변수 중요도")

            importances = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': available_features, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=True)

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            bars2 = ax2.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
            ax2.set_xlabel('중요도 (Importance)', fontsize=12)
            ax2.set_ylabel('특징 (Feature)', fontsize=12)
            ax2.set_title('📊 RandomForest 변수 중요도', fontsize=14)
            st.pyplot(fig2)
        else:
            st.error("❗ 예측에 필요한 특성 컬럼이 충분하지 않습니다. CSV 파일을 확인하세요.")
else:
    st.warning("📎 CSV 파일을 먼저 업로드 해주세요.")





