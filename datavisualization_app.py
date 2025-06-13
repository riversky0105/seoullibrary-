import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib as mpl

# -----------------------
# 한글 폰트 설정 (깨짐 방지)
# -----------------------
font_path = "C:/Windows/Fonts/malgun.ttf"  # 윈도우 기본 한글 폰트 경로
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# -----------------------
# 머신러닝 예측 + 시각화
# -----------------------
st.subheader("🤖 머신러닝 기반 도서관 방문자 수 예측")

@st.cache_data
def load_ml_data():
    file_path = "공공도서관 자치구별 통계 파일.csv"
    df = pd.read_csv(file_path, encoding='cp949', header=1)
    
    # '소계' 행 제거
    df = df[df.iloc[:,0] != '소계']
    
    # 컬럼명 설정
    df.columns = [
        '자치구명', '개소수', '좌석수', '자료수_도서', '자료수_비도서', '자료수_연속간행물',
        '도서관 방문자수', '연간대출책수', '직원수', '직원수_남', '직원수_여', '예산'
    ]
    
    # 숫자형 변환
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        
    return df

try:
    df_stat = load_ml_data()

    # 입력/출력 나누기
    X = df_stat.drop(columns=['자치구명', '도서관 방문자수'])
    y = df_stat['도서관 방문자수']
    
    # 학습 및 예측
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 성능 지표 출력
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.markdown(f"📊 **평균 제곱 오차 (MSE): `{mse:,.0f}`**")
    st.markdown(f"📈 **결정계수 (R²): `{r2:.4f}`**")
    
    # 변수 중요도 시각화 (한글 축 추가, 폰트 설정)
    st.subheader("🔍 변수 중요도")
    importance = pd.Series(model.feature_importances_, index=X.columns)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot(kind='barh', ax=ax2, color='skyblue')
    ax2.set_title("📌 RandomForest 변수 중요도", fontsize=16, pad=15)
    ax2.set_xlabel("중요도", fontsize=12)
    ax2.set_ylabel("변수 이름", fontsize=12)
    
    st.pyplot(fig2)
    
except Exception as e:
    st.error(f"❌ 오류 발생: {e}")


