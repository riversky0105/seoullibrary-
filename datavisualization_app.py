import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import platform
import matplotlib.font_manager as fm

# ✅ 한글 폰트 설정 함수 (강제 등록 포함)
def set_korean_font():
    try:
        if platform.system() == 'Windows':
            font_path = 'C:/Windows/Fonts/malgun.ttf'
        elif platform.system() == 'Darwin':
            font_path = '/System/Library/Fonts/AppleGothic.ttf'
        else:
            # Ubuntu 등 리눅스 계열
            font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False
        print(f"✅ 한글 폰트 설정 완료: {font_name}")
    except Exception as e:
        print("⚠️ 한글 폰트 설정 실패:", e)

set_korean_font()

# ✅ 예시 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_excel(
        "서울시 공공도서관 서울도서관 이용자 현황 전처리 완료 파일.xlsx",
        sheet_name="최신 이용자"
    )
    df = df[['실거주', '이용자수']].copy()
    df.columns = ['구', '이용자수']
    df = df[df['구'].str.endswith('구')]
    df.dropna(inplace=True)
    df['이용자수'] = pd.to_numeric(df['이용자수'], errors='coerce')
    return df

df = load_data()
df_sorted = df.sort_values(by="이용자수", ascending=False)

# ✅ 그래프 그리기
st.title("📊 서울시 도서관 이용자 수")
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(df_sorted['구'], df_sorted['이용자수'], color='skyblue')
ax.set_ylabel("도서관 이용자 수")
ax.set_xlabel("자치구")
plt.xticks(rotation=45)
st.pyplot(fig)

