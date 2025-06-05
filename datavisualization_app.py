import streamlit as st
import pandas as pd
import altair as alt

st.title("📚 서울시 자치구별 도서관 이용자 수 분석")

@st.cache_data
def load_data():
    df = pd.read_excel("서울시 공공도서관 서울도서관 이용자 현황.xlsx", sheet_name=0)

    # 첫 번째 행이 실제 컬럼명일 가능성 있을 경우 처리
    if "자치구" not in df.columns:
        df.columns = df.iloc[0]
        df = df[1:]

    # 컬럼명을 문자열로 변환 (NaN 방지)
    df.columns = df.columns.map(str).str.strip()

    # "이용자" 들어간 컬럼 자동 탐색
    usage_col = None
    for col in df.columns:
        if "이용자" in col:
            usage_col = col
            break

    if usage_col is None:
        st.error("❌ '이용자'가 포함된 컬럼을 찾을 수 없습니다.")
        st.write("사용 가능한 컬럼들:", list(df.columns))
        st.stop()

    # 컬럼명 통일
    df = df.rename(columns={usage_col: "총이용자수"})
    df["총이용자수"] = pd.to_numeric(df["총이용자수"], errors="coerce")
    df = df.dropna(subset=["총이용자수", "자치구"])
    return df

df = load_data()
df_gu = df.groupby("자치구")["총이용자수"].sum().reset_index()

st.subheader("📊 자치구별 도서관 이용자 수")
chart = alt.Chart(df_gu).mark_bar().encode(
    x=alt.X("총이용자수:Q", title="이용자 수"),
    y=alt.Y("자치구:N", sort="-x", title="자치구"),
    tooltip=["자치구", "총이용자수"]
).properties(width=700, height=500)

st.altair_chart(chart)



