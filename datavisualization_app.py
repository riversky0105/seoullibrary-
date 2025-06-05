import streamlit as st
import pandas as pd
import plotly.express as px

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_excel("서울시 공공도서관 서울도서관 이용자 현황.xlsx", sheet_name="최신 이용자")
    df_clean = df.iloc[2:, [1, 2]]  # 구 이름과 이용자 수만 추출
    df_clean.columns = ['District', 'Users']
    df_clean = df_clean.dropna()
    df_clean['Users'] = df_clean['Users'].astype(int)
    return df_clean

# 가장 이용자 수가 많은 도서관 정보 불러오기
@st.cache_data
def load_top_libraries():
    df_full = pd.read_excel("서울시 공공도서관 서울도서관 이용자 현황.xlsx", sheet_name="최신 이용자", usecols="B,C,E")
    df_full.columns = ['District', 'Users', 'Library']
    df_full = df_full.dropna()
    df_full['Users'] = pd.to_numeric(df_full['Users'], errors='coerce')
    df_full = df_full.dropna()

    top_libs = df_full.groupby('District').apply(lambda g: g.loc[g['Users'].idxmax()]).reset_index(drop=True)
    return top_libs

# 메인 실행
st.title("📚 서울시 도서관 이용자 수 분석")
df = load_data()
top_libraries = load_top_libraries()

# 지도 시각화
st.subheader("서울시 자치구별 도서관 이용자 수")
fig = px.choropleth(
    df,
    geojson="https://raw.githubusercontent.com/southkorea/southkorea-maps/master/kostat/2013/json/skorea_municipalities_geo_simple.json",
    featureidkey="properties.name",
    locations="District",
    color="Users",
    color_continuous_scale="Blues",
    scope="asia",
    title="서울시 자치구별 도서관 이용자 수"
)
fig.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig, use_container_width=True)

# 도서관 정보 표시
st.subheader("구별 최다 이용 도서관")
for _, row in top_libraries.iterrows():
    st.markdown(f"**{row['District']}**: {row['Library']} ({int(row['Users'])}명)")




