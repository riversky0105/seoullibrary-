# -----------------------
# 4. 자치구별 도서관 이용자 수 시각화 (CSV 기반으로 수정됨)
# -----------------------
st.subheader("📊 자치구별 도서관 이용자 수")

# 🔹 데이터 준비
df_users = df_stat[['자치구명', '도서관 방문자수']].copy()
df_users.columns = ['구', '이용자수']
df_users['이용자수'] = df_users['이용자수'].astype(int)  # ✅ 소수점 제거
df_sorted = df_users.sort_values(by="이용자수", ascending=False)

# 🔹 데이터 확인 테이블 추가
with st.expander("📄 이용자 수 데이터 확인"):
    st.dataframe(df_sorted)

# 🔹 시각화
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(df_sorted['구'], df_sorted['이용자수'], color='skyblue')

ax.set_title("📌 자치구별 이용자 수", fontsize=16, fontproperties=font_prop)
ax.set_xlabel("자치구", fontproperties=font_prop)
ax.set_ylabel("이용자 수", fontproperties=font_prop)
ax.set_xticks(range(len(df_sorted)))
ax.set_xticklabels(df_sorted['구'], rotation=45, fontproperties=font_prop)
ax.set_yticklabels([f"{int(y):,}" for y in ax.get_yticks()], fontproperties=font_prop)  # ✅ y축 정수 표시

st.pyplot(fig)
