import io
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ============================================================
# 0. Streamlit 기본 설정
# ============================================================
st.set_page_config(page_title="극지식물 생육 분석", page_icon="🌱", layout="wide")


# ============================================================
# 1. 한글 폰트 CSS
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 2. CSV 업로드 / 인코딩 처리
# ============================================================
def read_csv_file(file):
    """file.getvalue()로 바이트를 꺼내 utf-8-sig → cp949 순서로 읽기"""
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=enc)
            return df
        except Exception:
            pass
    return pd.DataFrame()


# ============================================================
# 3. 열 이름 표준화 (KeyError 방지) — 지시된 std_cols 그대로
# ============================================================
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


# ============================================================
# 4. 학교명 추출 (파일명에서)
# ============================================================
def extract_school_name(filename):
    if filename is None:
        return "OO고"
    name = os.path.basename(filename)
    # 확장자 제거
    name = name.rsplit(".", 1)[0]
    # 언더스코어 앞부분까지만
    if "_" in name:
        name = name.split("_")[0]
    return name


# ============================================================
# 5. CSV 다운로드 헬퍼 (utf-8-sig)
# ============================================================
def make_csv_download(df, label):
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    return st.download_button(
        label=f"📥 {label} 다운로드 (utf-8-sig)",
        data=csv_bytes,
        file_name=f"{label}.csv",
        mime="text/csv",
    )


# ============================================================
# 6. 개체번호가 아닌 지표만 골라내는 헬퍼
# ============================================================
def numeric_cols_for_analysis(df):
    """개체번호·time을 제외한 숫자 열만 반환"""
    skip = {"개체번호", "time"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


# ============================================================
# 7. Numpy 추세선 계산 (px.scatter trendline 대신)
# ============================================================
def add_trendline(fig, x, y, color="red", dash="dash"):
    """go.Scatter로 추세선 추가. 값이 2개 미만이면 아무것도 안 함."""
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 2:
        return
    m, b = np.polyfit(clean["x"], clean["y"], 1)
    x_seq = clean["x"].sort_values().values
    fig.add_trace(
        go.Scatter(
            x=x_seq,
            y=m * x_seq + b,
            mode="lines",
            line=dict(color=color, dash=dash, width=2),
            name=f"추세선 (r)",
            showlegend=True,
        )
    )


# ============================================================
# 8. 상관계수 계산 헬퍼
# ============================================================
def safe_corr(df, col1, col2):
    clean = df[[col1, col2]].dropna()
    if len(clean) < 2:
        return None
    return float(clean[col1].corr(clean[col2]))


# ============================================================
# 9. 값이 충분히 남았는지 체크
# ============================================================
def enough_vals(series):
    return series.dropna().shape[0] >= 2


# ============================================================
# 사이드바: 파일 업로드
# ============================================================
st.sidebar.title("📁 CSV 업로드")
env_file = st.sidebar.file_uploader("🌡️ 환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("🌿 생육 결과 데이터 CSV", type="csv")

school_name = extract_school_name(env_file.name if env_file else None)
st.sidebar.success(f"학교명: {school_name} (파일명에서 자동 추출)")

# 데이터 로딩 (캐시 + spinner)
@st.cache_data
def load_and_std(env_file, growth_file):
    env_df = read_csv_file(env_file)
    growth_df = read_csv_file(growth_file)
    if not env_df.empty:
        env_df = std_cols(env_df)
    if not growth_df.empty:
        growth_df = std_cols(growth_df)
    return env_df, growth_df


with st.sidebar.spinner("데이터 읽는 중..."):
    env_df, growth_df = load_and_std(env_file, growth_file)

st.sidebar.info(
    "📌 업로드 전에는 탭이 안내 문구만 보여줍니다.\n"
    "📌 한 파일만 올려도 해당 탭은 동작합니다."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📏 생육 측정 항목 선택")
growth_num_cols = numeric_cols_for_analysis(growth_df)
if growth_num_cols:
    selected_items = st.sidebar.multiselect(
        "비교할 측정 항목 (개체번호 제외)",
        options=growth_num_cols,
        default=growth_num_cols[:3] if len(growth_num_cols) >= 3 else growth_num_cols,
    )
else:
    selected_items = []
    st.sidebar.warning("생육 CSV에서 숫자 열을 찾지 못했습니다.")

# ============================================================
# 탭 구성
# ============================================================
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ============================================================
# TAB 1 — 실험 개요
# ============================================================
with tab1:
    try:
        st.header("📖 실험 개요")

        # 연구 배경 및 목적
        st.subheader("🔬 연구 배경 및 목적")
        st.markdown(
            f"""
**{school_name} 극지식물 생육 차이 분석**

- **배경**: 같은 환경에서 자란 극지식물이라도 개체마다 생중량·잎 수·줄기 길이 등에 차이가 있습니다.
- **목적**: 개체별 생육 데이터를 비교해 어떤 개체가 잘 자랐는지, 어떤 측정 항목이 생중량과 관련 있는지 살펴봅니다.
- **통계 개념**:
  - **평균**: 모든 값을 더해 개수로 나눈 대표값
  - **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 숫자 (작을수록 평균 근처에 몰려 있음)
  - **상관계수(r)**: 두 변수가 함께 움직이는 정도 (-1 ~ 1). 1에 가까울수록 함께 증가, -1에 가까울수록 반대 방향
"""
        )

        # 데이터 요약 표
        st.subheader("📋 데이터 요약")
        summary_rows = []

        # 개체 수
        if not growth_df.empty and "개체번호" in growth_df.columns:
            n_plants = growth_df["개체번호"].dropna().nunique()
            summary_rows.append(
                {"항목": "총 개체 수", "값": f"{n_plants}개체"}
            )
        else:
            summary_rows.append({"항목": "총 개체 수", "값": "데이터 없음"})

        # 측정 기간 (환경)
        if not env_df.empty and "time" in env_df.columns:
            t_series = env_df["time"].dropna()
            if enough_vals(t_series):
                summary_rows.append(
                    {
                        "항목": "측정 기간",
                        "값": f"{t_series.min().strftime('%Y-%m-%d')} ~ {t_series.max().strftime('%Y-%m-%d')}",
                    }
                )
            else:
                summary_rows.append({"항목": "측정 기간", "값": "날짜 정보 부족"})
        else:
            summary_rows.append({"항목": "측정 기간", "값": "환경 데이터 없음"})

        # 측정 항목 (생육)
        if growth_num_cols:
            summary_rows.append(
                {"항목": "생육 측정 항목", "값": ", ".join(growth_num_cols)}
            )
        else:
            summary_rows.append({"항목": "생육 측정 항목", "값": "없음"})

        # 측정 항목 (환경)
        env_num_cols = [
            c
            for c in ["temperature", "humidity", "ph", "ec"]
            if c in env_df.columns and pd.api.types.is_numeric_dtype(env_df[c])
        ]
        if env_num_cols:
            summary_rows.append(
                {"항목": "환경 측정 항목", "값": ", ".join(env_num_cols)}
            )
        else:
            summary_rows.append({"항목": "환경 측정 항목", "값": "없음"})

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        # 주요 지표 카드 4개
        st.subheader("📊 주요 지표")

        col1, col2, col3, col4 = st.columns(4)

        # 총 개체수
        if not growth_df.empty and "개체번호" in growth_df.columns:
            n_plants = growth_df["개체번호"].dropna().nunique()
            col1.metric("🌱 총 개체 수", f"{n_plants}개체")
        else:
            col1.metric("🌱 총 개체 수", "—")

        # 평균 온도
        if "temperature" in env_df.columns and enough_vals(env_df["temperature"]):
            avg_temp = float(env_df["temperature"].dropna().mean())
            col2.metric("🌡️ 평균 온도", f"{avg_temp:.1f}℃")
        else:
            col2.metric("🌡️ 평균 온도", "데이터 없음")

        # 평균 습도
        if "humidity" in env_df.columns and enough_vals(env_df["humidity"]):
            avg_hum = float(env_df["humidity"].dropna().mean())
            col3.metric("💧 평균 습도", f"{avg_hum:.1f}%")
        else:
            col3.metric("💧 평균 습도", "데이터 없음")

        # 평균 생중량
        if "생중량" in growth_df.columns and enough_vals(growth_df["생중량"]):
            avg_fw = float(growth_df["생중량"].dropna().mean())
            col4.metric("⚖️ 평균 생중량", f"{avg_fw:.2f}g")
        else:
            col4.metric("⚖️ 평균 생중량", "데이터 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ============================================================
# TAB 2 — 환경 데이터
# ============================================================
with tab2:
    try:
        st.header("🌡️ 환경 데이터")

        if env_df.empty:
            st.info("🌡️ 환경 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            # 환경 수치형 열 목록 (ph·ec가 없는 학교 대응)
            env_num_cols = [
                c
                for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns and pd.api.types.is_numeric_dtype(env_df[c])
            ]

            if not env_num_cols:
                st.info("🌡️ 환경 데이터에서 숫자 측정 열을 찾지 못했습니다.")
            else:
                # 환경 요약 표
                st.subheader("📋 환경 요약 (평균·최소·최대)")
                env_summary_rows = []
                for col in env_num_cols:
                    s = env_df[col].dropna()
                    if len(s) == 0:
                        continue
                    env_summary_rows.append(
                        {
                            "측정 항목": col,
                            "평균": f"{s.mean():.2f}",
                            "최소": f"{s.min():.2f}",
                            "최대": f"{s.max():.2f}",
                            "측정 횟수": f"{len(s)}회",
                        }
                    )
                st.dataframe(
                    pd.DataFrame(env_summary_rows), use_container_width=True
                )

                # 날짜별 시계열 (일별 평균 꺾은선)
                st.subheader("📈 날짜별 시계열 (일별 평균)")

                if "time" in env_df.columns and enough_vals(env_df["time"]):
                    env_df["date"] = env_df["time"].dt.date

                    # 온도 변화
                    if "temperature" in env_df.columns:
                        temp_daily = (
                            env_df.groupby("date")["temperature"]
                            .mean()
                            .reset_index()
                            .rename(columns={"temperature": "평균 온도"})
                        )
                        temp_daily["date"] = pd.to_datetime(temp_daily["date"])

                        fig_t = go.Figure()
                        fig_t.add_trace(
                            go.Scatter(
                                x=temp_daily["date"],
                                y=temp_daily["평균 온도"],
                                mode="lines+markers",
                                name="평균 온도",
                                line=dict(color="red", width=2),
                                marker=dict(size=5),
                            )
                        )
                        fig_t.update_layout(
                            title="🌡️ 일별 평균 온도",
                            xaxis_title="날짜",
                            yaxis_title="온도 (℃)",
                            height=350,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        )
                        st.plotly_chart(fig_t, width="stretch")
                        plt_close_t = True
                    else:
                        st.info("🌡️ 온도 열이 없어 온도 그래프를 표시할 수 없습니다.")

                    # 습도 변화
                    if "humidity" in env_df.columns:
                        hum_daily = (
                            env_df.groupby("date")["humidity"]
                            .mean()
                            .reset_index()
                            .rename(columns={"humidity": "평균 습도"})
                        )
                        hum_daily["date"] = pd.to_datetime(hum_daily["date"])

                        fig_h = go.Figure()
                        fig_h.add_trace(
                            go.Scatter(
                                x=hum_daily["date"],
                                y=hum_daily["평균 습도"],
                                mode="lines+markers",
                                name="평균 습도",
                                line=dict(color="blue", width=2),
                                marker=dict(size=5),
                            )
                        )
                        fig_h.update_layout(
                            title="💧 일별 평균 습도",
                            xaxis_title="날짜",
                            yaxis_title="습도 (%)",
                            height=350,
                            font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        )
                        st.plotly_chart(fig_h, width="stretch")
                    else:
                        st.info("💧 습도 열이 없어 습도 그래프를 표시할 수 없습니다.")

                    # pH·EC 변화 (열이 있을 때만)
                    for col, title, color in [
                        ("ph", "🧪 일별 평균 pH", "purple"),
                        ("ec", "⚡ 일별 평균 EC", "orange"),
                    ]:
                        if col in env_df.columns:
                            daily = (
                                env_df.groupby("date")[col]
                                .mean()
                                .reset_index()
                            )
                            daily["date"] = pd.to_datetime(daily["date"])

                            fig_e = go.Figure()
                            fig_e.add_trace(
                                go.Scatter(
                                    x=daily["date"],
                                    y=daily[col],
                                    mode="lines+markers",
                                    name=title,
                                    line=dict(color=color, width=2),
                                    marker=dict(size=5),
                                )
                            )
                            fig_e.update_layout(
                                title=title,
                                xaxis_title="날짜",
                                yaxis_title=col.upper(),
                                height=350,
                                font=dict(
                                    family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                                ),
                            )
                            st.plotly_chart(fig_e, width="stretch")
                else:
                    st.info("📅 날짜(time) 열이 없거나 유효한 날짜가 부족해 시계열을 표시할 수 없습니다.")

                # 원본 테이블 + 다운로드
                with st.expander("📄 환경 데이터 원본 테이블 보기"):
                    st.dataframe(env_df, width="stretch", height=300)
                    make_csv_download(env_df, f"{school_name}_환경데이터")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ============================================================
# TAB 3 — 생육 결과
# ============================================================
with tab3:
    try:
        st.header("📊 생육 결과")

        if growth_df.empty:
            st.info("🌿 생육 CSV를 업로드하면 이 탭의 내용이 표시됩니다.")
        else:
            # 생중량 열이 있는지 확인
            has_fresh = "생중량" in growth_df.columns and pd.api.types.is_numeric_dtype(
                growth_df["생중량"]
            )

            if not has_fresh:
                st.info(
                    "⚖️ '생중량' 열이 없습니다. 지상부 질량 + 지하부 질량으로 생중량을 계산한 뒤 다시 시도해 주세요."
                )
            else:
                fw = growth_df["생중량"].dropna()

                # ── 핵심 결과 카드 ──
                st.subheader("🥇 핵심 결과 카드")

                c_avg, c_max, c_min = st.columns(3)

                avg_fw = float(fw.mean())
                c_avg.metric("📊 평균 생중량", f"{avg_fw:.2f}g")

                if len(fw) > 0:
                    max_idx = fw.idxmax()
                    min_idx = fw.idxmin()
                    plant_id_col = growth_df["개체번호"]

                    max_id = plant_id_col.loc[max_idx] if max_idx in plant_id_col.index else "?"
                    min_id = plant_id_col.loc[min_idx] if min_idx in plant_id_col.index else "?"

                    # 개체번호가 float로 표시되는 것 방지 (원래 타입 그대로)
                    if isinstance(max_id, float) and max_id == int(max_id):
                        max_id_str = str(int(max_id))
                    else:
                        max_id_str = str(max_id)

                    if isinstance(min_id, float) and min_id == int(min_id):
                        min_id_str = str(int(min_id))
                    else:
                        min_id_str = str(min_id)

                    c_max.metric(
                        "🏆 최대 생중량 개체",
                        f"{max_id_str}번: {fw.max():.2f}g",
                    )
                    c_min.metric(
                        "📉 최소 생중량 개체",
                        f"{min_id_str}번: {fw.min():.2f}g",
                    )
                else:
                    c_max.metric("🏆 최대 생중량 개체", "데이터 없음")
                    c_min.metric("📉 최소 생중량 개체", "데이터 없음")

                # ── 개체별 생육 비교 (막대 그래프) ──
                st.subheader("📊 개체별 생육 비교 (막대 그래프)")

                growth_numeric_cols = [
                    c
                    for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                    if c in growth_df.columns and pd.api.types.is_numeric_dtype(
                        growth_df[c]
                    )
                ]

                if not growth_numeric_cols:
                    st.info("측정 항목(잎 수, 지상부 길이 등) 숫자 열이 없습니다.")
                else:
                    # 선택한 항목이 있으면 그것만, 없으면 기본 4개
                    plot_cols = [
                        c for c in selected_items if c in growth_df.columns
                    ]
                    if not plot_cols:
                        plot_cols = growth_numeric_cols[:4]

                    n_plots = len(plot_cols)
                    if n_plots == 0:
                        st.info("표시할 측정 항목이 없습니다.")
                    else:
                        fig_bar = make_subplots(
                            rows=n_plots,
                            cols=1,
                            subplot_titles=plot_cols,
                            vertical_spacing=0.12,
                        )

                        for i, col in enumerate(plot_cols):
                            vals = growth_df[col].dropna()
                            if len(vals) == 0:
                                continue
                            # 개체번호 유지 (float 방지)
                            ids = growth_df.loc[vals.index, "개체번호"]
                            id_list = []
                            for v in ids:
                                if isinstance(v, float) and v == int(v):
                                    id_list.append(int(v))
                                else:
                                    id_list.append(v)

                            is_fw = col == "생중량"
                            color = "gold" if is_fw else "steelblue"

                            fig_bar.add_trace(
                                go.Bar(
                                    x=id_list,
                                    y=vals.values,
                                    name=col,
                                    marker_color=color,
                                    showlegend=False,
                                ),
                                row=i + 1,
                                col=1,
                            )

                        fig_bar.update_layout(
                            height=200 * n_plots + 50,
                            font=dict(
                                family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                            ),
                            margin=dict(l=40, r=20, t=60, b=40),
                        )
                        for i in range(n_plots):
                            fig_bar.update_yaxes(title_text=plot_cols[i], row=i + 1, col=1)

                        st.plotly_chart(fig_bar, width="stretch")

                # ── 생중량 분포 (박스플롯) ──
                if has_fresh and len(fw) > 0:
                    st.subheader("📦 생중량 분포 (박스플롯)")
                    fig_box = go.Figure()
                    fig_box.add_trace(
                        go.Box(
                            y=fw.values,
                            name="생중량",
                            fillcolor="lightgreen",
                            boxpoints="outliers",
                            marker=dict(color="darkgreen", size=6),
                            line=dict(color="darkgreen"),
                            showlegend=False,
                        )
                    )
                    fig_box.update_layout(
                        title="생중량 분포 (박스플롯) — 상자: 중간 50%, 선: 중앙값",
                        yaxis_title="생중량 (g)",
                        height=350,
                        font=dict(
                            family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                        ),
                    )
                    st.plotly_chart(fig_box, width="stretch")

                    # 통계 설명
                    std_fw = float(fw.std())
                    cv_fw = float(std_fw / avg_fw) if avg_fw != 0 else 0
                    st.caption(
                        f"""
**분포 통계**
- 개수: {len(fw)}개체
- 평균: {avg_fw:.2f}g
- 표준편차: {std_fw:.2f}g (값이 평균에서 얼마나 흩어져 있는지)
- 변동계수(CV): {cv_fw:.2f} (표준편차 ÷ 평균, 단위 없이 상대적 산포 비교용)
- 최소값: {fw.min():.2f}g  |  최대값: {fw.max():.2f}g
"""
                    )

                # ── 상관관계 분석 ──
                st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")

                corr_pairs = [
                    ("잎 수", "생중량"),
                    ("지상부 길이", "생중량"),
                ]
                corr_pairs = [
                    (a, b) for a, b in corr_pairs if a in growth_df.columns and b in growth_df.columns
                ]

                if not corr_pairs:
                    st.info("상관관계 분석할 측정 항목 쌍이 없습니다.")
                else:
                    for x_col, y_col in corr_pairs:
                        fig_s = go.Figure()

                        x_raw = growth_df[x_col].dropna()
                        y_raw = growth_df.loc[x_raw.index, y_col].dropna()
                        common_idx = x_raw.index.intersection(y_raw.index)
                        x_vals = growth_df.loc[common_idx, x_col]
                        y_vals = growth_df.loc[common_idx, y_col]

                        if len(x_vals) > 0:
                            fig_s.add_trace(
                                go.Scatter(
                                    x=x_vals,
                                    y=y_vals,
                                    mode="markers",
                                    name=f"{x_col} vs {y_col}",
                                    marker=dict(size=8, color="steelblue", opacity=0.7),
                                    showlegend=False,
                                )
                            )
                            add_trendline(fig_s, x_vals, y_vals, color="red", dash="dash")

                        r_val = safe_corr(growth_df, x_col, y_col)
                        r_text = f"r = {r_val:.3f}" if r_val is not None else "r = 계산 불가"

                        fig_s.update_layout(
                            title=f"{x_col} vs {y_col}  |  {r_text}",
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            height=380,
                            font=dict(
                                family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"
                            ),
                        )
                        st.plotly_chart(fig_s, width="stretch")

                        # 상관계수 해석
                        if r_val is not None:
                            if abs(r_val) >= 0.7:
                                strength = "강한"
                            elif abs(r_val) >= 0.4:
                                strength = "중간"
                            else:
                                strength = "약한"

                            direction = "▲ 함께 증가" if r_val > 0 else "▼ 반대 방향"

                            st.caption(
                                f"""
**{x_col} vs {y_col} 상관 해석**
- 상관계수 r = {r_val:.3f} → **{strength} {direction}** 관계
- r 범위: **-1 ≤ r ≤ 1**
  - 1에 가까움: 함께 증가 (예: 잎 수↑ → 생중량↑)
  - 0에 가까움: 관계 거의 없음
  - -1에 가까움: 한쪽 증가 시 다른 쪽 감소
- **상관 ≠ 인과**예요. 함께 움직인다고 해서 한쪽이 원인이라고 단정할 수는 없습니다.
"""
                            )

                # ── 상위 25% vs 하위 25% 비교 ──
                if has_fresh and len(fw) >= 4:
                    st.subheader("🏅 생중량 상위 25% vs 하위 25% 비교")

                    q75 = float(fw.quantile(0.75))
                    q25 = float(fw.quantile(0.25))

                    top_df = growth_df.loc[fw[fw >= q75].index].copy()
                    bot_df = growth_df.loc[fw[fw <= q25].index].copy()

                    comp_cols = [
                        c
                        for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                        if c in growth_df.columns
                        and pd.api.types.is_numeric_dtype(growth_df[c])
                    ]

                    if comp_cols:
                        comp_rows = []
                        for col in comp_cols:
                            top_vals = top_df[col].dropna()
                            bot_vals = bot_df[col].dropna()
                            comp_rows.append(
                                {
                                    "측정 항목": col,
                                    "상위 25% 평균": f"{top_vals.mean():.2f}"
                                    if len(top_vals)
                                    else "—",
                                    "하위 25% 평균": f"{bot_vals.mean():.2f}"
                                    if len(bot_vals)
                                    else "—",
                                    "상위 개체 수": f"{len(top_df)}개체",
                                    "하위 개체 수": f"{len(bot_df)}개체",
                                }
                            )
                        st.dataframe(
                            pd.DataFrame(comp_rows), use_container_width=True
                        )
                    else:
                        st.info("비교할 측정 항목이 없습니다.")

                # ── 원본 테이블 + 다운로드 ──
                with st.expander("📄 생육 데이터 원본 테이블 보기"):
                    st.dataframe(growth_df, width="stretch", height=300)
                    make_csv_download(growth_df, f"{school_name}_생육결과데이터")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

