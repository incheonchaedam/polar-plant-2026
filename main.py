import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless 서버 대응
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings("ignore")
import streamlit as st


# ============================================================
# 0. 한글 폰트 설정 (Streamlit Cloud · 로컬 모두 대응)
# ============================================================
def setup_korean_font():
    if st.session_state.get("font_setup_done", False):
        return
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf",
        "/usr/share/fonts/truetype/nfk/NFHanaSorizAE.ttf",
        "/usr/share/fonts/truetype/nfk/NFHanaMiso.ttf",
    ]
    fp = None
    for path in candidates:
        if os.path.exists(path):
            fp = fm.FontProperties(fname=path)
            plt.rcParams["font.family"] = fp.get_name()
            break
    if fp is None:
        for f in fm.fontManager.ttflist:
            name = f.name
            if any(k in name for k in ("Nanum", "Noto Sans CJK", "Apple SD Gothic",
                                        "KoPub", "UnBatang", "Baekmuk")):
                plt.rcParams["font.family"] = name
                fp = f
                break
    if fp is None:
        plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False
    st.session_state["font_setup_done"] = True


setup_korean_font()


# ============================================================
# 1. 통계 용어 설명 (고등학생용)
# ============================================================
TERM_NOTE = {
    "평균": "모든 값을 더해 개체로 나눈 대표값.",
    "표준편차": "값들이 평균에서 얼마나 떨어져 있는지. 작을수록 평균 근처에 몰려 있고 클수록 넓게 퍼짐.",
    "변동계수(CV)": "표준편차 ÷ 평균. 단위가 다른 데이터끼리 상대적 산포를 비교할 때 씀.",
    "중앙값": "데이터를 크기순으로 나열했을 때 정가운데 값. 극단값의 영향을 덜 받음.",
    "상관관계": "두 변수가 함께 움직이는 정도. -1 ~ 1 사이이고 1에 가까울수록 함께 증가, -1에 가까울수록 반대 방향.",
    "상관계수 r": "상관관계를 숫자로 나타낸 것. -1 ≤ r ≤ 1. r ≈ 0이면 관계가 거의 없음.",
    "보 조정Interquartile": "데이터를 4등분했을 때 1사분위(Q1)·3사분위(Q3) 사이 구간. 전체 중 가운데 50%.",
    "아웃라이어": "다른 값들과 동떨어져 있는 극단값.",
    "추세선": "산점도에서 점들의 전반적인 방향을 직선으로 근사한 선. np.polyfit으로 계산.",
    "상관계수 ≠ 인과": "두 값이 함께 움직인다고 해서 한쪽이 원인이라고 단정할 수는 없음.",
}


# ============================================================
# 2. CSV 읽기 (utf-8-sig → cp949 → euc-kr)
# ============================================================
def read_csv_file(file):
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ============================================================
# 3. 열 이름 표준화 — 지시된 std_cols 그대로
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
# 4. 학교명 추출
# ============================================================
def extract_school_name(filename):
    if filename is None:
        return "OO고"
    name = os.path.basename(filename)
    name = name.rsplit(".", 1)[0]
    if "_" in name:
        name = name.split("_")[0]
    return name


# ============================================================
# 5. 다운로드 버튼 (utf-8-sig)
# ============================================================
def make_csv_download(df, label):
    return st.download_button(
        label=f"📥 {label} 다운로드 (utf-8-sig)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{label}.csv",
        mime="text/csv",
    )


# ============================================================
# 6. 개체번호 제외 숫자 열
# ============================================================
def numeric_cols_for_analysis(df):
    skip = {"개체번호", "time"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


# ============================================================
# 7. trendline (np.polyfit)
# ============================================================
def add_trendline(ax, x, y, color="red", ls="--", lw=2):
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 2:
        return
    m, b = np.polyfit(clean["x"], clean["y"], 1)
    xs = np.sort(clean["x"].values)
    ax.plot(xs, m * xs + b, color=color, linestyle=ls, linewidth=lw,
            label=f"추세선 (r)")


# ============================================================
# 8. 상관계수
# ============================================================
def safe_corr(df, c1, c2):
    clean = df[[c1, c2]].dropna()
    if len(clean) < 2:
        return None
    return float(clean[c1].corr(clean[c2]))


# ============================================================
# 9. 충분한 값 여부
# ============================================================
def enough(series):
    return series.dropna().shape[0] >= 2


# ============================================================
# 사이드바
# ============================================================
st.set_page_config(page_title="극지식물 생육 분석", page_icon="🌱", layout="wide")

st.sidebar.title("📁 CSV 업로드")
env_file = st.sidebar.file_uploader("🌡️ 환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("🌿 생육 결과 데이터 CSV", type="csv")

school_name = extract_school_name(env_file.name if env_file else None)
st.sidebar.success(f"학교명: {school_name} (파일명에서 자동 추출)")


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
gn_cols = numeric_cols_for_analysis(growth_df)
if gn_cols:
    selected_items = st.sidebar.multiselect(
        "비교할 측정 항목 (개체번호 제외)",
        options=gn_cols,
        default=gn_cols[:3] if len(gn_cols) >= 3 else gn_cols,
    )
else:
    selected_items = []
    st.sidebar.warning("생육 CSV에서 숫자 열을 찾지 못했습니다.")


# ============================================================
# 탭
# ============================================================
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])


# ============================================================
# TAB 1 — 실험 개요
# ============================================================
with tab1:
    try:
        st.header("📖 실험 개요")

        st.subheader("🔬 연구 배경 및 목적")
        st.markdown(
            f"""
**{school_name} 극지식물 생육 차이 분석**

- **배경**: 같은 환경에서 자란 극지식물이라도 개체마다 생중량, 잎 수, 줄기 길이 등에 차이가 있습니다.
- **목적**: 개체별 생육 데이터를 비교해 어떤 개체가 잘 자랐는지, 어떤 측정 항목이 생중량과 관련 있는지 살펴봅니다.
- **통계 개념**:
  - **평균**: 모든 값을 더해 개수로 나눈 대표값
  - **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 숫자
  - **상관계수(r)**: 두 변수가 함께 움직이는 정도 (-1 ~ 1)
"""
        )

        st.subheader("📋 데이터 요약")
        rows = []

        if not growth_df.empty and "개체번호" in growth_df.columns:
            n = growth_df["개체번호"].dropna().nunique()
            rows.append({"항목": "총 개체 수", "값": f"{n}개체"})
        else:
            rows.append({"항목": "총 개체 수", "값": "데이터 없음"})

        if not env_df.empty and "time" in env_df.columns and enough(env_df["time"]):
            tser = env_df["time"].dropna()
            rows.append({
                "항목": "측정 기간",
                "값": f"{tser.min().strftime('%Y-%m-%d')} ~ {tser.max().strftime('%Y-%m-%d')}",
            })
        else:
            rows.append({"항목": "측정 기간", "값": "환경 데이터 없음"})

        if gn_cols:
            rows.append({"항목": "생육 측정 항목", "값": ", ".join(gn_cols)})
        else:
            rows.append({"항목": "생육 측정 항목", "값": "없음"})

        env_num_cols_list = [
            c for c in ["temperature", "humidity", "ph", "ec"]
            if c in env_df.columns and pd.api.types.is_numeric_dtype(env_df[c])
        ]
        if env_num_cols_list:
            rows.append({"항목": "환경 측정 항목", "값": ", ".join(env_num_cols_list)})
        else:
            rows.append({"항목": "환경 측정 항목", "값": "없음"})

        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("📊 주요 지표")
        c1, c2, c3, c4 = st.columns(4)

        if not growth_df.empty and "개체번호" in growth_df.columns:
            n = growth_df["개체번호"].dropna().nunique()
            c1.metric("🌱 총 개체 수", f"{n}개체")
        else:
            c1.metric("🌱 총 개체 수", "—")

        if "temperature" in env_df.columns and enough(env_df["temperature"]):
            c2.metric("🌡️ 평균 온도", f"{env_df['temperature'].dropna().mean():.1f}℃")
        else:
            c2.metric("🌡️ 평균 온도", "데이터 없음")

        if "humidity" in env_df.columns and enough(env_df["humidity"]):
            c3.metric("💧 평균 습도", f"{env_df['humidity'].dropna().mean():.1f}%")
        else:
            c3.metric("💧 평균 습도", "데이터 없음")

        if "생중량" in growth_df.columns and enough(growth_df["생중량"]):
            c4.metric("⚖️ 평균 생중량", f"{growth_df['생중량'].dropna().mean():.2f}g")
        else:
            c4.metric("⚖️ 평균 생중량", "데이터 없음")

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
            env_num_cols = [
                c for c in ["temperature", "humidity", "ph", "ec"]
                if c in env_df.columns and pd.api.types.is_numeric_dtype(env_df[c])
            ]

            if not env_num_cols:
                st.info("🌡️ 환경 데이터에서 숫자 측정 열을 찾지 못했습니다.")
            else:
                st.subheader("📋 환경 요약 (평균·최소·최대)")
                summary_rows = []
                for col in env_num_cols:
                    s = env_df[col].dropna()
                    if len(s) == 0:
                        continue
                    summary_rows.append({
                        "측정 항목": col,
                        "평균": f"{s.mean():.2f}",
                        "최소": f"{s.min():.2f}",
                        "최대": f"{s.max():.2f}",
                        "측정 횟수": f"{len(s)}회",
                    })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                st.subheader("📈 날짜별 시계열 (일별 평균)")

                if "time" in env_df.columns and enough(env_df["time"]):
                    env_df["date"] = env_df["time"].dt.date

                    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
                    fig.suptitle(f"{school_name} 환경 데이터 — 일별 평균", fontsize=14)

                    if "temperature" in env_df.columns:
                        temp_daily = (
                            env_df.groupby("date")["temperature"]
                            .mean()
                            .reset_index()
                        )
                        temp_daily["date"] = pd.to_datetime(temp_daily["date"])
                        ax_t = axes[0]
                        ax_t.plot(temp_daily["date"], temp_daily["temperature"],
                                  color="red", linewidth=1.8, marker="o", markersize=4)
                        ax_t.axhline(temp_daily["temperature"].mean(), color="gray",
                                     linestyle="--", linewidth=1,
                                     label=f"평균 {temp_daily['temperature'].mean():.1f}℃")
                        ax_t.set_ylabel("온도 (℃)", color="red")
                        ax_t.tick_params(axis="y", labelcolor="red")
                        ax_t.set_title("🌡️ 일별 평균 온도")
                        ax_t.legend(loc="upper right")
                        ax_t.grid(True, alpha=0.3)
                    else:
                        axes[0].text(0.5, 0.5, "온도 열이 없습니다", ha="center",
                                     va="center", transform=axes[0].transAxes)
                        axes[0].set_title("🌡️ 일별 평균 온도 (데이터 없음)")

                    if "humidity" in env_df.columns:
                        hum_daily = (
                            env_df.groupby("date")["humidity"]
                            .mean()
                            .reset_index()
                        )
                        hum_daily["date"] = pd.to_datetime(hum_daily["date"])
                        ax_h = axes[1]
                        ax_h.plot(hum_daily["date"], hum_daily["humidity"],
                                  color="blue", linewidth=1.8, marker="o", markersize=4)
                        ax_h.axhline(hum_daily["humidity"].mean(), color="gray",
                                     linestyle="--", linewidth=1,
                                     label=f"평균 {hum_daily['humidity'].mean():.1f}%")
                        ax_h.set_ylabel("습도 (%)", color="blue")
                        ax_h.tick_params(axis="y", labelcolor="blue")
                        ax_h.set_title("💧 일별 평균 습도")
                        ax_h.legend(loc="upper right")
                        ax_h.grid(True, alpha=0.3)
                        ax_h.set_xlabel("날짜")
                    else:
                        axes[1].text(0.5, 0.5, "습도 열이 없습니다", ha="center",
                                     va="center", transform=axes[1].transAxes)
                        axes[1].set_title("💧 일별 평균 습도 (데이터 없음)")
                        axes[1].set_xlabel("날짜")

                    fig.autofmt_xdate(rotation=30)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                    if "ph" in env_df.columns:
                        st.info("🧪 pH 일별 변화도 확인할 수 있습니다 (아래 원본 테이블).")
                    if "ec" in env_df.columns:
                        st.info("⚡ EC 일별 변화도 확인할 수 있습니다 (아래 원본 테이블).")

                    with st.expander("📄 환경 데이터 원본 테이블 + 다운로드"):
                        st.dataframe(env_df, use_container_width=True, height=280)
                        make_csv_download(env_df, f"{school_name}_환경데이터")
                else:
                    st.info("📅 날짜(time) 열이 없거나 유효한 날짜가 부족해 시계열을 표시할 수 없습니다.")

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
            has_fresh = (
                "생중량" in growth_df.columns
                and pd.api.types.is_numeric_dtype(growth_df["생중량"])
            )

            if not has_fresh:
                st.info(
                    "⚖️ '생중량' 열이 없습니다. 지상부 질량 + 지하부 질량으로 생중량을 계산한 뒤 다시 시도해 주세요."
                )
            else:
                fw = growth_df["생중량"].dropna()
                avg_fw = float(fw.mean())
                std_fw = float(fw.std())
                cv_fw = float(std_fw / avg_fw) if avg_fw != 0 else 0.0

                # ── 핵심 결과 카드 ──
                st.subheader("🥇 핵심 결과 카드")
                cA, cB, cC = st.columns(3)
                cA.metric("📊 평균 생중량", f"{avg_fw:.2f}g")

                if len(fw) > 0:
                    max_idx = fw.idxmax()
                    min_idx = fw.idxmin()
                    pid = growth_df["개체번호"]
                    max_id = pid.loc[max_idx] if max_idx in pid.index else "?"
                    min_id = pid.loc[min_idx] if min_idx in pid.index else "?"
                    max_str = str(int(max_id)) if (isinstance(max_id, float) and max_id == int(max_id)) else str(max_id)
                    min_str = str(int(min_id)) if (isinstance(min_id, float) and min_id == int(min_id)) else str(min_id)
                    cB.metric("🏆 최대 생중량 개체", f"{max_str}번: {fw.max():.2f}g")
                    cC.metric("📉 최소 생중량 개체", f"{min_str}번: {fw.min():.2f}g")
                else:
                    cB.metric("🏆 최대 생중량 개체", "데이터 없음")
                    cC.metric("📉 최소 생중량 개체", "데이터 없음")

                st.caption(
                    f"""
**분포 요약**
- 개수: {len(fw)}개체
- 평균: {avg_fw:.2f}g
- 표준편차: {std_fw:.2f}g (값이 평균에서 얼마나 흩어져 있는지)
- 변동계수(CV): {cv_fw:.2f} (표준편차 ÷ 평균, 상대적 산포)
- 최소: {fw.min():.2f}g  |  최대: {fw.max():.2f}g
"""
                )

                # ── 개체별 막대 그래프 ──
                st.subheader("📊 개체별 생육 비교 (막대 그래프)")
                plot_cols = [
                    c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                    if c in growth_df.columns and pd.api.types.is_numeric_dtype(growth_df[c])
                ]
                if not plot_cols:
                    st.info("측정 항목 숫자 열이 없습니다.")
                else:
                    sel = [c for c in selected_items if c in growth_df.columns]
                    if not sel:
                        sel = plot_cols[:4]
                    n_plots = len(sel)
                    if n_plots == 0:
                        st.info("표시할 항목이 없습니다.")
                    else:
                        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.2 * n_plots))
                        if n_plots == 1:
                            axes = np.array([axes])
                        fig.suptitle(f"{school_name} 개체별 생육 측정값", fontsize=13)

                        for i, col in enumerate(sel):
                            ax = axes[i]
                            vals = growth_df[col].dropna()
                            if len(vals) == 0:
                                ax.text(0.5, 0.5, f"{col} 데이터 없음", ha="center",
                                        va="center", transform=ax.transAxes)
                                ax.set_title(col)
                                continue
                            ids = growth_df.loc[vals.index, "개체번호"]
                            id_list = [
                                int(v) if (isinstance(v, float) and v == int(v)) else v
                                for v in ids
                            ]
                            is_fw = col == "생중량"
                            color = "gold" if is_fw else "steelblue"
                            ax.bar(id_list, vals.values, color=color, edgecolor="white",
                                   linewidth=0.5)
                            ax.axhline(vals.mean(), color="gray", linestyle="--", linewidth=1,
                                       label=f"평균 {vals.mean():.2f}")
                            ax.set_title(col, fontsize=11)
                            ax.set_ylabel(col)
                            ax.legend(loc="upper right", fontsize=9)
                            ax.grid(True, axis="y", alpha=0.3)
                            ax.set_xlabel("개체번호" if i == n_plots - 1 else "")

                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

                # ── 박스플롯 ──
                if has_fresh and len(fw) > 0:
                    st.subheader("📦 생중량 분포 (박스플롯)")
                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    bp = ax2.boxplot(fw.values, vert=True, patch_artist=True,
                                     boxprops=dict(facecolor="lightgreen", color="darkgreen"),
                                     medianprops=dict(color="darkgreen", linewidth=2),
                                     whiskerprops=dict(color="darkgreen"),
                                     capprops=dict(color="darkgreen"),
                                     flierprops=dict(marker="o", markerfacecolor="red",
                                                     markeredgecolor="darkred", markersize=7))
                    ax2.set_title("생중량 분포 — 상자: 중간 50%(Q1~Q3), 선: 중앙값, ◆: 아웃라이어")
                    ax2.set_ylabel("생중량 (g)")
                    ax2.grid(True, axis="y", alpha=0.3)
                    ax2.set_xlabel("")

                    q1 = float(fw.quantile(0.25))
                    q3 = float(fw.quantile(0.75))
                    ax2.axhline(q1, color="orange", linestyle=":", linewidth=1,
                                label=f"Q1 (하위 25%) = {q1:.2f}g")
                    ax2.axhline(q3, color="purple", linestyle=":", linewidth=1,
                                label=f"Q3 (상위 25%) = {q3:.2f}g")
                    ax2.axhline(avg_fw, color="gray", linestyle="--", linewidth=1,
                                label=f"평균 = {avg_fw:.2f}g")
                    ax2.legend(fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)

                    st.caption(f"""
**박스플롯 읽기**
- 상자(위아래 경계): Q1({q1:.2f}g) ~ Q3({q3:.2f}g) — 전체 개체 중 가운데 50%가 이 범위
- 상자 안 굵은 선: 중앙값 (데이터 정가운데 값)
- 수염(whisker): 정상 범위 끝까지
- 빨간 ◆: 아웃라이어(극단값) — 56번처럼 매우 큰 개체
- **변동계수 CV = {cv_fw:.2f}**: 평균 대비 산포가 약 {cv_fw*100:.0f}% — 개체 간 생육 차이가 상당히 큼
"""
                    )

                # ── 상관관계: 산점도 + 추세선 ──
                st.subheader("🔗 상관관계 분석 (산점도 + 추세선)")

                corr_pairs = [
                    (a, b)
                    for a, b in [("잎 수", "생중량"), ("지상부 길이", "생중량")]
                    if a in growth_df.columns and b in growth_df.columns
                ]
                if not corr_pairs:
                    st.info("상관관계 분석할 항목 쌍이 없습니다.")
                else:
                    for x_col, y_col in corr_pairs:
                        fig3, ax3 = plt.subplots(figsize=(7, 5))
                        xv = growth_df[x_col].dropna()
                        yv = growth_df.loc[xv.index, y_col].dropna()
                        idx = xv.index.intersection(yv.index)
                        xs = growth_df.loc[idx, x_col]
                        ys = growth_df.loc[idx, y_col]

                        if len(xs) > 0:
                            ax3.scatter(xs, ys, s=40, color="steelblue", alpha=0.7,
                                        edgecolors="white", linewidth=0.5)
                            add_trendline(ax3, xs, ys, color="red", ls="--", lw=2)

                            r_val = safe_corr(growth_df, x_col, y_col)
                            r_text = f"r = {r_val:.3f}" if r_val is not None else "r = 계산 불가"
                            ax3.set_title(f"{x_col} vs {y_col}  |  {r_text}")
                            ax3.set_xlabel(x_col)
                            ax3.set_ylabel(y_col)
                            ax3.grid(True, alpha=0.3)

                            if r_val is not None:
                                if abs(r_val) >= 0.7:
                                    strength = "강한"
                                elif abs(r_val) >= 0.4:
                                    strength = "중간"
                                else:
                                    strength = "약한"
                                direction = "▲ 함께 증가" if r_val > 0 else "▼ 반대 방향"
                                ax3.text(0.05, 0.95,
                                         f"r = {r_val:.3f} → {strength} {direction}\n"
                                         f"※ 상관계수 ≠ 인과",
                                         transform=ax3.transAxes, fontsize=10,
                                         verticalalignment="top",
                                         bbox=dict(boxstyle="round", facecolor="wheat",
                                                   alpha=0.5))

                            plt.tight_layout()
                            st.pyplot(fig3, use_container_width=True)
                            plt.close(fig3)

                            if r_val is not None:
                                st.caption(
                                    f"""
**{x_col} vs {y_col} 상관 해석**
- 상관계수 r = {r_val:.3f} → **{strength} {direction}** 관계
- r 범위: **-1 ≤ r ≤ 1**
  - 1에 가까움: 함께 증가 (예: 잎 수↑ → 생중량↑)
  - 0에 가까움: 관계 거의 없음
  - -1에 가까움: 한쪽 증가 시 다른 쪽 감소
- **상관계수 ≠ 인과**: 함께 움직인다고 해서 한쪽이 원인이라고 단정할 수 없습니다.
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
                        c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"]
                        if c in growth_df.columns and pd.api.types.is_numeric_dtype(growth_df[c])
                    ]
                    if comp_cols:
                        comp_rows = []
                        for col in comp_cols:
                            tv = top_df[col].dropna()
                            bv = bot_df[col].dropna()
                            comp_rows.append({
                                "측정 항목": col,
                                "상위 25% 평균": f"{tv.mean():.2f}" if len(tv) else "—",
                                "하위 25% 평균": f"{bv.mean():.2f}" if len(bv) else "—",
                                "상위 개체 수": f"{len(top_df)}개체",
                                "하위 개체 수": f"{len(bot_df)}개체",
                            })
                        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)

                        # 막대 비교 차트
                        fig4, ax4 = plt.subplots(figsize=(8, 4))
                        labels = comp_cols
                        top_means = [float(top_df[c].dropna().mean()) if len(top_df[c].dropna()) else 0 for c in comp_cols]
                        bot_means = [float(bot_df[c].dropna().mean()) if len(bot_df[c].dropna()) else 0 for c in comp_cols]

                        x_pos = np.arange(len(labels))
                        w = 0.35
                        ax4.bar(x_pos - w/2, top_means, width=w, color="gold", edgecolor="white",
                                label=f"상위 25% (Q3={q75:.2f}g 이상, n={len(top_df)})")
                        ax4.bar(x_pos + w/2, bot_means, width=w, color="lightcoral", edgecolor="white",
                                label=f"하위 25% (Q1={q25:.2f}g 이하, n={len(bot_df)})")
                        ax4.set_xticks(x_pos)
                        ax4.set_xticklabels(labels, fontsize=10)
                        ax4.set_ylabel("평균값")
                        ax4.set_title(f"상위 25% vs 하위 25% 평균 비교")
                        ax4.legend(fontsize=9)
                        ax4.grid(True, axis="y", alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig4, use_container_width=True)
                        plt.close(fig4)
                    else:
                        st.info("비교할 측정 항목이 없습니다.")

                # ── 원본 테이블 + 다운로드 ──
                with st.expander("📄 생육 데이터 원본 테이블 + 다운로드"):
                    st.dataframe(growth_df, use_container_width=True, height=280)
                    make_csv_download(growth_df, f"{school_name}_생육결과데이터")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
