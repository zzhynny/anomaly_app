"""
app.py
AnomaLens — 다변량 시계열 이상탐지 대시보드
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from core.data_loader import load_csv, LoadResult
from core.detection import run_multivariate_detection, SCORER_REGISTRY, DetectionResult
from core.evaluation import evaluate_unsupervised, evaluate_with_labels, scorer_agreement_matrix
from core.preprocessing import fill_missing_values, remove_constant_columns
from core.quality import analyze_quality
from core.decomposition import decompose_series, estimate_period
from core.stationarity import test_stationarity_dataframe
from core.explain import explain_top_anomalies
from core.isolation_forest_detection import detect_isolation_forest
from sklearn.metrics import roc_auc_score as _roc_auc_score, average_precision_score as _ap_score
from sklearn.metrics import confusion_matrix as _confusion_matrix

st.set_page_config(page_title="AnomaLens", page_icon="🔎", layout="wide")

# 색상 팔레트
C_PRIMARY   = "#2563EB"
C_ANOMALY   = "#DC2626"
C_SUCCESS   = "#16A34A"
C_NEUTRAL   = "#6B7280"
C_BG_CARD   = "#F8FAFC"

# ── 캐시 함수 ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_load_csv(file_bytes: bytes, file_hash: str) -> LoadResult:
    class _B:
        def getvalue(self): return file_bytes
    return load_csv(_B())

@st.cache_data(show_spinner=False)
def _cached_preprocess(df: pd.DataFrame, fill_method: str, key: str):
    df, cc = remove_constant_columns(df)
    pr = fill_missing_values(df, method=fill_method)
    return pr, cc

@st.cache_data(show_spinner=False)
def _cached_detection(df, test_ratio, scorer_keys, threshold_quantile, key) -> DetectionResult:
    return run_multivariate_detection(df, test_ratio=test_ratio, scorer_keys=list(scorer_keys), threshold_quantile=threshold_quantile)

@st.cache_data(show_spinner=False)
def _cached_quality(raw_df, clean_df, freq, key):
    return analyze_quality(raw_df, clean_df, freq)

@st.cache_data(show_spinner=False)
def _cached_decomposition(df, freq, key):
    # 주파수 기반 고정 period(예: 분 단위 데이터 -> 60) 대신, 각 변수마다 ACF로 실제
    # 주기를 데이터 기반으로 추정(estimate_period)해 STL에 사용한다.
    results = {}
    for col in df.columns:
        try:
            p = estimate_period(df[col])
            results[col] = decompose_series(df[col], freq, period_override=p)
        except Exception:
            continue
    return results

@st.cache_data(show_spinner=False)
def _cached_stationarity(df, key):
    return test_stationarity_dataframe(df)

@st.cache_data(show_spinner=False)
def _cached_iforest(df, contamination, key):
    return detect_isolation_forest(df, contamination=contamination)

def file_hash(f) -> str:
    return hashlib.sha256(f.getvalue()).hexdigest()

class _BuiltinFile:
    """내장 데이터셋을 st.file_uploader가 반환하는 UploadedFile과 동일한 인터페이스로
    감싸서, 업로드 경로와 완전히 동일한 데이터 로드 파이프라인을 그대로 재사용한다."""
    def __init__(self, data: bytes, name: str):
        self._data = data
        self.name = name
    def getvalue(self) -> bytes:
        return self._data

# ── 시각화 헬퍼 ──────────────────────────────────────────────────────────────
def kpi_card(col, label: str, value: str, delta: str = "", color: str = C_PRIMARY, value_font_size: int = 28):
    # 줄바꿈으로 분리된 멀티라인 f-string은 delta가 빈 문자열일 때 그 줄이 공백만 남는
    # 빈 줄이 되어, 마크다운 HTML 블록 파서가 그 지점에서 블록을 끊고 이후의 "</div>"를
    # 원시 HTML이 아닌 일반 텍스트로 취급해 화면에 그대로 노출시킨다. 줄바꿈 없는 단일
    # 문자열로 합쳐 빈 줄이 생기지 않도록 한다.
    delta_html = f"<div style='font-size:12px;color:{C_NEUTRAL};margin-top:4px'>{delta}</div>" if delta else ""
    html = (
        f'<div style="background:{C_BG_CARD};border:1px solid #E2E8F0;border-radius:10px;'
        f'padding:16px 16px 14px;border-top:3px solid {color}">'
        f'<div style="font-size:11px;font-weight:600;letter-spacing:.02em;color:{C_NEUTRAL};'
        f'margin-bottom:6px;text-transform:uppercase">{label}</div>'
        f'<div style="font-size:{value_font_size}px;font-weight:800;color:{color};line-height:1.1;'
        f'white-space:nowrap">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )
    col.markdown(html, unsafe_allow_html=True)

def status_card(col, message: str, ok: bool = True):
    color = C_SUCCESS if ok else C_ANOMALY
    bg = "#F0FDF4" if ok else "#FEF2F2"
    col.markdown(
        f"""<div style="background:{bg};border:1px solid {color}40;border-radius:10px;
        padding:20px 16px;text-align:center;color:{color};font-weight:700;font-size:14px">{message}</div>""",
        unsafe_allow_html=True,
    )

def plot_series_anomalies(raw: pd.Series, anomaly_times: pd.DatetimeIndex,
                          title: str, zoom=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw.index, y=raw.values, mode="lines",
                             name="값", line=dict(color=C_PRIMARY, width=1.5)))
    if len(anomaly_times) > 0:
        y_pts = raw.reindex(anomaly_times)
        fig.add_trace(go.Scatter(x=anomaly_times, y=y_pts.values, mode="markers",
                                 name="이상", marker=dict(color=C_ANOMALY, size=7,
                                 symbol="circle-open", line=dict(width=2))))
    fig.update_layout(title=dict(text=title, font=dict(size=14)), height=300,
                      margin=dict(l=10, r=10, t=36, b=10), template="plotly_white",
                      legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    if zoom:
        fig.update_xaxes(range=list(zoom))
    return fig

def zoom_range(full_idx, anom_idx, pad=0.15):
    if not len(anom_idx): return None
    s, e = anom_idx.min(), anom_idx.max()
    span = max((e - s), pd.Timedelta(seconds=1))
    p = span * pad
    return (max(full_idx.min(), s - p), min(full_idx.max(), e + p))

def plot_score_comp(comp, keys) -> go.Figure:
    fig = make_subplots(rows=len(keys), cols=1, shared_xaxes=True,
                        subplot_titles=[SCORER_REGISTRY[k]["label"] for k in keys],
                        vertical_spacing=0.06)
    for i, k in enumerate(keys, 1):
        if k not in comp.scores: continue
        s = comp.scores[k]; b = comp.binary.get(k)
        times = s.time_index; vals = s.values().flatten()
        fig.add_trace(go.Scatter(x=times, y=vals, line=dict(color=C_PRIMARY, width=1.2)), row=i, col=1)
        if b is not None:
            cs = max(s.start_time(), b.start_time())
            ce = min(s.end_time(), b.end_time())
            sa = s.slice(cs, ce); ba = b.slice(cs, ce)
            bv = ba.values().flatten(); sv = sa.values().flatten()
            mask = bv == 1
            if mask.any():
                fig.add_trace(go.Scatter(x=sa.time_index[mask], y=sv[mask],
                                         mode="markers", marker=dict(color=C_ANOMALY, size=5),
                                         showlegend=False), row=i, col=1)
    fig.update_layout(height=200 * len(keys), showlegend=False,
                      template="plotly_white", margin=dict(t=30, b=10, l=10, r=10))
    return fig

# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='display:flex;align-items:center;gap:8px;margin-bottom:14px'>"
        "<span style='font-size:15px;font-weight:800;color:#1E293B'>🔎 AnomaLens</span>"
        "<span style='background:#E8F0FE;color:#2563EB;border-radius:6px;font-size:11px;padding:2px 8px'>시계열 이상탐지</span>"
        "</div>",
        unsafe_allow_html=True)

    uploaded_file = st.file_uploader("다변량 시계열 CSV 업로드", type=["csv"],
        help="시간/날짜 컬럼과 1개 이상의 수치형 변수 컬럼을 포함한 CSV",
        key="csv_uploader")

    if uploaded_file is not None:
        st.session_state.active_source = "upload"

    st.markdown("---")
    st.subheader("내장 데이터셋")
    with st.container(border=True):
        st.markdown(
            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>"
            "<div style='font-size:26px'>💧</div>"
            "<div><div style='font-weight:700;font-size:14px'>GECCO 2019</div>"
            f"<div style='font-size:12px;color:{C_NEUTRAL}'>6개 변수 · 6,000행</div></div>"
            "</div>", unsafe_allow_html=True)
        if st.button("📥 불러오기", use_container_width=True, key="load_gecco_btn"):
            with open("sample_data/gecco2019_sample.csv", "rb") as f:
                st.session_state.builtin_df_bytes = f.read()
            st.session_state.builtin_df_name = "gecco2019_sample.csv"
            st.session_state.active_source = "gecco2019"
            # 업로드 위젯 상태를 초기화해, 업로드된 파일이 있던 상태에서도
            # 내장 데이터셋으로 확실히 전환되도록 한다.
            if "csv_uploader" in st.session_state:
                del st.session_state["csv_uploader"]
            st.rerun()
        if st.session_state.get("active_source") == "gecco2019":
            st.caption("✅ 로드됨")

    if st.session_state.get("active_source") == "gecco2019" and st.session_state.get("builtin_df_bytes") is not None:
        uploaded_file = _BuiltinFile(st.session_state.builtin_df_bytes, st.session_state.builtin_df_name)

    st.markdown("---")
    st.subheader("탐지 설정")

    test_ratio_pct = st.slider("탐지 구간 비율 (%)", 10, 60, 30, 5,
        help="앞부분을 정상 패턴 학습에, 뒷부분(이 비율)을 이상탐지에 사용합니다.")

    selected_scorers = st.multiselect("Scorer 선택",
        options=list(SCORER_REGISTRY.keys()),
        default=list(SCORER_REGISTRY.keys()),
        format_func=lambda k: SCORER_REGISTRY[k]["label"])

    threshold_pct = st.slider("이상 판정 임계값 (상위 %)", 1, 20, 5, 1,
        help="점수 상위 N%를 이상으로 판정합니다.")
    threshold_q = 1 - threshold_pct / 100

    fill_method = st.selectbox("결측치 처리",
        options=["interpolate_linear", "ffill", "bfill", "moving_average"],
        format_func=lambda k: {"interpolate_linear":"선형 보간 (권장)",
                               "ffill":"직전값 유지 (LOCF)",
                               "bfill":"직후값 유지 (NOCB)",
                               "moving_average":"이동평균"}[k])

    st.markdown("---")
    st.subheader("추가 옵션")
    enable_iforest = st.checkbox("Isolation Forest 비교", value=True,
        help="머신러닝 기반 알고리즘(Isolation Forest)을 예측 오차 기반 알고리즘과 함께 비교합니다.")
    iforest_contam = st.slider("IForest 이상치 비율 추정 (%)", 1, 20, 5, 1,
        disabled=not enable_iforest) / 100
    auto_zoom = True  # 이상구간 자동 확대는 항상 적용

    st.markdown("---")
    st.markdown(
        "<div style='font-size:12px;color:#6B7280;line-height:1.7'><b>사용 기술</b><br>"
        "• Darts ForecastingAnomalyModel<br>"
        "• Isolation Forest<br>"
        "• STL Decomposition<br>"
        "• ADF Test</div>",
        unsafe_allow_html=True)

# ── 메인 영역 ────────────────────────────────────────────────────────────────
header_col, status_col = st.columns([3, 1])
with header_col:
    st.markdown("""
    <h1 style='margin-bottom:6px;font-size:1.7rem;font-weight:800'>AnomaLens</h1>
    <span style="
      background-color: #EFF6FF;
      color: #2563EB;
      border: 1px solid #BFDBFE;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      padding: 3px 10px;
    ">다변량 시계열 이상탐지 대시보드</span>
    <p style='color:#6B7280;margin-top:8px;font-size:14px'>다변량 시계열 데이터를 자동 분석하여 이상 구간 탐지, 정상성 검정, 성능 평가 결과를 제공합니다.</p>
    """, unsafe_allow_html=True)
status_placeholder = status_col.empty()

# ── 업로드 전 Quick Start Guide ───────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("---")
    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    <style>
    div[data-testid="stHorizontalBlock"]{align-items:stretch}
    div[data-testid="stColumn"]{display:flex}
    div[data-testid="stColumn"] > div{display:flex;flex-direction:column;width:100%}
    div[data-testid="stVerticalBlock"]{height:100%}
    .qs-card{background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:16px 20px;
        text-align:center;height:100%;min-height:210px;display:flex;flex-direction:column;
        align-items:center;justify-content:center;box-sizing:border-box}
    .qs-card .qs-icon{font-size:32px;line-height:1;margin-bottom:8px}
    .qs-card .qs-title{font-weight:700;font-size:16px;line-height:1.2;margin-bottom:6px}
    .qs-card .qs-desc{color:#6B7280;font-size:13px;line-height:1.5}
    .qs-info-card{min-height:150px;align-items:flex-start;justify-content:center;text-align:left}
    </style>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="qs-card">
        <div class="qs-icon">📂</div>
        <div class="qs-title">STEP 1 · 데이터 업로드</div>
        <div class="qs-desc">다변량 CSV 업로드<br>timestamp 자동 인식<br>라벨 컬럼 자동 감지</div></div>""",
        unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="qs-card">
        <div class="qs-icon">⚙️</div>
        <div class="qs-title">STEP 2 · 자동 분석</div>
        <div class="qs-desc">결측치 처리<br>STL 분해<br>정상성 검정 (ADF)<br>이상탐지 수행</div></div>""",
        unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="qs-card">
        <div class="qs-icon">📈</div>
        <div class="qs-title">STEP 3 · 결과 평가</div>
        <div class="qs-desc">이상 구간 시각화<br>Precision / Recall / ROC-AUC<br>변수 기여도 분석</div></div>""",
        unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("""<div class="qs-card qs-info-card">
        <div class="qs-title">🧩 지원 분석 기능</div>
        <div class="qs-desc">• STL 분해 (Trend / Seasonal / Residual)<br>• 정상성 검정 (ADF)<br>• 다중 Scorer 이상탐지<br>• Precision / Recall / ROC-AUC 평가</div></div>""",
        unsafe_allow_html=True)
    with ic2:
        st.markdown("""<div class="qs-card qs-info-card">
        <div class="qs-title">🧠 지원 알고리즘</div>
        <div class="qs-desc">• NormScorer — 예측 오차 기반 이상 탐지<br>• KMeansScorer — 군집 기반 이상 탐지<br>
        • WassersteinScorer — 분포 변화 기반 탐지<br>• Isolation Forest — 트리 기반 이상 탐지</div></div>""",
        unsafe_allow_html=True)

    st.markdown("---")
    st.info("👆 왼쪽 사이드바에서 CSV 파일을 업로드해 주세요.")
    st.stop()

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
data_bytes = uploaded_file.getvalue()
data_hash  = file_hash(uploaded_file)
source_label = uploaded_file.name

if not selected_scorers:
    st.warning("최소 1개 이상의 Scorer를 선택해 주세요.")
    st.stop()

with st.spinner("데이터를 읽는 중..."):
    load_result = _cached_load_csv(data_bytes, data_hash)

if load_result.warnings:
    with st.expander("⚠️ 데이터 로드 참고사항", expanded=False):
        for w in load_result.warnings: st.write("- " + w)

feature_df = load_result.df.copy()
label_col  = load_result.label_col
label_series = feature_df.pop(label_col) if label_col else None

pre_key = data_hash + fill_method
with st.spinner("전처리 중..."):
    preprocess_result, constant_cols = _cached_preprocess(feature_df, fill_method, pre_key)

if constant_cols:
    st.warning(f"상수 컬럼 {constant_cols} 제외됨")

clean_df = preprocess_result.df
if clean_df.shape[0] < 20:
    st.error("데이터가 너무 적습니다 (최소 20행 필요)."); st.stop()

# ── 이상탐지 ─────────────────────────────────────────────────────────────────
det_key = data_hash + fill_method + str(test_ratio_pct) + str(threshold_pct) + "_".join(sorted(selected_scorers))
with st.spinner("이상탐지 모델 학습 중... (데이터 크기에 따라 수십 초 소요될 수 있습니다)"):
    try:
        detection = _cached_detection(clean_df, test_ratio_pct/100,
                                       tuple(selected_scorers), threshold_q, det_key)
    except ValueError as e:
        st.error(str(e)); st.stop()

if detection.notes:
    with st.expander("ℹ️ 탐지 과정 참고사항"):
        for n in detection.notes: st.write("- " + n)

# ── 보조 분석 (캐시) ─────────────────────────────────────────────────────────
quality_report    = _cached_quality(feature_df, clean_df, load_result.freq_guess, pre_key)
stl_results       = _cached_decomposition(clean_df, load_result.freq_guess, pre_key)
stationarity_res  = _cached_stationarity(clean_df, pre_key)
explanations      = explain_top_anomalies(detection, top_n=5)

iforest_result = None
if enable_iforest:
    iforest_key = det_key + str(iforest_contam)
    with st.spinner("Isolation Forest 학습 중..."):
        try:
            iforest_result = _cached_iforest(clean_df, iforest_contam, iforest_key)
        except Exception as e:
            st.warning(f"Isolation Forest 계산 실패: {e}")

# ── 요약 KPI ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📌 이상탐지 결과 요약")

n_total   = len(detection.combined_binary)
n_anom    = int(detection.combined_binary.values().sum())
anom_rate = n_anom / n_total * 100 if n_total else 0

# ── 우측 상단 분석 상태 배지 채우기 ───────────────────────────────────────────
source_short = uploaded_file.name
status_text = f"{source_short} · 이상 탐지 완료 · 이상 비율 {anom_rate:.1f}%"
if label_series is not None:
    # 변수마다 Scorer 중 가장 좋은 AUC-ROC만 뽑아서 평균낸다. 단순 전체 평균은
    # temperature_c처럼 AUC가 낮은 변수에 끌려내려가 대표값이 실제보다 낮게 보인다.
    best_auc_per_component = [
        max(comp_aucs)
        for sd in evaluate_with_labels(detection, label_series).values()
        if (comp_aucs := [m.auc_roc for m in sd.values() if m.auc_roc is not None])
    ]
    if best_auc_per_component:
        status_text = f"{source_short} · 이상 탐지 완료 · AUC {np.mean(best_auc_per_component):.2f}"
status_placeholder.markdown(
    f"<div style='text-align:right;margin-top:16px;font-size:13px;font-weight:600;color:{C_PRIMARY};white-space:nowrap'>{status_text}</div>",
    unsafe_allow_html=True)

# 가장 영향력 큰 변수 (컴포넌트별 이상 합계 기준)
comp_anom_counts = {}
for cn, comp in detection.components.items():
    if comp.binary:
        bl = list(comp.binary.values())
        cs = max(b.start_time() for b in bl); ce = min(b.end_time() for b in bl)
        al = [b.slice(cs, ce) for b in bl]
        comp_anom_counts[cn] = int(np.stack([b.values().flatten() for b in al]).max(axis=0).sum())

top5 = sorted(comp_anom_counts.items(), key=lambda x: x[1], reverse=True)[:5]

has_is_anomaly = label_col == "is_anomaly" and label_series is not None
if has_is_anomaly:
    c1, c2, c3, c4, c5 = st.columns(5)
else:
    c1, c2, c3, c4 = st.columns(4)
kpi_card(c1, "분석 변수", f"{len(detection.components)}개")
kpi_card(c2, "탐지 구간", f"{n_total:,} 시점")
kpi_card(c3, "이상 판정 시점", f"{n_anom:,}개", color=C_ANOMALY)
kpi_card(c4, "탐지 비율(모델)" if has_is_anomaly else "이상 비율", f"{anom_rate:.1f}%",
         color=C_ANOMALY if anom_rate > 20 else (C_SUCCESS if anom_rate < 5 else C_PRIMARY))
if has_is_anomaly:
    true_anom_rate = label_series.mean() * 100
    kpi_card(c5, "실제 이상 비율", f"{true_anom_rate:.1f}%", color=C_NEUTRAL)

st.markdown("**변수별 이상 기여도 TOP 5**")
if top5:
    max_cnt = max(v for _, v in top5) or 1
    for name, cnt in top5:
        pct = cnt / max_cnt
        st.markdown(
            f"<div style='margin:3px 0'><span style='font-size:13px;width:200px;display:inline-block'>{name}</span>"
            f"<div style='display:inline-block;background:{C_ANOMALY};height:12px;width:{int(pct*200)}px;border-radius:4px;vertical-align:middle'></div>"
            f"<span style='font-size:12px;color:{C_NEUTRAL};margin-left:8px'>{cnt}개</span></div>",
            unsafe_allow_html=True)

st.markdown("---")

# ── 탭 ───────────────────────────────────────────────────────────────────────
tab_detect, tab_scorers, tab_eval, tab_agree, tab_explain, tab_quality, tab_stl, tab_stat = st.tabs([
    "📈 이상탐지 결과", "🧪 Scorer 비교", "📊 정량 평가", "🔁 일치도 분석",
    "💡 이상 원인 설명", "🗂️ 데이터 품질", "📐 STL 분해", "📏 정상성 검정"])

# ── TAB 1: 이상탐지 결과 ─────────────────────────────────────────────────────
with tab_detect:
    st.markdown("각 변수의 시계열과 탐지된 이상 시점을 표시합니다. 빨간 점이 이상으로 판정된 시점입니다.")
    for comp_name, comp in detection.components.items():
        bl = list(comp.binary.values())
        if not bl:
            st.warning(f"**{comp_name}**: 탐지 결과 없음 (결측치 과다 또는 상수 변수)"); continue
        cs = max(b.start_time() for b in bl); ce = min(b.end_time() for b in bl)
        al = [b.slice(cs, ce) for b in bl]
        union = np.stack([b.values().flatten() for b in al]).max(axis=0)
        atimes = al[0].time_index[union == 1]
        raw = pd.concat([comp.train.to_series(), comp.test.to_series()])
        zr = zoom_range(raw.index, atimes) if auto_zoom and len(atimes) else None
        fig = plot_series_anomalies(raw, atimes, comp_name, zr)
        st.plotly_chart(fig, use_container_width=True)
        if zr:
            st.caption(f"🔍 자동 확대: {zr[0]} ~ {zr[1]} 구간 표시 중")

# ── TAB 2: Scorer 비교 ───────────────────────────────────────────────────────
with tab_scorers:
    st.markdown(
        "동일 변수에 대해 서로 다른 Scorer가 산출한 이상 점수를 비교합니다. "
        "여러 Scorer가 동시에 높은 점수를 주는 구간일수록 이상 신뢰도가 높습니다.")
    comp_sel = st.selectbox("변수 선택", options=list(detection.components.keys()), key="sc_comp")
    comp = detection.components[comp_sel]
    active = [k for k in detection.scorer_keys if k in comp.scores]
    if active:
        st.plotly_chart(plot_score_comp(comp, active), use_container_width=True)
        for k in active:
            st.caption(f"**{SCORER_REGISTRY[k]['label']}**: {SCORER_REGISTRY[k]['description']}")
    else:
        st.warning("이 변수에 계산된 점수가 없습니다.")

    if iforest_result and iforest_result.components:
        st.markdown("---")
        st.markdown("#### 🌲 Isolation Forest 비교")
        st.caption("예측 오차 기반 Scorer와는 독립적인 트리 앙상블 기반 알고리즘 결과를 함께 비교합니다.")
        if_sel = st.selectbox("변수 선택", options=list(iforest_result.components.keys()), key="if_comp")
        icomp = iforest_result.components[if_sel]
        fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=["Isolation Forest 이상 점수", "Darts 기반 판정 (OR)"],
                             vertical_spacing=0.1)
        fig2.add_trace(go.Scatter(x=icomp.anomaly_score.index, y=icomp.anomaly_score.values,
                                  line=dict(color="#059669", width=1.2)), row=1, col=1)
        ap = icomp.binary[icomp.binary == 1]
        if len(ap):
            fig2.add_trace(go.Scatter(x=ap.index,
                                      y=icomp.anomaly_score.reindex(ap.index).values,
                                      mode="markers", marker=dict(color=C_ANOMALY, size=5),
                                      showlegend=False), row=1, col=1)
        if if_sel in detection.components:
            dc = detection.components[if_sel]
            if dc.binary:
                dbl = list(dc.binary.values())
                dcs = max(b.start_time() for b in dbl); dce = min(b.end_time() for b in dbl)
                dal = [b.slice(dcs, dce) for b in dbl]
                duv = np.stack([b.values().flatten() for b in dal]).max(axis=0)
                fig2.add_trace(go.Scatter(x=dal[0].time_index, y=duv.astype(float),
                                          line=dict(color=C_PRIMARY, width=1)), row=2, col=1)
        fig2.update_layout(height=400, showlegend=False, template="plotly_white",
                           margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig2, use_container_width=True)

# ── TAB 3: 정량 평가 ─────────────────────────────────────────────────────────
with tab_eval:
    if label_series is not None:
        st.markdown(f"라벨 컬럼 **'{label_col}'** 기준 정량 평가입니다.")
        sup = evaluate_with_labels(detection, label_series)
        rows_eval = []
        for cn, sd in sup.items():
            for k, m in sd.items():
                rows_eval.append({
                    "변수": cn,
                    "Scorer": SCORER_REGISTRY[k]["label"],
                    "AUC-ROC": round(m.auc_roc, 3) if m.auc_roc is not None else None,
                    "AUC-PR": round(m.auc_pr, 3) if m.auc_pr is not None else None,
                    "Precision": round(m.precision, 3),
                    "Recall": round(m.recall, 3),
                    "F1": round(m.f1, 3),
                    "실제 이상": m.n_true_anomalies,
                    "탐지": m.n_detected,
                })
        eval_df = pd.DataFrame(rows_eval)
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
        st.caption("AUC-ROC / AUC-PR → 1에 가까울수록 우수. Precision: 탐지 정밀도. Recall: 실제 이상 포착률.")

        # 혼동 행렬 (combined_binary 기준)
        cb_vals = detection.combined_binary.values().flatten().astype(int)
        cb_idx  = detection.combined_binary.time_index
        y_true = label_series.reindex(cb_idx).fillna(0).astype(int).values
        if len(np.unique(y_true)) > 1:
            st.markdown("#### Confusion Matrix (전체 변수 OR 결합 기준)")
            cm = _confusion_matrix(y_true, cb_vals)
            cm_fig = go.Figure(go.Heatmap(
                z=cm, x=["정상(예측)", "이상(예측)"], y=["정상(실제)", "이상(실제)"],
                colorscale="Blues", text=cm, texttemplate="%{text}",
                showscale=False))
            cm_fig.update_layout(height=280, margin=dict(t=20, b=20, l=20, r=20), template="plotly_white")
            st.plotly_chart(cm_fig, use_container_width=True)

        # Isolation Forest 정량 평가
        if iforest_result:
            st.markdown("#### Isolation Forest 정량 평가")
            if_rows = []
            for cn, ic in iforest_result.components.items():
                try:
                    al_true = label_series.reindex(ic.anomaly_score.index).fillna(0).astype(int).values
                    if len(np.unique(al_true)) > 1:
                        auc = _roc_auc_score(al_true, ic.anomaly_score.values)
                        apr = _ap_score(al_true, ic.anomaly_score.values)
                        if_rows.append({"변수": cn, "AUC-ROC": round(auc, 3), "AUC-PR": round(apr, 3)})
                except Exception:
                    pass
            if if_rows:
                st.dataframe(pd.DataFrame(if_rows), use_container_width=True, hide_index=True)
    else:
        st.info("정답 라벨이 없어 정량 평가를 수행할 수 없습니다. 아래는 라벨 없이 탐지 적절성을 가늠하는 보조 지표입니다.")
        unsup = evaluate_unsupervised(detection)
        rows_u = []
        for cn, sd in unsup.items():
            for k, m in sd.items():
                rows_u.append({
                    "변수": cn, "Scorer": SCORER_REGISTRY[k]["label"],
                    "탐지 수": m.n_detected,
                    "탐지 비율(%)": round(m.detection_rate * 100, 2),
                    "점수 평균": round(m.score_mean, 3),
                    "점수 최대": round(m.score_max, 3),
                    "이상/정상 점수 비율": round(m.separation_ratio, 2) if not np.isnan(m.separation_ratio) else None,
                })
        st.dataframe(pd.DataFrame(rows_u), use_container_width=True, hide_index=True)
        st.caption("'이상/정상 점수 비율'이 클수록 이상 시점과 정상 시점의 점수 차이가 뚜렷해 탐지 신뢰도가 높다고 해석됩니다.")

# ── TAB 4: 일치도 분석 ───────────────────────────────────────────────────────
with tab_agree:
    st.markdown("한 변수 내에서 서로 다른 Scorer들이 같은 시점을 이상으로 판단하는 일치율입니다. "
                "1에 가까울수록 두 Scorer가 거의 동일하게 판단합니다.")
    ag_sel = st.selectbox("변수 선택", options=list(detection.components.keys()), key="ag_comp")
    ag_df = scorer_agreement_matrix(detection, ag_sel)
    if not ag_df.empty:
        st.dataframe(ag_df.style.format("{:.3f}").background_gradient(cmap="Blues"), use_container_width=True)
    else:
        st.warning("계산할 수 있는 Scorer 결과가 없습니다.")

# ── TAB 5: 이상 원인 설명 ────────────────────────────────────────────────────
with tab_explain:
    st.markdown("#### 💡 주요 이상 시점 자동 설명")
    st.caption("종합 이상도(여러 Scorer 평균)가 가장 높은 시점들에 대해 이상 원인을 자동으로 분석합니다.")
    if not explanations:
        st.info("설명할 이상 시점이 충분히 탐지되지 않았습니다.")
    else:
        for e in explanations:
            badge = "🔴" if "높음" in e.confidence_label else ("🟡" if "보통" in e.confidence_label else "⚪")
            with st.container():
                st.markdown(
                    f"""<div style="background:{C_BG_CARD};border-left:4px solid {C_ANOMALY};
                    border-radius:6px;padding:12px 16px;margin:8px 0;font-size:13px">
                    <b>{e.timestamp}</b> · <code>{e.column}</code><br>
                    • 값: {e.value:.4g} (평균 대비 <b>{e.sigma_from_mean:+.1f}σ</b>)<br>
                    • {e.direction}{f' · 직전 구간 대비 <b>{e.pct_change_recent:+.1f}%</b>' if e.pct_change_recent else ''}<br>
                    {"• 탐지 지표: " + ", ".join(SCORER_REGISTRY[k]["label"] for k in e.detected_by) + "<br>" if e.detected_by else ""}
                    {badge} <b>{e.confidence_label}</b>
                    </div>""", unsafe_allow_html=True)

# ── TAB 6: 데이터 품질 ───────────────────────────────────────────────────────
with tab_quality:
    q = quality_report
    st.markdown("업로드된 데이터의 기초 품질 분석 결과입니다.")
    qc1, qc2, qc3, qc4, qc5, qc6 = st.columns(6)
    kpi_card(qc1, "총 행 수", f"{q.n_rows:,}")
    kpi_card(qc2, "변수 수", f"{q.n_cols}개")
    kpi_card(qc3, "시작일", str(q.start_date)[:10], value_font_size=20)
    kpi_card(qc4, "종료일", str(q.end_date)[:10], value_font_size=20)
    kpi_card(qc5, "추정 주기", q.freq_guess or "불규칙")
    kpi_card(qc6, "분석 기간", q.duration_str)

    st.markdown("<br>", unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)
    with mc1:
        if q.n_missing_total == 0:
            status_card(st, "✅ 결측치 없음")
        else:
            st.warning(f"⚠️ 결측치 {q.n_missing_total:,}개 ({q.missing_ratio_total*100:.2f}%)")
            miss_fig = go.Figure(go.Bar(
                x=q.missing_by_column.index.tolist(),
                y=q.missing_by_column.values.tolist(),
                marker_color=C_PRIMARY))
            miss_fig.update_layout(height=220, template="plotly_white",
                                   margin=dict(t=10, b=10, l=10, r=10),
                                   title="컬럼별 결측치 수")
            st.plotly_chart(miss_fig, use_container_width=True)
    with mc2:
        if q.n_duplicated_timestamps == 0:
            status_card(st, "✅ 중복 타임스탬프 없음")
        else:
            st.warning(f"⚠️ 중복 타임스탬프 {q.n_duplicated_timestamps:,}개")

    st.markdown("#### 변수별 기술통계")
    st.dataframe(q.descriptive_stats.round(4), use_container_width=True)

# ── TAB 7: STL 분해 ──────────────────────────────────────────────────────────
with tab_stl:
    st.markdown("STL(Seasonal-Trend Decomposition using Loess) 분해로 각 변수를 추세·계절성·잔차 성분으로 분리합니다.")
    stl_col_sel, stl_col_period = st.columns([3, 1])
    with stl_col_sel:
        stl_sel = st.selectbox("변수 선택", options=list(clean_df.columns), key="stl_sel")
    with stl_col_period:
        stl_period = st.number_input("계절 주기", min_value=2, max_value=500,
            value=int(estimate_period(clean_df[stl_sel])), key=f"stl_period_{stl_sel}")
    try:
        r = decompose_series(clean_df[stl_sel], load_result.freq_guess, period_override=int(stl_period))
    except Exception as e:
        r = None
        st.info(str(e) or "데이터 길이가 충분하지 않아 계절성 분석을 생략했습니다.")
    if r is not None:
        sc1, sc2, sc3 = st.columns(3)
        kpi_card(sc1, "계절 주기 (period)", f"{r.period}")
        kpi_card(sc2, "추세 강도", f"{r.trend_strength:.3f}")
        kpi_card(sc3, "계절성 강도", f"{r.seasonal_strength:.3f}")
        stl_fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
            subplot_titles=["원본 (Observed)", "추세 (Trend)", "계절성 (Seasonal)", "잔차 (Residual)"],
            vertical_spacing=0.05)
        stl_fig.add_trace(go.Scatter(x=r.observed.index, y=r.observed.values, line=dict(color=C_NEUTRAL, width=1.2)), row=1, col=1)
        stl_fig.add_trace(go.Scatter(x=r.trend.index, y=r.trend.values, line=dict(color=C_PRIMARY, width=1.5)), row=2, col=1)
        stl_fig.add_trace(go.Scatter(x=r.seasonal.index, y=r.seasonal.values, line=dict(color="#059669", width=1.2)), row=3, col=1)
        stl_fig.add_trace(go.Scatter(x=r.resid.index, y=r.resid.values, line=dict(color=C_ANOMALY, width=1)), row=4, col=1)
        stl_fig.update_layout(height=640, showlegend=False, template="plotly_white",
                              margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(stl_fig, use_container_width=True)

    if stl_results:
        st.markdown("#### 전체 변수 STL 강도 요약")
        stl_rows = []
        for cn, sr in stl_results.items():
            stl_rows.append({"변수": cn, "계절 주기": sr.period,
                              "추세 강도": round(sr.trend_strength, 3),
                              "계절성 강도": round(sr.seasonal_strength, 3)})
        st.dataframe(pd.DataFrame(stl_rows), use_container_width=True, hide_index=True)

# ── TAB 8: 정상성 검정 ───────────────────────────────────────────────────────
with tab_stat:
    st.markdown("**ADF(Augmented Dickey-Fuller) 검정**을 통해 각 변수의 정상성을 평가합니다.")
    st.markdown("판정 기준: **p-value < 0.05 → 정상성 있음** (귀무가설: 단위근 존재 = 비정상)")
    if not stationarity_res:
        st.warning("정상성 검정을 수행할 수 있는 변수가 없습니다.")
    else:
        n_stat = sum(1 for r in stationarity_res.values() if r.is_stationary)
        ss1, ss2 = st.columns(2)
        kpi_card(ss1, "정상성 변수", f"{n_stat} / {len(stationarity_res)}개", color=C_SUCCESS)
        kpi_card(ss2, "비정상성 변수", f"{len(stationarity_res)-n_stat} / {len(stationarity_res)}개", color=C_ANOMALY)
        st.markdown("<br>", unsafe_allow_html=True)

        stat_rows = []
        for col, r in stationarity_res.items():
            stat_rows.append({
                "변수": col, "ADF 통계량": round(r.adf_statistic, 3),
                "p-value": round(r.adf_pvalue, 4),
                "판정": "✅ 정상성" if r.is_stationary else "⚠️ 비정상성",
                "Ljung-Box p": f"{r.ljungbox_pvalue:.4f}" if r.ljungbox_pvalue is not None else "-",
            })
        st.dataframe(pd.DataFrame(stat_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 변수별 해석")
        stat_items = list(stationarity_res.items())
        if "stat_show_all" not in st.session_state:
            st.session_state.stat_show_all = False
        show_items = stat_items if (st.session_state.stat_show_all or len(stat_items) <= 10) else stat_items[:10]
        for col, r in show_items:
            icon = "✅" if r.is_stationary else "⚠️"
            st.markdown(
                f"""<div style="background:{C_BG_CARD};border:1px solid #E2E8F0;border-radius:8px;
                padding:12px 16px;margin:6px 0;font-size:13px">
                <b>{icon} {col}</b> — ADF={r.adf_statistic:.3f}, p={r.adf_pvalue:.4f}<br>
                {r.interpretation}
                </div>""", unsafe_allow_html=True)
        if len(stat_items) > 10 and not st.session_state.stat_show_all:
            if st.button(f"▼ 나머지 {len(stat_items) - 10}개 더 보기", key="stat_show_more_btn"):
                st.session_state.stat_show_all = True
                st.rerun()

st.markdown("---")
st.caption("다변량 시계열 이상탐지 대시보드 · Darts ForecastingAnomalyModel + Isolation Forest + STL + ADF")
