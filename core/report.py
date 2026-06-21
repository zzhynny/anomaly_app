"""
report.py
분석 결과 전체(데이터 개요, 품질, STL 분해, 정상성, 이상탐지, 평가지표)를
하나의 다운로드 가능한 HTML 파일로 만든다.

다른 모듈들의 결과 객체를 입력으로 받아 조립만 하므로, 각 모듈의 내부 로직에는
영향을 주지 않는다 (순수 리포팅 레이어).
"""
from __future__ import annotations

import html as html_lib
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go

from core.decomposition import STLResult
from core.detection import DetectionResult, SCORER_REGISTRY
from core.explain import AnomalyExplanation
from core.quality import QualityReport
from core.stationarity import StationarityResult

REPORT_CSS = """
<style>
  body { font-family: -apple-system, 'Segoe UI', 'Noto Sans KR', sans-serif; margin: 0; padding: 40px;
         background: #f4f6f9; color: #1a1f29; line-height: 1.6; }
  .container { max-width: 1080px; margin: 0 auto; }
  h1 { font-size: 26px; margin-bottom: 4px; }
  h2 { font-size: 19px; margin-top: 40px; padding-bottom: 8px; border-bottom: 2px solid #2E5EAA; color: #2E5EAA; }
  h3 { font-size: 15px; margin-top: 24px; color: #333; }
  .subtitle { color: #666; font-size: 13px; margin-bottom: 28px; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; background: white; }
  th, td { padding: 8px 12px; border: 1px solid #e0e3e8; text-align: left; }
  th { background: #eef1f6; font-weight: 600; }
  .card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }
  .card { background: white; border: 1px solid #e0e3e8; border-radius: 10px; padding: 14px 16px; }
  .card-label { font-size: 11px; color: #888; margin-bottom: 4px; }
  .card-value { font-size: 20px; font-weight: 700; color: #1a1f29; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
  .badge-ok { background: #e6f4ea; color: #1e7a3c; }
  .badge-warn { background: #fdecea; color: #c0392b; }
  .explanation-card { background: white; border-left: 4px solid #E2543D; border-radius: 6px;
                       padding: 12px 16px; margin: 10px 0; font-size: 13px; }
  .footer { margin-top: 50px; font-size: 11px; color: #999; text-align: center; }
  .chart-wrap { background: white; border-radius: 10px; padding: 10px; margin: 12px 0; }
</style>
"""


def _fig_to_div(fig: go.Figure, include_js: bool = False) -> str:
    return fig.to_html(include_plotlyjs="cdn" if include_js else False, full_html=False)


def _quality_section(quality, source_label: str) -> str:
    return f"""
    <h2>1. 데이터 개요 및 품질 분석</h2>
    <p>분석 대상 파일: <strong>{html_lib.escape(source_label)}</strong></p>
    <div class="card-grid">
      <div class="card"><div class="card-label">총 행 수</div><div class="card-value">{quality.n_rows:,}</div></div>
      <div class="card"><div class="card-label">총 컬럼 수</div><div class="card-value">{quality.n_cols}</div></div>
      <div class="card"><div class="card-label">분석 기간</div><div class="card-value" style="font-size:14px">{quality.duration_str}</div></div>
      <div class="card"><div class="card-label">추정 주기</div><div class="card-value" style="font-size:16px">{quality.freq_guess or '불규칙'}</div></div>
      <div class="card"><div class="card-label">결측치</div><div class="card-value" style="font-size:16px">{quality.n_missing_total:,}개 ({quality.missing_ratio_total*100:.2f}%)</div></div>
      <div class="card"><div class="card-label">중복 타임스탬프</div><div class="card-value">{quality.n_duplicated_timestamps:,}</div></div>
    </div>
    <p>시작일: {quality.start_date} &nbsp;|&nbsp; 종료일: {quality.end_date}</p>
    <h3>변수별 기술통계</h3>
    {quality.descriptive_stats.round(4).to_html(classes='', border=0)}
    """


def _stl_section(stl_results: dict[str, STLResult]) -> str:
    rows = ""
    charts = ""
    for col, r in stl_results.items():
        rows += f"<tr><td>{html_lib.escape(col)}</td><td>{r.period}</td><td>{r.trend_strength:.3f}</td><td>{r.seasonal_strength:.3f}</td></tr>"

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=r.observed.index, y=r.observed.values, name="원본", line=dict(color="#999")))
        fig.add_trace(go.Scatter(x=r.trend.index, y=r.trend.values, name="Trend", line=dict(color="#2E5EAA")))
        fig.update_layout(title=f"{col} — Trend", height=220, margin=dict(t=30, b=10, l=10, r=10), template="plotly_white")
        charts += f'<div class="chart-wrap"><h3>{html_lib.escape(col)} (계절 주기={r.period})</h3>{_fig_to_div(fig)}</div>'

    return f"""
    <h2>2. 시계열 분석 (STL Decomposition)</h2>
    <p>추세(Trend)와 계절성(Seasonal)을 STL(Loess 기반 분해법)로 분리하여 분석했습니다.
    강도는 0~1 범위로, 1에 가까울수록 해당 성분이 시계열 변동을 더 많이 설명합니다.</p>
    <table><tr><th>변수</th><th>계절 주기</th><th>추세 강도</th><th>계절성 강도</th></tr>{rows}</table>
    {charts}
    """


def _stationarity_section(stat_results: dict[str, StationarityResult]) -> str:
    rows = ""
    for col, r in stat_results.items():
        badge = '<span class="badge badge-ok">정상성 있음</span>' if r.is_stationary else '<span class="badge badge-warn">비정상성</span>'
        lb_display = f"{r.ljungbox_pvalue:.4f}" if r.ljungbox_pvalue is not None else "-"
        rows += (
            f"<tr><td>{html_lib.escape(col)}</td><td>{r.adf_statistic:.3f}</td>"
            f"<td>{r.adf_pvalue:.4f}</td><td>{badge}</td>"
            f"<td>{lb_display}</td></tr>"
        )
    return f"""
    <h2>3. 정상성 분석 (ADF Test)</h2>
    <p>판정 기준: <strong>p-value &lt; 0.05 → 정상성 있음</strong> (귀무가설: 단위근이 존재한다 = 비정상성이다)</p>
    <table><tr><th>변수</th><th>ADF 통계량</th><th>p-value</th><th>판정</th><th>Ljung-Box p-value</th></tr>{rows}</table>
    """


def _detection_section(detection: DetectionResult, charts_html: str) -> str:
    n_total = len(detection.combined_binary)
    n_anomaly = int(detection.combined_binary.values().sum())
    scorer_list = ", ".join(SCORER_REGISTRY[k]["label"] for k in detection.scorer_keys)
    return f"""
    <h2>4. 이상탐지 결과</h2>
    <p>사용된 지표: {html_lib.escape(scorer_list)}</p>
    <div class="card-grid">
      <div class="card"><div class="card-label">탐지 구간 길이</div><div class="card-value">{n_total:,}</div></div>
      <div class="card"><div class="card-label">이상 판정 시점</div><div class="card-value">{n_anomaly:,}</div></div>
      <div class="card"><div class="card-label">이상 비율</div><div class="card-value">{n_anomaly/n_total*100:.1f}%</div></div>
    </div>
    {charts_html}
    """


def _explanation_section(explanations: list[AnomalyExplanation]) -> str:
    cards = ""
    for e in explanations:
        lines = "<br>".join(html_lib.escape(line) for line in e.summary_lines)
        cards += f'<div class="explanation-card">{lines}</div>'
    return f"""
    <h2>5. 주요 이상 시점 설명</h2>
    {cards if cards else '<p>설명할 만한 이상 시점이 충분히 탐지되지 않았습니다.</p>'}
    """


def _evaluation_section(eval_rows: list[dict]) -> str:
    if not eval_rows:
        return ""
    df = pd.DataFrame(eval_rows)
    return f"""
    <h2>6. 평가 지표</h2>
    {df.to_html(index=False, border=0)}
    """


def build_html_report(
    source_label: str,
    quality: QualityReport,
    stl_results: dict[str, STLResult],
    stationarity_results: dict[str, StationarityResult],
    detection: DetectionResult,
    detection_charts_html: str,
    explanations: list[AnomalyExplanation],
    eval_rows: list[dict],
) -> str:
    """모든 분석 결과를 받아 하나의 self-contained HTML 문자열로 합친다."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    body = (
        _quality_section(quality, source_label)
        + _stl_section(stl_results)
        + _stationarity_section(stationarity_results)
        + _detection_section(detection, detection_charts_html)
        + _explanation_section(explanations)
        + _evaluation_section(eval_rows)
    )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>시계열 이상탐지 분석 리포트</title>
{REPORT_CSS}
</head>
<body>
<div class="container">
  <h1>시계열 이상탐지 종합 분석 리포트</h1>
  <p class="subtitle">생성 시각: {generated_at} · 데이터 품질 분석 → STL 분해 → 정상성 검정 → 이상탐지 → 평가지표 순으로 구성됨</p>
  {body}
  <div class="footer">Generated by 다변량 시계열 이상탐지 대시보드 (Darts ForecastingAnomalyModel + Isolation Forest)</div>
</div>
</body>
</html>"""
