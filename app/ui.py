from __future__ import annotations

import calendar
import html
import math

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from predictor import LABELS

RAIN, DRY = "#38bdf8", "#fbbf24"
MONTHS = {m: calendar.month_name[m] for m in range(1, 13)}
# farming region shown under each city
CITY_META = {
    "Karachi": "Lower Sindh · coastal farming",
    "Lahore": "Punjab · irrigated plains",
    "Islamabad": "Pothohar · rain-fed (barani) farming",
    "Quetta": "Balochistan · orchards",
}


# remove extra spaces from HTML
def compact(h: str) -> str:
    return " ".join(line.strip() for line in h.strip().splitlines())

def esc(x) -> str:
    return html.escape(str(x))


# convert wind degrees to a compass direction
def compass(deg: float) -> str:
    pts = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return pts[int(((deg % 360) + 22.5) // 45) % 8]


# circular gauge showing the rain probability
def gauge(p: float, thr: float, rain: bool) -> str:
    r, c = 62, 2 * math.pi * 62
    color = RAIN if rain else DRY
    a = 2 * math.pi * thr - math.pi / 2
    x1, y1 = 80 + 52 * math.cos(a), 80 + 52 * math.sin(a)
    x2, y2 = 80 + 74 * math.cos(a), 80 + 74 * math.sin(a)
    return f"""
<svg class="rc-gauge" viewBox="0 0 160 160" width="168" height="168" aria-label="{p:.0%} chance of rain">
  <circle cx="80" cy="80" r="{r}" stroke="rgba(148,163,184,.18)" stroke-width="14" fill="none"/>
  <circle cx="80" cy="80" r="{r}" stroke="{color}" stroke-width="14" fill="none" stroke-linecap="round"
          stroke-dasharray="{p * c:.1f} {c:.1f}" transform="rotate(-90 80 80)" class="rc-arc"/>
  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#e2e8f0" stroke-width="3" stroke-linecap="round"/>
  <text x="80" y="82" text-anchor="middle" class="rc-gauge-num">{p * 100:.0f}%</text>
  <text x="80" y="104" text-anchor="middle" class="rc-gauge-lbl">chance of rain</text>
</svg>"""


# main result card: rain or no rain, probability and farm advice
def verdict_card(p: float, thr: float, title_line: str, sub: str) -> str:
    rain = p >= thr
    margin = abs(p - thr)
    conf = "High" if margin >= 0.25 else "Moderate" if margin >= 0.10 else "Low"
    icon, verdict = ("🌧️", "Rain likely tomorrow") if rain else ("☀️", "No rain expected tomorrow")
    advice = ("🚜 Farm advice: skip irrigation, hold off on pesticide spraying and keep harvested crops covered."
              if rain else "🚜 Farm advice: good day to irrigate, spray or dry harvested crops.")
    return f"""
<div class="rc-result {'is-rain' if rain else 'is-dry'}">
  <div class="rc-result-left">{gauge(p, thr, rain)}</div>
  <div class="rc-result-right">
    <div class="rc-eyebrow">{esc(title_line)}</div>
    <div class="rc-verdict"><span class="rc-verdict-icon">{icon}</span>{verdict}</div>
    <div class="rc-sub">{esc(sub)}</div>
    <div class="rc-pills">
      <span class="rc-pill">Confidence: <b>{conf}</b></span>
      <span class="rc-pill">Threshold: <b>{thr:.0%}</b></span>
      <span class="rc-pill">Rain = ≥ 1 mm</span>
    </div>
    <div class="rc-advice">{advice}</div>
  </div>
</div>"""


# six small cards with today's weather values
def tiles(v: dict) -> str:

    def f(key, nd=1):
        x = v.get(key, np.nan)
        return "—" if pd.isna(x) else f"{x:.{nd}f}"

    wd = v.get("wind_direction_10m_dominant", np.nan)
    pc = v.get("pressure_change_24h", np.nan)
    arrow = "" if pd.isna(pc) else ("▲" if pc > 0.3 else "▼" if pc < -0.3 else "▬")
    items = [
        ("🌡️", "Temperature", f"{f('temperature_2m_max', 0)}°" + ("" if pd.isna(v.get("temperature_2m_min", np.nan))
                                 else f" / {f('temperature_2m_min', 0)}°"), "max / min °C"),
        ("💧", "Humidity", f"{f('rh_mean', 0)}%", f"dew point {f('dew_point_mean')} °C"),
        ("☁️", "Cloud cover", f"{f('cloud_cover_mean', 0)}%", f"NASA cloud {f('nasa_cloud_amt', 0)}%"),
        ("🧭", "Pressure", f"{f('pressure_msl_mean', 0)}", f"hPa · 24 h {arrow} {f('pressure_change_24h')}"),
        ("🌬️", "Wind", f"{f('wind_speed_10m_max', 0)} km/h",
         "" if pd.isna(wd) else f"<span class='rc-arrow' style='transform:rotate({wd + 180:.0f}deg)'>↑</span> from {compass(wd)}"),
        ("☔", "Rain", f"{f('precipitation_sum')} mm", f"today · {f('precip_3d_sum')} in 3 d · {f('precip_7d_sum')} in 7 d"),
    ]
    cells = "".join(
        f"<div class='rc-tile'><div class='rc-tile-top'><span>{i}</span>{t}</div>"
        f"<div class='rc-tile-val'>{val}</div><div class='rc-tile-sub'>{s}</div></div>"
        for i, t, val, s in items)
    return f"<div class='rc-tiles'>{cells}</div>"


# bar chart of rain over the last 7 days
def history_bars(hist: pd.DataFrame) -> str:
    mx = max(5.0, float(hist["precipitation_sum"].max() or 0))
    bars = []
    for i, r in hist.iterrows():
        mm = float(r["precipitation_sum"] or 0)
        h = max(3, 100 * mm / mx)
        lbl = "Today" if i == len(hist) - 1 else pd.Timestamp(r["date"]).strftime("%a")
        bars.append(f"<div class='rc-bar-col'><div class='rc-bar-mm'>{mm:.1f}</div>"
                    f"<div class='rc-bar-track'><div class='rc-bar-fill {'wet' if mm >= 1 else ''}' "
                    f"style='height:{h:.0f}%'></div></div><div class='rc-bar-day'>{lbl}</div></div>")
    return (f"<div class='rc-panel'><div class='rc-panel-title'>☔ Rain over the last week (mm)</div>"
            f"<div class='rc-bars'>{''.join(bars)}</div></div>")


def empty_state(text: str, icon: str = "🌦️") -> str:
    return f"<div class='rc-empty'><div class='rc-empty-icon'>{icon}</div><div>{esc(text)}</div></div>"


# chart showing which inputs pushed the prediction up or down
def explain_plot(P, X: pd.DataFrame):
    ex = P.explain(X, k=8).iloc[::-1]
    fig = Figure(figsize=(6.6, 3.8), dpi=110)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111)
    ax.set_facecolor("none")
    colors = [RAIN if v > 0 else DRY for v in ex["effect"]]
    ax.barh(ex["label"], ex["effect"] * 100, color=colors, height=0.62)
    ax.axvline(0, color="#64748b", lw=1)
    for y, (v, val, typ) in enumerate(zip(ex["effect"], ex["value"], ex["typical"])):
        ax.text(v * 100, y, f"  {val:.1f} (usual {typ:.1f})  ", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5, color="#cbd5e1")
    lim = max(5.0, float(np.abs(ex["effect"]).max() * 100) * 1.9)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("effect on rain probability (percentage points)", color="#94a3b8", fontsize=8.5)
    ax.set_title("Why?  blue pushes towards rain · amber towards dry", color="#e2e8f0", fontsize=10, loc="left")
    ax.tick_params(colors="#cbd5e1", labelsize=8.5)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#475569")
    fig.tight_layout()
    return fig


MANUAL_KEYS = ["temperature_2m_max", "rh_mean", "dew_point_mean", "pressure_msl_mean", "pressure_change_24h",
               "cloud_cover_mean", "wind_speed_10m_max", "wind_direction_10m_dominant",
               "precipitation_sum", "precip_3d_sum", "precip_7d_sum"]


# typical values used to pre-fill the manual input form
def typical_values(P, city, month):
    t = P.typical(city, int(month))
    out = []
    for k in MANUAL_KEYS:
        v = t.get(k, 0.0)
        v = 0.0 if v is None or pd.isna(v) else float(v)
        out.append(int(round(v)) if k == "wind_direction_10m_dominant" else round(v, 1))
    k = {name: MANUAL_KEYS.index(name) for name in MANUAL_KEYS}
    out[k["dew_point_mean"]] = min(out[k["dew_point_mean"]], out[k["temperature_2m_max"]])
    out[k["precip_3d_sum"]] = max(out[k["precip_3d_sum"]], out[k["precipitation_sum"]])
    out[k["precip_7d_sum"]] = max(out[k["precip_7d_sum"]], out[k["precip_3d_sum"]])
    return out


# ranked cards for the compare cities tab
def compare_html(results: list[tuple[str, float, dict]], thr: float, when: str, failed: list[str]) -> str:
    results = sorted(results, key=lambda r: r[1], reverse=True)
    cards = []
    for rank, (city, p, v) in enumerate(results, 1):
        region = CITY_META.get(city, "")
        rain = p >= thr
        cards.append(f"""
<div class='rc-cmp {'is-rain' if rain else 'is-dry'}'>
  <div class='rc-cmp-rank'>#{rank}</div>
  <div class='rc-cmp-head'><div class='rc-cmp-city'>{city}</div><div class='rc-cmp-region'>{region}</div></div>
  <div class='rc-cmp-p'>{p:.0%}<span>chance of rain</span></div>
  <div class='rc-cmp-v'>{'🌧️ Rain likely' if rain else '☀️ No rain expected'}</div>
  <div class='rc-cmp-track'><div class='rc-cmp-fill' style='width:{p * 100:.0f}%'></div>
       <div class='rc-cmp-thr' style='left:{thr * 100:.0f}%'></div></div>
  <div class='rc-cmp-facts'>🌡️ {v.get('temperature_2m_max', float('nan')):.0f}°/{v.get('temperature_2m_min', float('nan')):.0f}°
       · 💧 {v.get('rh_mean', float('nan')):.0f}% · ☁️ {v.get('cloud_cover_mean', float('nan')):.0f}%
       · ☔ {v.get('precipitation_sum', float('nan')):.1f} mm today</div>
</div>""")
    top_city, top_p, _ = results[0]
    n_rain = sum(p >= thr for _, p, _ in results)
    summary = (f"🌧️ Rain is likely in <b>{n_rain}</b> of {len(results)} cities." if n_rain
               else "☀️ No city is expected to get rain.")
    note = f"<div class='rc-note'>Live data unavailable for: {', '.join(failed)}.</div>" if failed else ""
    return f"""
<div class='rc-cmp-banner'>
  <div><div class='rc-eyebrow'>Forecast for {when}</div>
       <div class='rc-cmp-summary'>{summary} Highest chance: <b>{top_city}</b> ({top_p:.0%}).</div></div>
  <div class='rc-cmp-legend'><span class='rc-cmp-thr-key'></span> decision threshold {thr:.0%} — above it the model says “rain”</div>
</div>
<div class='rc-cmps'>{''.join(cards)}</div>{note}"""


def stats_strip(P) -> str:
    m = P.meta
    t = m["test_metrics"]
    rows = sum(m["rows"].values())
    items = [(f"{t['F1']:.2f}", "Test F1"), (f"{t['ROC-AUC']:.2f}", "ROC-AUC"),
             (f"{rows:,}", "city-days"), (str(len(m["input_columns"])), "features")]
    return "".join(f"<div class='rc-stat'><b>{a}</b><span>{b}</span></div>" for a, b in items)


# header at the top of the page
def hero(P) -> str:
    drops = "".join(
        f"<i style='left:{(i * 37) % 100}%;animation-delay:{(i * 0.23) % 2:.2f}s;"
        f"animation-duration:{0.9 + (i % 5) * 0.18:.2f}s'></i>" for i in range(46))
    model = P.meta["model_name"] if P else "ML model"
    return f"""
<div class="rc-hero">
  <div class="rc-rain">{drops}</div>
  <div class="rc-hero-inner">
    <div>
      <h1 style="margin-top:0">RainCast <span>Pakistan</span></h1>
      <p>Will it rain tomorrow? Helping <b>farmers</b> decide whether to irrigate, spray or dry crops — a
         {esc(model)} trained on five years of weather from <b>Open-Meteo</b> and <b>NASA POWER</b> for
         Karachi, Lahore, Islamabad and Quetta.</p>
    </div>
    <div class="rc-stats">{stats_strip(P)}</div>
  </div>
</div>"""


# content of the about the model tab
def about_html(P) -> str:
    m = P.meta
    tm, pb = m["test_metrics"], m["test_persistence_baseline"]
    kpis = "".join(
        f"<div class='rc-kpi'><div class='rc-kpi-v'>{tm[k]:.3f}</div><div class='rc-kpi-l'>{k}</div>"
        f"<div class='rc-kpi-b'>baseline {pb.get(k, float('nan')):.3f}</div></div>"
        for k in ["F1", "Precision", "Recall", "ROC-AUC"])
    rows = "".join(
        f"<tr class='{'best' if n == m['model_name'] else ''}'><td>{esc(n)}</td><td>{r['F1']:.3f}</td>"
        f"<td>{r['ROC-AUC']:.3f}</td><td>{r['PR-AUC']:.3f}</td></tr>"
        for n, r in m["validation_comparison"].items())
    feats = "".join(f"<span class='rc-chip'>{esc(LABELS.get(c, c))}</span>" for c in m["input_columns"])
    return f"""
<div class="rc-panel">
  <div class="rc-panel-title">🏆 Test results — 2024, never seen during training</div>
  <div class="rc-kpis">{kpis}</div>
  <div class="rc-note">Baseline = “it will rain tomorrow if it rained today”.
     Model: <b>{esc(m['model_name'])}</b> in a scikit-learn Pipeline (feature engineering → outlier clipping →
     median imputation → scaling + one-hot city). Threshold <b>{P.threshold:.2f}</b> tuned on 2023.</div>
</div>
<div class="rc-grid2">
  <div class="rc-panel">
    <div class="rc-panel-title">📊 All models — validation year 2023</div>
    <table class="rc-table"><tr><th>Model</th><th>F1 @0.5</th><th>ROC-AUC</th><th>PR-AUC</th></tr>{rows}</table>
  </div>
  <div class="rc-panel">
    <div class="rc-panel-title">🧩 The {len(m['input_columns'])} input features</div>
    <div class="rc-chips">{feats}</div>
    <div class="rc-note">Split by time: train {esc(m['split']['train'])} · validation {esc(m['split']['validation'])}
      · test {esc(m['split']['test'])}.</div>
  </div>
</div>
<div class="rc-panel">
  <div class="rc-panel-title">🚜 Who it is for</div>
  <div class="rc-note" style="margin-top:0">Farmers in Pakistan's main farming regions, deciding whether to irrigate,
    spray pesticide or dry a harvest tomorrow. Irrigating before rain wastes water; spraying before rain washes the
    pesticide off.</div>
  <table class="rc-table" style="margin-top:10px"><tr><th>City</th><th>Farming region</th><th>Why rain matters</th></tr>
    <tr><td>Lahore</td><td>Punjab irrigated plains</td><td>irrigation &amp; spraying; monsoon rice and cotton</td></tr>
    <tr><td>Karachi</td><td>Lower Sindh coast</td><td>coastal vegetable farming; monsoon rain</td></tr>
    <tr><td>Islamabad</td><td>Pothohar plateau (barani)</td><td>rain-fed wheat depends almost entirely on rain</td></tr>
    <tr><td>Quetta</td><td>Balochistan valleys</td><td>apple &amp; grape orchards depend on winter rain</td></tr></table>
</div>
<div class="rc-panel">
  <div class="rc-panel-title">📚 Data sources &amp; limitations</div>
  <ul class="rc-list">
    <li><a href="https://open-meteo.com/" target="_blank">Weather data by Open-Meteo.com</a> — Historical Weather API
        (ERA5 reanalysis) and the live Forecast API, licensed
        <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">CC BY 4.0</a>; aggregated from hourly to daily.</li>
    <li>NASA Langley Research Center (LaRC) POWER Project, funded through the NASA Earth Science Division. In live and
        manual mode the 5 NASA inputs are estimated from the matching Open-Meteo values.</li>
    <li>Grid-based reanalysis (~25 km), not rain-gauge data · 4 cities only · not an official forecast — follow the
        Pakistan Meteorological Department for warnings.</li>
  </ul>
</div>"""


FOOTER = """
<div class="rc-footer">
  <a href="https://open-meteo.com/" target="_blank">Weather data by Open-Meteo.com</a> ·
  Educational project, not an official forecast.
</div>
"""


# styling for the whole app
CSS = """@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
/* ---------- page (Streamlit) ---------- */
html, body, .stApp, .stMarkdown, button, input, label, p {font-family:'Plus Jakarta Sans', system-ui, sans-serif;}
.stApp {background: radial-gradient(1200px 600px at 10% -10%, rgba(14,165,233,.14), transparent 60%),
                    radial-gradient(900px 500px at 100% 0%, rgba(99,102,241,.14), transparent 60%), #070d1a;}
.block-container, [data-testid="stMainBlockContainer"] {max-width:1220px; padding-top:1.6rem; padding-bottom:1.2rem !important;}
header[data-testid="stHeader"] {background:transparent;}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton {display:none !important;}
a {color:#7dd3fc;}
/* tabs */
.stTabs [data-baseweb="tab-list"] {gap:6px;}
.stTabs [data-baseweb="tab"] {font-weight:600; font-size:1rem; padding:10px 14px;}
/* form and inputs */
[data-testid="stForm"] {border:1px solid rgba(148,163,184,.16); border-radius:18px; background:rgba(15,23,42,.55); padding:18px 20px;}
.stButton button[kind="primary"], [data-testid="stFormSubmitButton"] button {
  background:linear-gradient(90deg,#0ea5e9,#6366f1) !important; border:0 !important; color:#fff !important;
  font-weight:700 !important; height:48px; border-radius:14px !important; box-shadow:0 10px 30px rgba(14,165,233,.28);}
[data-testid="stImage"] img {border-radius:14px;}
/* ---------- hero ---------- */
.rc-hero {position:relative; overflow:hidden; border-radius:24px; padding:34px 36px;
  background: linear-gradient(135deg,#0b1a36 0%,#13315f 45%,#0c4a6e 100%);
  border:1px solid rgba(125,211,252,.18); box-shadow:0 20px 60px rgba(2,6,23,.55);}
.rc-hero-inner {position:relative; z-index:2; display:flex; gap:28px; align-items:flex-end; justify-content:space-between; flex-wrap:wrap;}
.rc-hero h1 {font-size:2.7rem; line-height:1.05; margin:10px 0 10px; color:#fff; letter-spacing:-.03em; font-weight:800;}
.rc-hero h1 span {background:linear-gradient(90deg,#7dd3fc,#a5b4fc); -webkit-background-clip:text; background-clip:text; color:transparent;}
.rc-hero p {color:#cbd5e1; max-width:610px; margin:0; font-size:1.02rem; line-height:1.55;}
.rc-badge {display:inline-block; font-size:.74rem; letter-spacing:.06em; text-transform:uppercase; color:#bae6fd;
  background:rgba(56,189,248,.12); border:1px solid rgba(56,189,248,.3); padding:5px 11px; border-radius:999px;}
.rc-stats {display:grid; grid-template-columns:repeat(4,minmax(92px,1fr)); gap:10px;}
.rc-stat {background:rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.12); border-radius:16px; padding:12px 14px;
  backdrop-filter:blur(6px); text-align:center;}
.rc-stat b {display:block; font-size:1.45rem; color:#fff;}
.rc-stat span {font-size:.74rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.05em;}
.rc-rain {position:absolute; inset:0; z-index:1; pointer-events:none;}
.rc-rain i {position:absolute; top:-20%; width:1.5px; height:46px; border-radius:2px;
  background:linear-gradient(to bottom, rgba(186,230,253,0), rgba(186,230,253,.55));
  animation-name:rcfall; animation-timing-function:linear; animation-iteration-count:infinite;}
@keyframes rcfall {to {transform:translateY(520px);}}
/* ---------- result card ---------- */
.rc-result {display:flex; gap:22px; align-items:center; padding:20px 24px; border-radius:22px;
  border:1px solid rgba(148,163,184,.18); background:rgba(15,23,42,.75); margin-bottom:14px;}
.rc-result.is-rain {background:linear-gradient(135deg,rgba(14,165,233,.20),rgba(30,58,138,.35)); border-color:rgba(56,189,248,.45);}
.rc-result.is-dry {background:linear-gradient(135deg,rgba(251,191,36,.16),rgba(120,53,15,.25)); border-color:rgba(251,191,36,.4);}
.rc-gauge-num {font-size:30px; font-weight:800; fill:#fff;}
.rc-gauge-lbl {font-size:10.5px; fill:#94a3b8; letter-spacing:.04em;}
.rc-arc {animation:rcarc 1s ease-out;}
@keyframes rcarc {from {stroke-dasharray:0 400;}}
.rc-eyebrow {color:#94a3b8; font-size:.86rem; font-weight:600; letter-spacing:.02em;}
.rc-verdict {font-size:1.75rem; font-weight:800; color:#fff; margin:4px 0 2px; letter-spacing:-.02em;}
.rc-verdict-icon {margin-right:10px;}
.rc-sub {color:#cbd5e1; font-size:.95rem;}
.rc-pills {display:flex; gap:8px; flex-wrap:wrap; margin:12px 0 10px;}
.rc-pill {font-size:.8rem; color:#cbd5e1; background:rgba(2,6,23,.35); border:1px solid rgba(148,163,184,.22);
  padding:4px 10px; border-radius:999px;}
.rc-advice {color:#e2e8f0; font-size:.9rem; opacity:.9;}
/* ---------- weather tiles ---------- */
.rc-tiles {display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-bottom:14px;}
.rc-tile {background:rgba(15,23,42,.72); border:1px solid rgba(148,163,184,.15); border-radius:16px; padding:12px 14px;}
.rc-tile-top {color:#94a3b8; font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; display:flex; gap:6px; align-items:center;}
.rc-tile-val {font-size:1.35rem; font-weight:800; color:#f8fafc; margin-top:4px;}
.rc-tile-sub {font-size:.78rem; color:#94a3b8;}
.rc-arrow {display:inline-block; color:#7dd3fc; font-weight:800;}
/* ---------- panels, bars ---------- */
.rc-panel {background:rgba(15,23,42,.72); border:1px solid rgba(148,163,184,.15); border-radius:18px; padding:16px 18px; margin-bottom:14px; color:#e2e8f0;}
.rc-panel-title {font-weight:700; color:#e2e8f0; margin-bottom:12px;}
.rc-bars {display:flex; gap:10px; align-items:flex-end; height:150px;}
.rc-bar-col {flex:1; display:flex; flex-direction:column; align-items:center; height:100%;}
.rc-bar-mm {font-size:.72rem; color:#94a3b8; height:16px;}
.rc-bar-track {flex:1; width:100%; max-width:34px; display:flex; align-items:flex-end; background:rgba(148,163,184,.08); border-radius:8px;}
.rc-bar-fill {width:100%; border-radius:8px; background:rgba(148,163,184,.35);}
.rc-bar-fill.wet {background:linear-gradient(to top,#0284c7,#38bdf8);}
.rc-bar-day {font-size:.74rem; color:#cbd5e1; margin-top:6px;}
.rc-empty {text-align:center; padding:58px 20px; color:#94a3b8; border:1px dashed rgba(148,163,184,.28);
  border-radius:22px; background:rgba(15,23,42,.45);}
.rc-empty-icon {font-size:3rem; margin-bottom:8px;}
/* ---------- compare tab ---------- */
.rc-cmp-banner {display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;
  padding:16px 20px; border-radius:18px; margin-bottom:14px;
  background:linear-gradient(135deg,rgba(14,165,233,.14),rgba(99,102,241,.12)); border:1px solid rgba(125,211,252,.22);}
.rc-cmp-summary {font-size:1.15rem; color:#f8fafc; margin-top:4px;}
.rc-cmp-legend {font-size:.82rem; color:#94a3b8; display:flex; align-items:center; gap:8px;}
.rc-cmp-thr-key {display:inline-block; width:3px; height:16px; background:#e2e8f0; border-radius:2px;}
.rc-cmps {display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:12px;}
.rc-cmp {position:relative; border-radius:20px; padding:18px; border:1px solid rgba(148,163,184,.16);
  background:rgba(15,23,42,.75);}
.rc-cmp.is-rain {border-color:rgba(56,189,248,.55); background:linear-gradient(160deg,rgba(14,165,233,.22),rgba(15,23,42,.8));}
.rc-cmp.is-dry {border-color:rgba(251,191,36,.35);}
.rc-cmp-rank {position:absolute; top:14px; right:16px; font-size:.8rem; font-weight:800; color:#64748b;}
.rc-cmp-city {font-weight:800; font-size:1.1rem; color:#fff;}
.rc-cmp-region {font-size:.76rem; color:#94a3b8;}
.rc-cmp-p {font-size:2.6rem; font-weight:800; color:#fff; margin:12px 0 0; line-height:1;}
.rc-cmp-p span {display:block; font-size:.74rem; font-weight:600; color:#94a3b8; letter-spacing:.04em; margin-top:4px;}
.rc-cmp-v {margin-top:10px; font-weight:700; color:#e2e8f0;}
.rc-cmp.is-rain .rc-cmp-v {color:#7dd3fc;}
.rc-cmp.is-dry .rc-cmp-v {color:#fcd34d;}
.rc-cmp-track {position:relative; height:8px; border-radius:8px; background:rgba(148,163,184,.18); margin:12px 0 10px;}
.rc-cmp-fill {height:100%; border-radius:8px; background:linear-gradient(90deg,#fbbf24,#38bdf8);}
.rc-cmp-thr {position:absolute; top:-5px; width:3px; height:18px; background:#e2e8f0; border-radius:2px;}
.rc-cmp-facts {font-size:.78rem; color:#94a3b8; line-height:1.5;}
/* ---------- about ---------- */
.rc-kpis {display:grid; grid-template-columns:repeat(4,1fr); gap:10px;}
.rc-kpi {border-radius:16px; padding:14px; background:linear-gradient(135deg,rgba(14,165,233,.14),rgba(99,102,241,.12));
  border:1px solid rgba(125,211,252,.22); text-align:center;}
.rc-kpi-v {font-size:1.8rem; font-weight:800; color:#fff;}
.rc-kpi-l {color:#7dd3fc; font-weight:700; font-size:.85rem;}
.rc-kpi-b {color:#94a3b8; font-size:.75rem;}
.rc-note {color:#94a3b8; font-size:.86rem; margin-top:12px; line-height:1.55;}
.rc-grid2 {display:grid; grid-template-columns:1fr 1fr; gap:14px;}
.rc-table {width:100%; border-collapse:collapse; font-size:.9rem;}
.rc-table th {text-align:left; color:#94a3b8; font-weight:600; padding:7px 8px; border-bottom:1px solid rgba(148,163,184,.2);}
.rc-table td {padding:7px 8px; border-bottom:1px solid rgba(148,163,184,.08); color:#e2e8f0;}
.rc-table tr.best td {color:#7dd3fc; font-weight:700;}
.rc-chips {display:flex; flex-wrap:wrap; gap:6px;}
.rc-chip {font-size:.78rem; padding:4px 10px; border-radius:999px; background:rgba(56,189,248,.1);
  border:1px solid rgba(56,189,248,.25); color:#bae6fd;}
.rc-list {margin:0; padding-left:18px; color:#cbd5e1; line-height:1.7; font-size:.9rem;}
.rc-section {font-weight:700; color:#7dd3fc; font-size:.82rem; text-transform:uppercase; letter-spacing:.06em; margin:6px 0 -4px;}
.rc-footer {text-align:center; color:#64748b; font-size:.8rem; margin:6px 0 0;}
@media (max-width: 820px) {
  .rc-tiles, .rc-cmps, .rc-kpis {grid-template-columns:repeat(2,1fr);}
  .rc-grid2 {grid-template-columns:1fr;}
  .rc-result {flex-direction:column; text-align:center;}
  .rc-pills {justify-content:center;}
  .rc-hero h1 {font-size:2.1rem;}
  .rc-stats {width:100%; grid-template-columns:repeat(2,1fr);}
}
@media (max-width: 560px) {
  .rc-hero {padding:24px 20px;}
  .rc-hero h1 {font-size:1.8rem;}
  .rc-cmps {grid-template-columns:1fr;}
  .rc-stat {padding:10px 8px;}
  .rc-stat b {font-size:1.25rem;}
  .rc-stat span {white-space:nowrap; font-size:.7rem;}
}
"""
