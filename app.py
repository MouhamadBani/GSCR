"""
app.py — International Trade Risk Dashboard (Final)
=====================================================
✅ Real World Bank API data (LIVE → CSV cache → parquet cache → fallback)
✅ All map layers show proper CONTINUOUS color gradients
✅ Risk Cluster map uses distinct categorical colors
✅ KPIs computed directly from real indicator values
✅ ~25% of countries in each risk bucket (percentile thresholds)
✅ Country borders + ocean clearly visible
✅ No pyarrow dependency (CSV cache)
"""

from __future__ import annotations
import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

from data_fetcher import fetch_all_indicators, _get_fallback_data, _read_cache, _write_cache
from ml_models import build_composite_score, cluster_countries, classify_risk, forecast_risk, CLUSTER_NAMES

# ── Color constants ────────────────────────────────────────────────────────
RISK_COLORS = {
    "Low":         "#27ae60",
    "Medium-Low":  "#f39c12",
    "Medium-High": "#e67e22",
    "High":        "#c0392b",
}
CLUSTER_COLORS = {
    "Open & Stable":  "#2980b9",
    "Emerging Risk":  "#f1c40f",
    "High Barrier":   "#e74c3c",
    "Fragile Trade":  "#8e44ad",
}

# Map layer definitions: (display label, colorscale, reverse scale?)
LAYERS = {
    "composite_risk_score": ("Composite Risk Score (0–10)",    "RdYlGn_r", False),
    "trade_openness":       ("Trade Openness (% of GDP)",      "Blues",    False),
    "tariff_rate":          ("Tariff Rate (%)",                 "YlOrRd",   False),
    "lpi_score":            ("Logistics Performance (LPI)",    "Greens",   False),
    "fx_volatility":        ("FX Volatility",                  "Oranges",  False),
    "governance_score":     ("Governance Score (0–10)",        "YlGn",     False),
    "cluster":              ("Risk Cluster",                    None,       False),
}

# ── Data loading chain ─────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, str, str]:
    # 1) Live API
    try:
        df = fetch_all_indicators(use_live_api=True, allow_cache_write=True)
        if len(df) > 20:
            return df, "LIVE", f"✅ Live World Bank data — {len(df)} countries."
    except Exception as e:
        live_err = f"{type(e).__name__}: {str(e)[:80]}"
    else:
        live_err = "returned <20 rows"

    # 2) CSV/parquet cache
    cached = _read_cache()
    if cached is not None and len(cached) > 20:
        return cached, "CACHE", f"📦 Cached data ({len(cached)} countries). Live: {live_err}"

    # 3) Built-in fallback
    df_fb = _get_fallback_data()
    return df_fb, "FALLBACK", f"⚠️ Sample data ({len(df_fb)} countries). Live: {live_err}"


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    num = ["trade_openness","tariff_rate","lpi_score","fx_volatility",
           "governance_score","gdp_growth","inflation"]
    for c in num:
        if c not in df.columns: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        med = df[c].median()
        df[c] = df[c].fillna(med if pd.notna(med) else 0.0)

    df["iso3"]    = df["iso3"].astype(str).str.strip().str.upper()
    df["country"] = df["country"].astype(str).str.strip()
    df = df[df["iso3"].str.len() == 3].dropna(subset=["country"]).reset_index(drop=True)

    df = build_composite_score(df)
    df = cluster_countries(df)
    df = classify_risk(df)
    df["cluster_label"] = df["cluster"].map(CLUSTER_NAMES).fillna("Other")
    return df


def flt(df: pd.DataFrame, risk_cats, clusters) -> pd.DataFrame:
    return df[df["risk_label"].isin(risk_cats) & df["cluster"].isin(clusters)].copy()


def sfmt(v, d=1, s=""):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{v:.{d}f}{s}"


# ── Initial load ───────────────────────────────────────────────────────────
print("⏳  Loading data…")
raw0, src0, msg0 = load_data()
df0 = prepare(raw0)
print(f"\n✅  {len(df0)} countries ready  [{src0}]")
print("── Real indicator stats ──────────────────────────────────────────────")
for col in ["trade_openness","tariff_rate","lpi_score","fx_volatility",
            "governance_score","composite_risk_score"]:
    s = df0[col]
    print(f"  {col:25s}  mean={s.mean():.2f}  min={s.min():.2f}  max={s.max():.2f}")
print("── Risk distribution ─────────────────────────────────────────────────")
print(df0["risk_label"].value_counts().to_string())
print("─────────────────────────────────────────────────────────────────────\n")

# ── App ────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                title="Trade Risk Intelligence")
server = app.server

_src_badge_map = {"LIVE":("success","LIVE"), "CACHE":("warning","CACHE"), "FALLBACK":("danger","FALLBACK")}

SIDEBAR = dbc.Col([
    html.H4("🌐 Trade Risk", className="fw-bold mt-3 mb-1 text-primary"),
    html.Div([
        html.Span("Intelligence Dashboard ", className="text-muted small"),
        dbc.Badge(_src_badge_map[src0][1], id="src-badge",
                  color=_src_badge_map[src0][0]),
    ], className="mb-2"),
    dbc.Alert(msg0, id="src-alert",
              color="info" if src0=="LIVE" else ("warning" if src0=="CACHE" else "danger"),
              className="small py-2 px-2 mb-2"),
    dbc.Button("↻ Refresh Live Data", id="btn-refresh",
               color="primary", outline=True, size="sm", className="mb-3 w-100"),
    html.Hr(),
    html.Label("🗺️ Map Layer", className="small fw-bold"),
    dcc.RadioItems(
        id="map-layer",
        options=[
            {"label": " Composite Risk Score",  "value": "composite_risk_score"},
            {"label": " Trade Openness",         "value": "trade_openness"},
            {"label": " Tariff Rate",            "value": "tariff_rate"},
            {"label": " Logistics (LPI)",        "value": "lpi_score"},
            {"label": " FX Volatility",          "value": "fx_volatility"},
            {"label": " Governance Score",       "value": "governance_score"},
            {"label": " Risk Cluster",           "value": "cluster"},
        ],
        value="composite_risk_score",
        className="small",
        inputStyle={"marginRight": "6px"},
        labelStyle={"display": "block", "marginBottom": "8px"},
    ),
    html.Hr(className="mt-3"),
    html.Label("🚦 Risk Category Filter", className="small fw-bold"),
    dcc.Checklist(
        id="risk-filter",
        options=[{"label": f"  {r}", "value": r} for r in RISK_COLORS],
        value=list(RISK_COLORS.keys()),
        className="small",
        inputStyle={"marginRight": "6px"},
        labelStyle={"display": "block", "marginBottom": "6px"},
    ),
    html.Hr(className="mt-3"),
    html.Label("🔵 Cluster Filter", className="small fw-bold"),
    dcc.Checklist(
        id="cluster-filter",
        options=[{"label": f"  {v}", "value": k} for k, v in CLUSTER_NAMES.items()],
        value=list(CLUSTER_NAMES.keys()),
        className="small",
        inputStyle={"marginRight": "6px"},
        labelStyle={"display": "block", "marginBottom": "6px"},
    ),
    html.Hr(className="mt-3"),
    html.Label("📈 Forecast Country", className="small fw-bold"),
    dcc.Dropdown(
        id="forecast-country",
        options=[{"label": c, "value": c}
                 for c in sorted(df0["country"].dropna().unique())],
        value=df0["country"].iloc[0] if len(df0) else None,
        clearable=False,
    ),
], width=2, style={"backgroundColor":"#f8f9fa","minHeight":"100vh",
                   "padding":"0 14px","borderRight":"1px solid #dee2e6"})


MAIN = dbc.Col([
    dcc.Store(id="store-df",   data=df0.to_json(orient="split")),
    dcc.Store(id="store-meta", data={"source": src0, "message": msg0}),

    # KPI row
    dbc.Row(id="kpi-row", className="g-2 mt-2 mb-3"),

    # World map — big and clear
    dbc.Card(dbc.CardBody(
        dcc.Graph(id="world-map", style={"height":"580px"},
                  config={"displayModeBar": True,
                          "modeBarButtonsToRemove": ["select2d","lasso2d"],
                          "responsive": True})
    ), className="mb-3 shadow-sm"),

    # Scatter + bar
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(id="cluster-scatter", style={"height":"360px"},
                      config={"displayModeBar":False,"responsive":True})
        ), className="shadow-sm"), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(id="risk-bar", style={"height":"360px"},
                      config={"displayModeBar":False,"responsive":True})
        ), className="shadow-sm"), width=6),
    ], className="g-3 mb-3"),

    # Forecast + heatmap
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(id="forecast-chart", style={"height":"330px"},
                      config={"displayModeBar":False,"responsive":True})
        ), className="shadow-sm"), width=6),
        dbc.Col(dbc.Card(dbc.CardBody(
            dcc.Graph(id="corr-heatmap", style={"height":"330px"},
                      config={"displayModeBar":False,"responsive":True})
        ), className="shadow-sm"), width=6),
    ], className="g-3 mb-3"),

    # Data table
    dbc.Card(dbc.CardBody([
        html.H6("📋 Country Risk Table", className="text-primary fw-bold mb-3"),
        dash_table.DataTable(
            id="risk-table", page_size=12,
            sort_action="native", filter_action="native",
            style_table={"overflowX":"auto"},
            style_header={"backgroundColor":"#e9ecef","color":"#212529",
                          "fontWeight":"bold","border":"1px solid #dee2e6"},
            style_cell={"backgroundColor":"#fff","color":"#212529",
                        "border":"1px solid #dee2e6","fontSize":"12px",
                        "padding":"5px 10px","whiteSpace":"nowrap",
                        "textOverflow":"ellipsis","maxWidth":200},
            style_data_conditional=[
                {"if":{"filter_query":'{Risk Level} = "High"'},
                 "backgroundColor":"#fde8e8","color":"#c0392b"},
                {"if":{"filter_query":'{Risk Level} = "Medium-High"'},
                 "backgroundColor":"#fef3e2","color":"#e67e22"},
                {"if":{"filter_query":'{Risk Level} = "Medium-Low"'},
                 "backgroundColor":"#fefde2","color":"#b7950b"},
                {"if":{"filter_query":'{Risk Level} = "Low"'},
                 "backgroundColor":"#e8f8ee","color":"#27ae60"},
                {"if":{"state":"active"},
                 "backgroundColor":"#cfe2ff","border":"1px solid #0d6efd"},
            ],
        ),
    ]), className="shadow-sm mb-4"),
], width=10, style={"padding":"0 18px"})

app.layout = dbc.Container(dbc.Row([SIDEBAR, MAIN]),
                            fluid=True, style={"backgroundColor":"#ffffff"})


# ── Geo style shared by all map callbacks ──────────────────────────────────
def _geos():
    return dict(
        projection_type="natural earth",
        showcountries=True,  countrycolor="#888888", countrywidth=0.4,
        showcoastlines=True, coastlinecolor="#888888", coastlinewidth=0.4,
        showland=True,       landcolor="#f5f5f5",
        showocean=True,      oceancolor="#d0e8f5",
        showlakes=True,      lakecolor="#d0e8f5",
        bgcolor="white",
        lataxis_range=[-60, 85],
    )

def _map_layout(title):
    return dict(
        title=dict(text=title, x=0.01, font=dict(size=13, color="#212529")),
        paper_bgcolor="white", plot_bgcolor="white",
        font_color="#212529",
        margin=dict(l=0, r=0, t=44, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.14,
            xanchor="left", x=0, font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#ddd", borderwidth=1,
        ),
        coloraxis_colorbar=dict(
            thickness=16, len=0.75,
            tickfont=dict(size=11, color="#212529"),
            title=dict(font=dict(size=11, color="#212529")),
            outlinewidth=0,
        ),
    )


# ── Callbacks ───────────────────────────────────────────────────────────────

@app.callback(
    Output("store-df", "data"),
    Output("store-meta", "data"),
    Input("btn-refresh", "n_clicks"),
    prevent_initial_call=True,
)
def refresh(_):
    raw, src, msg = load_data()
    dff = prepare(raw)
    return dff.to_json(orient="split"), {"source": src, "message": msg}


@app.callback(
    Output("src-badge", "children"), Output("src-badge", "color"),
    Output("src-alert", "children"), Output("src-alert", "color"),
    Input("store-meta", "data"),
)
def update_meta(meta):
    src = (meta or {}).get("source", "FALLBACK")
    msg = (meta or {}).get("message", "")
    lbl, col = _src_badge_map.get(src, ("?","secondary"))
    alert_col = "info" if src=="LIVE" else ("warning" if src=="CACHE" else "danger")
    return lbl, col, msg, alert_col


@app.callback(
    Output("kpi-row", "children"),
    Input("store-df", "data"),
    Input("risk-filter", "value"),
    Input("cluster-filter", "value"),
)
def kpis(djson, risk_cats, clusters):
    df = pd.read_json(djson, orient="split")
    fdf = flt(df, risk_cats, clusters)

    n      = len(fdf)
    n_high = int((fdf["risk_label"] == "High").sum())
    t_open = fdf["trade_openness"].mean()   if n else float("nan")
    score  = fdf["composite_risk_score"].mean() if n else float("nan")
    tariff = fdf["tariff_rate"].mean()      if n else float("nan")
    fxvol  = fdf["fx_volatility"].mean()    if n else float("nan")

    cards = [
        ("🌍 Countries",       str(n),                    "primary"),
        ("🔴 High Risk",       str(n_high),               "danger"),
        ("📦 Avg Trade Open.", sfmt(t_open, 1, "%"),       "info"),
        ("📊 Avg Risk Score",  sfmt(score,  2),            "warning"),
        ("🏭 Avg Tariff",      sfmt(tariff, 1, "%"),       "secondary"),
        ("⚡ Avg FX Vol.",     sfmt(fxvol,  2),            "dark"),
    ]
    return [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P(lbl, className="text-muted small mb-1"),
            html.H4(val, className=f"text-{col} fw-bold mb-0"),
        ]), className="shadow-sm text-center"), width=2)
        for lbl, val, col in cards
    ]


@app.callback(
    Output("world-map", "figure"),
    Input("store-df", "data"),
    Input("map-layer", "value"),
    Input("risk-filter", "value"),
    Input("cluster-filter", "value"),
)
def world_map(djson, layer, risk_cats, clusters):
    df  = pd.read_json(djson, orient="split")
    fdf = flt(df, risk_cats, clusters)

    if fdf.empty:
        fig = go.Figure()
        fig.update_geos(**_geos())
        fig.update_layout(**_map_layout("No data matches current filters"))
        return fig

    # ── Cluster map ────────────────────────────────────────────────────────
    if layer == "cluster":
        fig = px.choropleth(
            fdf, locations="iso3", locationmode="ISO-3",
            color="cluster_label",
            color_discrete_map=CLUSTER_COLORS,
            hover_name="country",
            hover_data={
                "risk_label": True,
                "composite_risk_score": ":.2f",
                "trade_openness": ":.1f",
                "tariff_rate": ":.1f",
                "lpi_score": ":.2f",
            },
            category_orders={"cluster_label": list(CLUSTER_NAMES.values())},
        )
        fig.update_geos(**_geos())
        fig.update_layout(**_map_layout("World Map — Risk Clusters"))
        return fig

    # ── Numeric continuous choropleth ──────────────────────────────────────
    if layer not in fdf.columns:
        fig = go.Figure()
        fig.update_geos(**_geos())
        fig.update_layout(**_map_layout(f"Layer '{layer}' not found in data"))
        return fig

    label, cscale, _ = LAYERS.get(layer, (layer, "Blues", False))
    col_data = pd.to_numeric(fdf[layer], errors="coerce")

    # Use 2nd–98th percentile for color range (outliers don't wash out the map)
    vmin = float(col_data.quantile(0.02))
    vmax = float(col_data.quantile(0.98))
    if vmin >= vmax:
        vmin = float(col_data.min())
        vmax = float(col_data.max())

    # Build rich hover
    hover_cols = {}
    for c in ["risk_label","composite_risk_score","trade_openness",
              "tariff_rate","lpi_score","fx_volatility","governance_score"]:
        if c in fdf.columns and c != layer:
            hover_cols[c] = True if c == "risk_label" else ":.2f"

    fig = px.choropleth(
        fdf,
        locations="iso3",
        locationmode="ISO-3",
        color=layer,
        color_continuous_scale=cscale,
        range_color=[vmin, vmax],
        hover_name="country",
        hover_data=hover_cols,
        labels={layer: label},
    )
    layout = _map_layout(f"World Map — {label}")
    layout["coloraxis_colorbar"]["title"] = dict(text=label, font=dict(size=10))
    fig.update_geos(**_geos())
    fig.update_layout(**layout)
    return fig


@app.callback(
    Output("cluster-scatter", "figure"),
    Input("store-df", "data"),
    Input("risk-filter", "value"),
    Input("cluster-filter", "value"),
)
def scatter(djson, risk_cats, clusters):
    fdf = flt(pd.read_json(djson, orient="split"), risk_cats, clusters)
    if fdf.empty:
        return go.Figure()

    fig = px.scatter(
        fdf, x="trade_openness", y="governance_score",
        color="cluster_label", size="composite_risk_score", size_max=22,
        color_discrete_map=CLUSTER_COLORS,
        hover_name="country",
        hover_data={"tariff_rate":":.1f","lpi_score":":.2f",
                    "fx_volatility":":.2f","composite_risk_score":":.2f"},
        title="Trade Openness vs Governance Score",
        labels={"trade_openness":"Trade Openness (%GDP)",
                "governance_score":"Governance (0–10)"},
        category_orders={"cluster_label": list(CLUSTER_NAMES.values())},
    )
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8f9fa", font_color="#212529",
        margin=dict(l=10,r=10,t=50,b=35), legend_title_text="Cluster",
    )
    return fig


@app.callback(
    Output("risk-bar", "figure"),
    Input("store-df", "data"),
    Input("risk-filter", "value"),
    Input("cluster-filter", "value"),
)
def bar(djson, risk_cats, clusters):
    fdf = flt(pd.read_json(djson, orient="split"), risk_cats, clusters)
    if fdf.empty:
        return go.Figure()

    top = (fdf.nlargest(20, "composite_risk_score")
             [["country","composite_risk_score","risk_label"]]
             .sort_values("composite_risk_score", ascending=True))

    fig = px.bar(
        top, x="composite_risk_score", y="country",
        color="risk_label", orientation="h",
        color_discrete_map=RISK_COLORS,
        title="Top 20 Highest Trade Risk Countries",
        labels={"composite_risk_score":"Risk Score (0–10)","country":""},
        category_orders={"risk_label":["Low","Medium-Low","Medium-High","High"]},
    )
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="#f8f9fa", font_color="#212529",
        margin=dict(l=10,r=10,t=50,b=25), legend_title_text="Risk Level",
        xaxis=dict(range=[0,10]),
    )
    return fig


@app.callback(
    Output("forecast-chart", "figure"),
    Input("store-df", "data"),
    Input("forecast-country", "value"),
)
def forecast(djson, country):
    dff = pd.read_json(djson, orient="split")
    if not country:
        return go.Figure()
    try:
        hist, fut = forecast_risk(dff, country)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=str(e), x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["year"], y=hist["composite_risk_score"],
        mode="lines+markers", name="Historical",
        line=dict(color="#2980b9", width=2), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=fut["year"], y=fut["forecast"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#c0392b", width=2, dash="dash"), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fut["year"], fut["year"].iloc[::-1]]),
        y=pd.concat([fut["upper"], fut["lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(192,57,43,0.1)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI",
    ))
    fig.update_layout(
        title=f"Risk Score Forecast — {country}",
        xaxis_title="Year", yaxis_title="Risk Score (0–10)",
        yaxis=dict(range=[0,10]),
        paper_bgcolor="white", plot_bgcolor="#f8f9fa", font_color="#212529",
        margin=dict(l=10,r=10,t=50,b=35), legend_title_text="Series",
    )
    return fig


@app.callback(
    Output("corr-heatmap", "figure"),
    Input("store-df", "data"),
    Input("risk-filter", "value"),
    Input("cluster-filter", "value"),
)
def heatmap(djson, risk_cats, clusters):
    fdf = flt(pd.read_json(djson, orient="split"), risk_cats, clusters)
    cols = ["trade_openness","tariff_rate","lpi_score","fx_volatility",
            "governance_score","composite_risk_score"]
    sub = fdf[cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if sub.empty:
        return go.Figure()

    corr = sub.corr(numeric_only=True).round(2)
    nice = {"trade_openness":"Trade Open.","tariff_rate":"Tariff",
            "lpi_score":"Logistics","fx_volatility":"FX Vol.",
            "governance_score":"Governance","composite_risk_score":"Risk Score"}
    corr.index = corr.columns = [nice[c] for c in corr.columns]

    fig = px.imshow(corr, text_auto=True, aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    title="Indicator Correlation Heatmap")
    fig.update_layout(paper_bgcolor="white", font_color="#212529",
                      margin=dict(l=10,r=10,t=50,b=15))
    return fig


@app.callback(
    Output("risk-table", "data"),
    Output("risk-table", "columns"),
    Input("store-df", "data"),
    Input("risk-filter", "value"),
    Input("cluster-filter", "value"),
)
def table(djson, risk_cats, clusters):
    fdf = flt(pd.read_json(djson, orient="split"), risk_cats, clusters)

    dcols = {
        "country":              "Country",
        "iso3":                 "ISO3",
        "composite_risk_score": "Risk Score",
        "risk_label":           "Risk Level",
        "cluster_label":        "Cluster",
        "trade_openness":       "Trade Open. (%GDP)",
        "tariff_rate":          "Tariff (%)",
        "lpi_score":            "LPI",
        "fx_volatility":        "FX Vol.",
        "governance_score":     "Governance",
    }
    out = fdf[list(dcols.keys())].copy()
    for c in ["composite_risk_score","trade_openness","tariff_rate",
              "lpi_score","fx_volatility","governance_score"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    out.columns = list(dcols.values())

    cols = [{"name":v,"id":v,
             "type":"text" if v in ("Country","ISO3","Risk Level","Cluster") else "numeric"}
            for v in dcols.values()]
    return out.to_dict("records"), cols


import os

server = app.server

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)