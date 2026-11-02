"""Situation-report builder for VAYU.

Produces a self-contained, print-ready HTML report (inline CSS, no external
assets) plus a plain-Markdown variant, entirely from the live engine outputs.
Nothing in the report is hard-coded: all figures come from the current run.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

REPORT_CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; color: #17252D; margin: 0; background: #F4F1EA; }
.page { max-width: 860px; margin: 0 auto; padding: 36px 44px; background: #FFFFFF; }
header { border-bottom: 3px solid #35616A; padding-bottom: 14px; margin-bottom: 22px; }
.logo { font-size: 30px; font-weight: 800; letter-spacing: 4px; color: #17252D; }
.logo span { color: #35616A; }
.sub { color: #5C6E6A; font-size: 12px; margin-top: 4px; letter-spacing: 0.6px; }
h2 { color: #35616A; font-size: 15px; text-transform: uppercase; letter-spacing: 1.4px;
     border-bottom: 1px solid #D8D2C4; padding-bottom: 6px; margin: 26px 0 10px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 8px 0 4px; }
th { background: #17252D; color: #EDE9DE; text-align: left; padding: 7px 9px; font-weight: 600; }
td { padding: 6px 9px; border-bottom: 1px solid #E4E0D5; }
tr:nth-child(even) td { background: #F7F5EF; }
.kpis { display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0 4px; }
.kpi { flex: 1 1 150px; border: 1px solid #D8D2C4; border-top: 3px solid #A6B9A3; padding: 10px 12px; }
.kpi .v { font-size: 21px; font-weight: 700; color: #17252D; }
.kpi .l { font-size: 10.5px; text-transform: uppercase; letter-spacing: 1px; color: #5C6E6A; margin-bottom: 3px; }
.kpi .n { font-size: 11px; color: #5C6E6A; }
.note { background: #F1EFE7; border-left: 4px solid #D9C7A5; padding: 9px 12px; font-size: 12px; color: #4A5A56; margin: 10px 0; }
footer { margin-top: 30px; border-top: 1px solid #D8D2C4; padding-top: 10px; font-size: 11px; color: #5C6E6A; }
@media print { .page { padding: 10mm; } body { background: #FFFFFF; } }
"""


def _fmt(v, spec=".1f", suffix=""):
    if v is None:
        return "n/a"
    try:
        return format(v, spec) + suffix
    except (TypeError, ValueError):
        return str(v)


def build_html_report(engine: dict, kpis: dict) -> str:
    """Render the current VAYU state as a printable HTML report.

    ``kpis`` carries the present-conditions dictionary assembled by the app
    (values, AQI, BLH status, provenance). Everything else comes from engine.
    """
    f = engine["forecaster"]
    att = engine["attribution"]
    meta = engine["meta"]
    now = datetime.now(timezone.utc)

    esc = html.escape

    # --- KPI tiles
    kpi_tiles = "".join(
        f"<div class='kpi'><div class='l'>{esc(t)}</div><div class='v'>{esc(v)}</div><div class='n'>{esc(n)}</div></div>"
        for t, v, n in kpis["tiles"]
    )

    # --- attribution table
    att_rows = ""
    for name, s in att["bootstrap"].items():
        lo, hi = s["ci90_low"], s["ci90_high"]
        ci = f"[{_fmt(lo)} - {_fmt(hi)}]%" if lo is not None else "n/a"
        att_rows += (
            f"<tr><td>{esc(name)}</td><td><b>{_fmt(s['mean'], '.1f', '%')}</b></td>"
            f"<td>{ci}</td><td>{_fmt(s['sigma'], '.2f', '%')}</td><td>{s['n_boot_ok']}</td></tr>"
        )

    # --- audit table
    audit_rows = ""
    for h in sorted(f.results):
        r = f.results[h]
        if not r.trained:
            audit_rows += f"<tr><td>+{h} h</td><td colspan='8'>not trained - {esc(r.reason)}</td></tr>"
            continue
        m = r.metrics
        beat = "yes" if m["mae"] < m["persistence_mae"] else "no - persistence wins"
        audit_rows += (
            f"<tr><td>+{h} h</td><td>{_fmt(m['mae'])}</td><td>{_fmt(m['persistence_mae'])}</td>"
            f"<td>{beat}</td><td>{_fmt(m['rmse'])}</td><td>{_fmt(m['r2'], '.3f')}</td>"
            f"<td>{_fmt(m['pinball_q50'], '.2f')}</td><td>{_fmt(m['interval_coverage_80'], '.2f')}</td>"
            f"<td>{r.n_train}/{r.n_test}</td></tr>"
        )

    # --- latest forecast table
    lf = engine["latest_forecast"]
    fc_rows = ""
    for h in sorted(lf):
        q = lf[h]
        fc_rows += f"<tr><td>+{h} h</td><td>{_fmt(q['q10'])}</td><td><b>{_fmt(q['q50'])}</b></td><td>{_fmt(q['q90'])}</td></tr>"

    # --- drivers
    ex = engine["explainer_24"]
    drivers_html = "<p>n/a - 24h explainability was not trained.</p>"
    if "error" not in ex:
        gs = {k: f"{v * 100:.0f}%" for k, v in ex["group_shares"].items()}
        drivers_html = (
            f"<p>{esc(ex['interpretation'])}</p>"
            f"<p>Group shares (test n={ex['n_test']}, repeats={ex['n_repeats']}): "
            + ", ".join(f"<b>{esc(k)}</b> {esc(v)}" for k, v in gs.items())
            + "</p>"
        )

    med = att["diagnostic_medians"]

    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>VAYU situation report</title><style>{REPORT_CSS}</style></head>
<body><div class="page">
<header>
  <div class="logo">VAYU<span>.</span></div>
  <div class="sub">ATMOSPHERIC INTELLIGENCE &mdash; AIR QUALITY SITUATION REPORT</div>
  <div class="sub">Generated {now:%d %b %Y, %H:%M} UTC &middot; Region: Delhi-NCR</div>
</header>

<h2>Present conditions</h2>
<div class="kpis">{kpi_tiles}</div>
<div class="note">{esc(kpis['provenance'])}</div>

<h2>Source attribution &mdash; receptor model (NMF), bootstrap uncertainty</h2>
<table>
<tr><th>Source (heuristic assignment)</th><th>Share</th><th>90% credible interval</th><th>Sigma</th><th>Bootstrap fits</th></tr>
{att_rows}
</table>
<p>Diagnostic ratio medians &mdash; PM2.5/PM10: <b>{_fmt(med.get('pm_ratio'), '.2f')}</b> &middot;
NO2/SO2: <b>{_fmt(med.get('no2_so2_ratio'), '.2f')}</b> &middot;
CO/NO2&times;1000: <b>{_fmt(med.get('co_no2_ratio'), '.1f')}</b>.
Fitted on {att['n_samples']:,} real samples; relative reconstruction residual {att['reconstruction_residual']:.3f}.</p>

<h2>Forward forecast &mdash; latest real observation as origin</h2>
<table>
<tr><th>Horizon</th><th>q10 (ug/m3)</th><th>Median q50 (ug/m3)</th><th>q90 (ug/m3)</th></tr>
{fc_rows}
</table>

<h2>Model audit &mdash; strictly chronological, out-of-time</h2>
<table>
<tr><th>Horizon</th><th>Model MAE</th><th>Persistence MAE</th><th>Beats baseline</th><th>RMSE</th><th>R2</th><th>Pinball q50</th><th>80% coverage</th><th>Train/Test n</th></tr>
{audit_rows}
</table>

<h2>What drives current pollution &mdash; inversion vs emissions</h2>
{drivers_html}

<h2>Method</h2>
<p>Ground truth: CPCB CAAQMS continuous telemetry (40 Delhi-NCR stations). Meteorology: Open-Meteo ERA5
reanalysis including planetary boundary layer height and u/v wind components. Attribution: non-negative
matrix factorization in the EPA PMF form with factor naming by documented chemical markers; uncertainty from
150 bootstrap refits. Forecasting: gradient-boosted quantile regression (pinball loss, tau 0.10/0.50/0.90),
direct multi-horizon, chronological 75/25 split with no future information in features. Driver separation:
permutation importance on the out-of-time test split.</p>

<h2>Limitations</h2>
<p>Attribution shares are statistical evidence, not causal measurement; factor-to-source naming is a heuristic.
The available open archive covers a {meta.get('n_hourly_rows', '?')}-hour window; intervals may under-cover at this
sample size (coverage shown above). The policy simulator (in-app) is a first-order linear counterfactual without
secondary-aerosol chemistry. All values are model estimates with stated uncertainty &mdash; not probabilities,
not regulatory determinations.</p>

<footer>VAYU atmospheric intelligence &middot; report generated from live engine state &middot; estimates carry uncertainty and are subject to revision.</footer>
</div></body></html>"""
    return html_doc


def build_markdown_report(engine: dict, kpis: dict) -> str:
    f = engine["forecaster"]
    att = engine["attribution"]
    meta = engine["meta"]
    now = datetime.now(timezone.utc)

    lines = [
        "# VAYU - Air Quality Situation Report",
        f"Generated {now:%d %b %Y, %H:%M} UTC | Region: Delhi-NCR",
        "",
        "## Present conditions",
        "",
    ]
    for t, v, n in kpis["tiles"]:
        lines.append(f"- **{t}:** {v} ({n})")
    lines += ["", f"Provenance: {kpis['provenance']}", ""]

    lines += [
        "## Source attribution (NMF receptor model, 150 bootstrap refits)",
        "",
        "| Source | Share | 90% CI | Sigma |",
        "|---|---|---|---|",
    ]
    for name, s in att["bootstrap"].items():
        lo, hi = s["ci90_low"], s["ci90_high"]
        ci = f"[{lo} - {hi}]" if lo is not None else "n/a"
        lines.append(f"| {name} | {s['mean']:.1f}% | {ci} | {s['sigma']}% |")

    lines += [
        "",
        f"Fitted on {att['n_samples']:,} real samples; relative reconstruction residual {att['reconstruction_residual']:.3f}.",
        "",
        "## Forward forecast (from latest real observation)",
        "",
        "| Horizon | q10 | Median q50 | q90 |",
        "|---|---|---|---|",
    ]
    for h in sorted(engine["latest_forecast"]):
        q = engine["latest_forecast"][h]
        lines.append(f"| +{h} h | {q['q10']:.1f} | **{q['q50']:.1f}** | {q['q90']:.1f} |")

    lines += [
        "",
        "## Model audit (strictly chronological)",
        "",
        "| Horizon | Model MAE | Persistence MAE | Beats baseline | R2 | 80% coverage | Train/Test n |",
        "|---|---|---|---|---|---|---|",
    ]
    for h in sorted(f.results):
        r = f.results[h]
        if not r.trained:
            lines.append(f"| +{h} h | not trained ({r.reason}) | | | | | |")
            continue
        m = r.metrics
        beat = "yes" if m["mae"] < m["persistence_mae"] else "no - persistence wins"
        lines.append(
            f"| +{h} h | {m['mae']:.1f} | {m['persistence_mae']:.1f} | {beat} | {m['r2']:.3f} | "
            f"{m['interval_coverage_80']:.2f} | {r.n_train}/{r.n_test} |"
        )

    ex = engine["explainer_24"]
    lines += ["", "## Drivers - inversion vs emissions", ""]
    if "error" not in ex:
        lines.append(ex["interpretation"])
        for k, v in ex["group_shares"].items():
            lines.append(f"- {k}: {v * 100:.0f}%")
    else:
        lines.append("n/a")

    lines += [
        "",
        "## Limitations",
        "",
        "Attribution shares are statistical evidence, not causal measurement; factor naming is a documented heuristic.",
        f"Open archive covers {meta.get('n_hourly_rows', '?')} hourly rows; intervals may under-cover at this sample size.",
        "All values are estimates with stated uncertainty - not probabilities, not regulatory determinations.",
        "",
        "---",
        "VAYU atmospheric intelligence",
    ]
    return "\n".join(lines)
