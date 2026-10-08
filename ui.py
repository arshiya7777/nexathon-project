"""Look and feel of the workbench: one CSS block plus small functions that return HTML strings.
No Streamlit import here, so everything can be rendered and checked in a plain browser."""
import html as _html

INK = "#2b1f16"        # deep umber (text, masthead)
PAPER = "#f2f4f0"      # cool paper
PANEL = "#ffffff"
LINE = "#d9ddd3"
GREEN = "#2e7d4f"
BLUE = "#2f6db5"
BROWN = "#7a4e2d"
RED = "#b3402a"
AMBER = "#b9811a"
GREY = "#8a9086"

TONES = {"high": RED, "medium": AMBER, "low": GREY, "green": GREEN, "blue": BLUE, "brown": BROWN,
         "open": BLUE, "confirmed fraud": RED, "dismissed": GREY, "escalated": AMBER}

CSS = f"""
<style>
:root {{ --ink:{INK}; --paper:{PAPER}; --line:{LINE}; --green:{GREEN}; --blue:{BLUE}; --brown:{BROWN}; --red:{RED}; --amber:{AMBER}; }}
.stApp {{ background: var(--paper); color: var(--ink); font-family: "Segoe UI", system-ui, -apple-system, Roboto, sans-serif; }}
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu {{ display: none !important; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.2rem; max-width: 1280px; }}
[data-testid="stSidebar"] {{ background: #e8ebe3; border-right: 1px solid var(--line); }}
h1, h2, h3, h4 {{ color: var(--ink); letter-spacing: -0.01em; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 6px; font-weight: 600; border: 1px solid var(--line); }}
.stButton > button[kind="primary"] {{ background: var(--green); border-color: var(--green); color: #fff; }}
.stButton > button[kind="primary"]:hover {{ background: #256a41; border-color: #256a41; }}

.mast {{ background: var(--ink); color: #f4efe8; border-radius: 8px; padding: 18px 24px; margin-bottom: 18px;
        display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; flex-wrap: wrap; }}
.mast .t {{ font-family: Georgia, "Times New Roman", serif; font-size: 1.7rem; line-height: 1.15; margin: 0; }}
.mast .s {{ margin: 6px 0 0; color: #cfc6b8; font-size: 0.95rem; max-width: 62ch; }}
.mast .r {{ text-align: right; color: #cfc6b8; font-size: 0.85rem; line-height: 1.5; }}
.mast .r b {{ color: #fff; font-weight: 600; }}

.kpi {{ background: var(--panel, #fff); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; min-height: 104px; box-sizing: border-box; }}
.kpi .l {{ font-size: 0.85rem; color: #5e665a; }}
.kpi .v {{ font-family: Georgia, "Times New Roman", serif; font-size: 2rem; line-height: 1.15; margin-top: 2px; }}
.kpi .n {{ font-size: 0.82rem; color: #5e665a; margin-top: 2px; }}
.kpi.accent {{ border-left: 4px solid var(--green); }}
.kpi.warn {{ border-left: 4px solid var(--red); }}

.panel {{ background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; margin-bottom: 14px; }}
.panel h4 {{ margin: 0 0 4px; font-size: 1.05rem; }}
.panel .sub {{ color: #5e665a; font-size: 0.88rem; margin-bottom: 10px; }}

.pill {{ display: inline-block; padding: 1px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; color: #fff; white-space: nowrap; }}

.callout {{ border-left: 4px solid var(--blue); background: #fff; border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 8px 0 14px;
           border-top: 1px solid var(--line); border-right: 1px solid var(--line); border-bottom: 1px solid var(--line); font-size: 0.93rem; }}

table.tbl {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
table.tbl th {{ text-align: left; font-weight: 600; color: #5e665a; border-bottom: 1px solid var(--line); padding: 6px 8px; }}
table.tbl td {{ padding: 7px 8px; border-bottom: 1px solid #eceee8; }}
table.tbl td.num, table.tbl th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

.facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 6px 0 4px; }}
.facts div {{ background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }}
.facts span {{ display: block; font-size: 0.8rem; color: #5e665a; }}
.facts b {{ font-family: Georgia, serif; font-size: 1.3rem; font-weight: 600; }}

ul.why {{ list-style: none; padding: 0; margin: 0; }}
ul.why li {{ display: flex; gap: 10px; padding: 8px 0; border-bottom: 1px solid #eceee8; font-size: 0.95rem; }}
ul.why li .ic {{ flex: 0 0 26px; height: 26px; border-radius: 6px; background: #e6efe9; color: var(--green); font-weight: 700;
                 display: flex; align-items: center; justify-content: center; font-size: 0.8rem; }}

.rail {{ margin: 6px 0 2px; }}
.rail .bar {{ position: relative; display: flex; height: 14px; border-radius: 7px; overflow: hidden; }}
.rail .bar i {{ display: block; height: 100%; }}
.rail .mk {{ position: absolute; top: -5px; width: 4px; height: 24px; background: var(--ink); border-radius: 2px; }}
.rail .lab {{ display: flex; justify-content: space-between; font-size: 0.78rem; color: #5e665a; margin-top: 4px; }}
.rail .cap {{ font-size: 0.9rem; margin-bottom: 8px; }}

.demo {{ background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px; min-height: 150px; box-sizing: border-box; }}
.demo h4 {{ margin: 0 0 6px; }}
.demo p {{ margin: 0 0 8px; font-size: 0.93rem; color: #3b3f38; }}
.empty {{ text-align: center; padding: 34px 10px; color: #5e665a; background: #fff; border: 1px dashed var(--line); border-radius: 8px; }}
.empty b {{ display: block; color: var(--ink); font-size: 1.05rem; margin-bottom: 4px; }}
</style>
"""


def H(s):
    """Collapse an HTML snippet to one line so Streamlit's markdown never treats it as code."""
    return "".join(line.strip() for line in s.strip().splitlines())


def esc(x):
    return _html.escape(str(x))


def pill(text, tone="blue"):
    col = TONES.get(str(tone).lower(), TONES.get(str(text).lower(), BLUE))
    return f'<span class="pill" style="background:{col}">{esc(text)}</span>'


def masthead(title, subtitle="", right=""):
    return H(f"""<div class="mast"><div><p class="t">{esc(title)}</p><p class="s">{esc(subtitle)}</p></div>
                 <div class="r">{right}</div></div>""")


def kpi(label, value, note="", tone=""):
    return H(f'<div class="kpi {tone}"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div>'
             f'<div class="n">{esc(note)}</div></div>')


def panel(title, body_html, sub=""):
    return H(f'<div class="panel"><h4>{esc(title)}</h4><div class="sub">{esc(sub)}</div>{body_html}</div>')


def callout(text, tone="blue"):
    col = TONES.get(tone, BLUE)
    return H(f'<div class="callout" style="border-left-color:{col}">{text}</div>')


def empty(title, text):
    return H(f'<div class="empty"><b>{esc(title)}</b>{esc(text)}</div>')


def facts(pairs):
    cells = "".join(f"<div><span>{esc(k)}</span><b>{esc(v)}</b></div>" for k, v in pairs)
    return H(f'<div class="facts">{cells}</div>')


_ICONS = (("first time", "N"), ("amount", "$"), ("device", "D"), ("email", "@"), ("hours", "T"), ("card", "C"))


def reasons_html(items):
    out = []
    for it in items:
        low = it.lower()
        ic = next((v for k, v in _ICONS if k in low), "i")
        out.append(f'<li><span class="ic">{ic}</span><span>{esc(it)}</span></li>')
    return H(f'<ul class="why">{"".join(out)}</ul>')


def table_html(columns, rows, num_cols=()):
    """columns: list of header strings. rows: list of lists; cell strings may contain trusted HTML (use pill())."""
    th = "".join(f'<th class="{"num" if c in num_cols else ""}">{esc(c)}</th>' for c in columns)
    body = ""
    for r in rows:
        tds = "".join(f'<td class="{"num" if c in num_cols else ""}">{v}</td>' for c, v in zip(columns, r))
        body += f"<tr>{tds}</tr>"
    return H(f'<table class="tbl"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>')


def triage_rail(rank, n):
    """The queue seen from the top: High (top 0.5%), Medium (next 1.5%), Low (up to 5%); a marker shows this case."""
    frac = rank / max(1, n) * 100
    shown = 5.0
    pos = min(frac, shown) / shown * 100
    cap = f"Rank {rank:,} of {n:,}, top {frac:.2f}% of the queue" if frac < shown else f"Rank {rank:,} of {n:,}, outside the top {shown:.0f}%"
    return H(f"""<div class="rail"><div class="cap">{esc(cap)}</div>
        <div class="bar"><i style="width:10%;background:{RED}"></i><i style="width:30%;background:{AMBER}"></i><i style="width:60%;background:#c9cdc3"></i>
        <span class="mk" style="left:calc({pos:.2f}% - 2px)"></span></div>
        <div class="lab"><span style="width:10%">High</span><span style="width:30%">Medium (to 2%)</span><span style="width:60%;text-align:right">Low (to 5%)</span></div></div>""")


def report_html(title, facts_pairs, reasons, decision=None):
    """Stand-alone printable case report (open in a browser, Print, Save as PDF)."""
    dec = ""
    if decision:
        dec = f"<h2>Analyst decision</h2><p><b>{esc(decision.get('status', ''))}</b>. {esc(decision.get('note', ''))}</p>"
    rows = "".join(f"<tr><td>{esc(k)}</td><td><b>{esc(v)}</b></td></tr>" for k, v in facts_pairs)
    li = "".join(f"<li>{esc(r)}</li>" for r in reasons)
    return (f"<!doctype html><meta charset='utf-8'><title>{esc(title)}</title>"
            "<style>body{font-family:Segoe UI,system-ui,sans-serif;max-width:720px;margin:32px auto;color:#2b1f16;line-height:1.5}"
            "h1{font-family:Georgia,serif}td{padding:4px 18px 4px 0}small{color:#666}</style>"
            f"<h1>{esc(title)}</h1><table>{rows}</table><h2>Why it was flagged</h2><ul>{li}</ul>{dec}"
            "<p><small>Research prototype on IEEE-CIS data. A model score is a reason to look, not proof of fraud.</small></p>")
