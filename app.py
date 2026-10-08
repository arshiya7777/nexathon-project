"""Fraud Investigation Workbench (Streamlit).  Run:  python -m streamlit run app.py
Needs outputs/ made by run_experiment.py (and explore_graph_signal.py for the Signals page)."""
import hashlib
import os

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dashboard_logic as L
import ui

st.set_page_config(page_title="Fraud Investigation Workbench", page_icon="🛡️", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)

BROWN, GREEN, BLUE, RED, GREY = ui.BROWN, ui.GREEN, ui.BLUE, ui.RED, ui.GREY


def md(html):
    st.markdown(html, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data(folder):
    return L.load_outputs(folder)


@st.cache_data(show_spinner="Choosing cases to show...")
def get_demo(view, col):
    return L.demo_cases(view, col)


def short(node_id):
    """Readable short label for the long card / device strings."""
    if ":" in node_id and not node_id[0].isdigit():
        kind, v = node_id.split(":", 1)
        return f"{'dev' if kind == 'device_fp' else 'mail'}-{hashlib.md5(v.encode()).hexdigest()[:4]}"
    return "card-" + hashlib.md5(node_id.encode()).hexdigest()[:4]


def draw_graph(nodes, edges, show_labels=False, height=480):
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"], **n)
    G.add_edges_from(edges)
    if len(G) == 0 or not edges:
        md(ui.empty("No links found", "No other card shares a device or address with this one before this transaction."))
        return
    pos = nx.spring_layout(G, seed=3, k=0.9 / max(1, len(G)) ** 0.5)
    ex, ey = [], []
    for a, b in G.edges:
        ex += [pos[a][0], pos[b][0], None]
        ey += [pos[a][1], pos[b][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="#c3c8bd"), hoverinfo="none", showlegend=False))
    groups = {"investigated card": (BROWN, "star", 22), "other card": (BLUE, "circle", 11), "card with fraud history": (RED, "circle", 13),
              "shared device": (GREEN, "square", 11), "shared email + address": (ui.AMBER, "diamond", 11)}
    buckets = {k: [] for k in groups}
    for nid, d in G.nodes(data=True):
        if d["kind"] == "card":
            key = "investigated card" if d["start"] else ("card with fraud history" if d["fraud"] and show_labels else "other card")
        else:
            key = "shared device" if d["kind"] == "device_fp" else "shared email + address"
        buckets[key].append(nid)
    for key, ids in buckets.items():
        if not ids:
            continue
        col_, sym, size = groups[key]
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in ids], y=[pos[i][1] for i in ids], mode="markers", name=key,
            marker=dict(color=col_, symbol=sym, size=size, line=dict(width=1, color="white")),
            text=[short(i) + (f" ({G.nodes[i]['n_tx']} tx)" if G.nodes[i]["kind"] == "card" else "") for i in ids],
            hoverinfo="text"))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=6, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="white", paper_bgcolor="white", legend=dict(orientation="h", y=-0.02))
    st.plotly_chart(fig, use_container_width=True)


# ---------------- data + sidebar ----------------
st.sidebar.markdown("**Fraud Investigation Workbench**")
folder = st.sidebar.text_input("Outputs folder", "outputs")
D = get_data(folder)
res, scored = D["results"], D["scored"]
models = L.available_models(res, scored)
if not models:
    md(ui.masthead("Fraud Investigation Workbench", "Rank card transactions by risk and investigate who is connected."))
    md(ui.empty("No results yet", "Run `python run_experiment.py --fast` in this folder, then refresh the page."))
    st.stop()
DEC_PATH = os.path.join(folder, "decisions.csv")

PAGES = ["Home", "Queue", "Case", "Alerts", "Budget", "Signals", "Decisions", "Guided demo", "Evidence"]
if "page" not in st.session_state:
    st.session_state.page = "Home"
page = st.sidebar.radio("Go to", PAGES, key="page")
model = st.sidebar.selectbox("Ranking model", models, index=0, format_func=L.pretty)
col = L.score_col(model)

d0, d1 = L.day_range(scored)
n_days = d1 - d0 + 1
replay_n = st.sidebar.slider("Replay: show activity up to day", 1, n_days, n_days)
view = L.replay_view(scored, d0 + replay_n - 1)

analyst = st.sidebar.text_input("Your name", "analyst")
eval_mode = st.sidebar.checkbox("Evaluation mode (reveal true labels)", value=False,
                                help="For judging the system. A real analyst would not see the true label.")


def jump():
    v = st.session_state.get("jump_id", "").strip()
    if v.isdigit() and int(v) in set(scored["TransactionID"]):
        st.session_state.case_tid = int(v)
        st.session_state.page = "Case"
        st.session_state.jump_msg = ""
    elif v:
        st.session_state.jump_msg = "No such transaction ID in this data."


st.sidebar.text_input("Jump to transaction ID", key="jump_id", on_change=jump)
if st.session_state.get("jump_msg"):
    st.sidebar.caption(st.session_state.jump_msg)
st.sidebar.caption("Data: IEEE-CIS (Vesta), 2017 e-commerce card payments, held-out test period. "
                   "Research prototype, not a production system.")

decisions = L.load_decisions(DEC_PATH)
rq = L.ranked_queue(view, col, decisions)
open_q = rq[rq["status"] == "open"]
open_ids = open_q["TransactionID"].head(300).tolist()
lat = L.latest_decisions(decisions)
if "case_tid" not in st.session_state or st.session_state.case_tid not in set(view["TransactionID"]):
    st.session_state.case_tid = open_ids[0] if open_ids else int(view["TransactionID"].iloc[0])
st.session_state["_open_ids"] = open_ids
st.session_state["_analyst"] = analyst


def go_case(tid):
    st.session_state.case_tid = int(tid)
    st.session_state.page = "Case"


def go_page(p):
    st.session_state.page = p


def start_first():
    if st.session_state["_open_ids"]:
        go_case(st.session_state["_open_ids"][0])


def step(delta):
    ids = st.session_state.get("_open_ids", [])
    cur = st.session_state.case_tid
    if cur in ids:
        st.session_state.case_tid = ids[(ids.index(cur) + delta) % len(ids)]
    elif ids:
        st.session_state.case_tid = ids[0]


def decide(status, tid):
    """Save the decision, then open the next open case."""
    L.save_decision(DEC_PATH, tid, status, st.session_state.get("case_note", ""), st.session_state.get("_analyst", "analyst"))
    ids = [i for i in st.session_state.get("_open_ids", []) if i != tid]
    st.session_state.case_tid = ids[0] if ids else tid
    st.session_state.case_note = ""
    word = {"confirmed fraud": "confirmed as fraud", "dismissed": "dismissed", "escalated": "escalated"}[status]
    st.session_state.flash = f"Transaction {tid} {word}. Showing the next open case."


def bulk_escalate(ids):
    for t in ids:
        L.save_decision(DEC_PATH, t, "escalated", "Matched the alert rule", st.session_state.get("_analyst", "analyst"))
    st.session_state.flash = f"{len(ids)} cases escalated."


st.sidebar.button("Start with the top open case", on_click=start_first, use_container_width=True, type="primary")

right_chip = f"Showing day <b>{replay_n}</b> of {n_days}<br>Ranking: <b>{ui.esc(L.pretty(model))}</b>"


def pr_pill(p):
    return ui.pill(p, p.lower())


def st_pill(s_):
    return ui.pill(s_, s_)


def queue_table(df, n=25):
    rows = [[f"{int(r['rank']):,}", str(int(r["TransactionID"])), f"{r['TransactionAmt']:,.2f}", f"{r[col]:.3f}",
             pr_pill(r["priority"]), st_pill(r["status"])] for _, r in df.head(n).iterrows()]
    return ui.table_html(["Rank", "Transaction", "Amount", "Score", "Priority", "Status"], rows, num_cols=("Rank", "Amount", "Score"))


# ---------------- Home ----------------
if page == "Home":
    md(ui.masthead("Fraud Investigation Workbench",
                   "Rank card transactions by risk, see who is connected, and record what you decide.", right_chip))
    k = st.columns(4)
    p1 = res["results"][model]["te"]["Precision@1%"]
    k[0].markdown(ui.kpi("Transactions in view", f"{len(view):,}", f"day 1 to day {replay_n}", "accent"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("High-priority open cases", f"{int((open_q.priority == 'High').sum()):,}", "top 0.5% by score", "warn"),
                  unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Decided by you", f"{len(lat):,}", "saved in decisions.csv"), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Fraud share of top 1%", f"{p1:.0%}", "full test period, this model"), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        dc = L.daily_counts(view, col)
        fig = go.Figure(go.Bar(x=dc["day"], y=dc["high"], marker_color=GREEN, hovertemplate="Day %{x}: %{y} high-priority<extra></extra>"))
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=6, b=0), plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="Day of test period", yaxis_title="High-priority cases")
        md('<div class="panel"><h4>High-priority cases per day</h4><div class="sub">The top 0.5% of the queue, by day.</div></div>')
        st.plotly_chart(fig, use_container_width=True)
    with right:
        bullets = "".join(f"<li>{ui.esc(m)}</li>" for m in L.honest_sentences(res))
        md(ui.panel("What the experiment found", f'<ul style="margin:0;padding-left:18px;font-size:0.92rem">{bullets}</ul>'))
        st.button("See the evidence", on_click=go_page, args=("Evidence",), use_container_width=True)
    md(ui.panel("Next up", queue_table(open_q, 6), "Open cases, highest risk first"))
    b = st.columns(3)
    b[0].button("Start investigating", on_click=start_first, type="primary", use_container_width=True)
    b[1].button("Open the full queue", on_click=go_page, args=("Queue",), use_container_width=True)
    b[2].button("Take the guided demo", on_click=go_page, args=("Guided demo",), use_container_width=True)

# ---------------- Queue ----------------
elif page == "Queue":
    md(ui.masthead("Queue", "Every transaction in view, ranked by risk. Filter, search, then open a case.", right_chip))
    f = st.columns([1, 1, 1, 2])
    pr = f[0].selectbox("Priority", ["all", "High", "Medium", "Low"])
    stt = f[1].selectbox("Status", ["all"] + L.STATUSES, index=1)
    min_amt = f[2].number_input("Minimum amount", 0.0, 100000.0, 0.0, 50.0)
    q_search = f[3].text_input("Search by transaction ID or card text")
    q = L.ranked_queue(view, col, decisions, min_amt=min_amt, status=stt, priority=pr, search=q_search)
    if q.empty:
        md(ui.empty("No cases match these filters", "Clear a filter or lower the minimum amount."))
    else:
        md(ui.panel(f"{len(q):,} cases", queue_table(q, 25),
                    "Showing the first 25. High = top 0.5% by score, Medium = next 1.5%, Low = the rest."))
        a = st.columns([2, 1, 1])
        pick = a[0].selectbox("Open a case", q["TransactionID"].head(100).tolist(), label_visibility="collapsed")
        a[1].button("Open case", on_click=go_case, args=(pick,), type="primary", use_container_width=True)
        a[2].download_button("Download CSV", q.head(5000)[["rank", "TransactionID", "priority", "status", "TransactionAmt", col]].to_csv(index=False),
                             "queue.csv", "text/csv", use_container_width=True)
        with st.expander("Show more rows (sortable)"):
            v = q.head(500)[["rank", "TransactionID", "priority", "status", "TransactionAmt", col]]
            if eval_mode:
                v = v.assign(true_label=q.head(500)["isFraud"].map({1: "fraud", 0: "not fraud"}))
            st.dataframe(v, use_container_width=True, hide_index=True)

# ---------------- Case ----------------
elif page == "Case":
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    tid = st.session_state.case_tid
    row = view[view.TransactionID == tid].iloc[0]
    me = rq[rq.TransactionID == tid].iloc[0]
    md(ui.masthead(f"Transaction {tid}", "Why it was flagged, who is connected, and what you decide.",
                   f"{pr_pill(me['priority'])} {st_pill(me['status'])}"))
    nv = st.columns([1, 1, 5])
    nv[0].button("Previous", on_click=step, args=(-1,), use_container_width=True)
    nv[1].button("Next", on_click=step, args=(1,), use_container_width=True)
    nv[2].caption(f"{len(open_ids)} open cases in the list, highest risk first")
    nlinks = L.linked_cards(view, row)
    n_prev = int(view[(view.card_id == row.card_id) & (view.TransactionDT <= row.TransactionDT)].shape[0]) - 1
    md(ui.panel("Where this case sits in the queue", ui.triage_rail(int(me["rank"]), len(view))))
    md(ui.facts([("Amount", f"{row.TransactionAmt:,.2f}"), ("Risk score", f"{row[col]:.3f}"),
                 ("Other linked cards", f"{nlinks}"), ("Earlier transactions on this card", f"{max(0, n_prev)}")]))
    st.write("")
    left, right = st.columns([2, 3])
    with left:
        rs = L.reasons(view, row)
        md(ui.panel("Why it was flagged", ui.reasons_html(rs)))
        md('<div class="panel"><h4>Your decision</h4><div class="sub">The next open case opens after you choose.</div></div>')
        st.text_area("Note (optional)", key="case_note", height=80)
        x = st.columns(3)
        x[0].button("Confirm fraud", on_click=decide, args=("confirmed fraud", tid), use_container_width=True, type="primary")
        x[1].button("Dismiss", on_click=decide, args=("dismissed", tid), use_container_width=True)
        x[2].button("Escalate", on_click=decide, args=("escalated", tid), use_container_width=True)
        dec_row = lat.loc[tid].to_dict() if tid in lat.index else None
        facts_pairs = [("Transaction", tid), ("Amount", f"{row.TransactionAmt:,.2f}"), ("Risk score", f"{row[col]:.3f}"),
                       ("Queue rank", f"{int(me['rank']):,}"), ("Priority", me["priority"]), ("Other linked cards", nlinks)]
        st.download_button("Download case report (open it, then Print to save as PDF)",
                           ui.report_html(f"Case report: transaction {tid}", facts_pairs, rs, dec_row),
                           f"case_{tid}.html", "text/html", use_container_width=True)
        if eval_mode:
            md(ui.callout("True label (evaluation only): <b>" + ("FRAUD" if row.isFraud == 1 else "not fraud") + "</b>",
                          "high" if row.isFraud == 1 else "blue"))
    with right:
        hub = st.slider("Ignore devices and addresses shared by more than this many cards", 5, 100, 30, 5)
        nodes, edges = L.case_graph(view, row, hub_cap=hub)
        md('<div class="panel"><h4>Who is connected</h4><div class="sub">Only activity before this transaction is used.</div></div>')
        draw_graph(nodes, edges, show_labels=eval_mode)
    tl = L.timeline(view, row["card_id"])
    md('<div class="panel"><h4>This card, over time</h4></div>')
    st.dataframe(tl if eval_mode else tl.drop(columns="isFraud"), use_container_width=True, hide_index=True)

# ---------------- Alerts ----------------
elif page == "Alerts":
    if st.session_state.get("flash"):
        st.success(st.session_state.pop("flash"))
    md(ui.masthead("Alerts", "Set a rule. Every open case that matches it appears here.", right_chip))
    r_ = st.columns(2)
    top_pct = r_[0].slider("Alert when the case is in the top this % of scores", 0.1, 2.0, 0.5, 0.1)
    amt_min = r_[1].number_input("and the amount is at least", 0.0, 100000.0, 0.0, 50.0)
    al = L.alert_rows(view, col, decisions, top_pct, amt_min)
    k = st.columns(3)
    k[0].markdown(ui.kpi("Open alerts", f"{len(al):,}", f"top {top_pct}% and amount at least {amt_min:,.0f}", "warn"), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Money in these alerts", f"{al.TransactionAmt.sum():,.0f}", "sum of amounts"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Biggest single alert", f"{al.TransactionAmt.max():,.0f}" if len(al) else "0", "amount"), unsafe_allow_html=True)
    st.write("")
    if al.empty:
        md(ui.empty("No open alerts", "Loosen the rule or choose a later replay day."))
    else:
        md(ui.panel("Alert feed", queue_table(al, 20), "Highest score first"))
        a = st.columns([2, 1, 1])
        pick = a[0].selectbox("Open an alert", al["TransactionID"].head(100).tolist(), label_visibility="collapsed")
        a[1].button("Open case", on_click=go_case, args=(pick,), type="primary", use_container_width=True)
        a[2].button(f"Escalate first {min(50, len(al))}", on_click=bulk_escalate, args=(al["TransactionID"].head(50).tolist(),),
                    use_container_width=True)

# ---------------- Budget ----------------
elif page == "Budget":
    md(ui.masthead("Budget", "Your team cannot check everything. See what a limited review catches.", right_chip))
    mode = st.radio("Plan by", ["Share of transactions", "Team capacity"], horizontal=True)
    if mode == "Share of transactions":
        pct = st.slider("Share of transactions reviewed (%)", 0.5, 20.0, 1.0, 0.5)
        r = L.budget_row(view, col, pct)
    else:
        w = st.columns(3)
        an = w[0].number_input("Analysts", 1, 100, 3)
        cph = w[1].number_input("Cases per analyst per hour", 1, 60, 12)
        hrs = w[2].number_input("Hours", 1, 24, 8)
        r = L.workload(view, col, an, cph, hrs)
        pct = r["pct"]
        md(ui.callout(f"Your team can review about <b>{r['capacity']:,}</b> cases, which is {pct:.2f}% of the transactions in view."))
    k = st.columns(4)
    k[0].markdown(ui.kpi("Transactions reviewed", f"{r['reviewed']:,}", ""), unsafe_allow_html=True)
    k[1].markdown(ui.kpi("Frauds caught", f"{r['frauds_caught']:,}", f"{r['precision']:.1%} of what you review", "accent"), unsafe_allow_html=True)
    k[2].markdown(ui.kpi("Share of all fraud found", f"{r['recall']:.1%}", f"random review would find {r['random_recall']:.1%}"), unsafe_allow_html=True)
    k[3].markdown(ui.kpi("Fraud money caught", f"{r['value_share']:.1%}", "share of fraud amount"), unsafe_allow_html=True)
    st.write("")
    fig = go.Figure()
    palette = [BROWN, GREEN, BLUE, RED, GREY, ui.AMBER]
    for i, mname in enumerate(models):
        cur = L.recall_curve(view, L.score_col(mname))
        fig.add_trace(go.Scatter(x=cur.pct_reviewed, y=cur.recall, mode="lines", name=L.pretty(mname),
                                 line=dict(color=palette[i % len(palette)], width=3 if mname == model else 1.5)))
    fig.add_trace(go.Scatter(x=[0, 20], y=[0, 0.2], mode="lines", name="random review", line=dict(color="#bbb", dash="dash")))
    fig.add_vline(x=min(pct, 20), line_color="#444", line_dash="dot")
    fig.update_layout(height=400, xaxis_title="% of transactions reviewed", yaxis_title="share of all fraud found",
                      yaxis_tickformat=".0%", plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=0, r=0, t=6, b=0))
    st.plotly_chart(fig, use_container_width=True)
    if len(models) > 1:
        other = st.selectbox("Compare the ranking model with", [m for m in models if m != model], format_func=L.pretty)
        o = L.overlap(view, col, L.score_col(other), pct)
        md(ui.callout(f"At this budget both models agree on <b>{o['both']:,}</b> of {o['n']:,} cases. "
                      f"Frauds found only by {ui.esc(L.pretty(model).split('  ')[0])}: <b>{o['fraud_only_a']}</b>. "
                      f"Only by {ui.esc(L.pretty(other).split('  ')[0])}: <b>{o['fraud_only_b']}</b>. Both: <b>{o['fraud_both']}</b>."))
    rows = []
    for mname in models:
        x = L.budget_row(view, L.score_col(mname), pct)
        rows.append([ui.esc(L.pretty(mname)), f"{x['precision']:.1%}", f"{x['recall']:.1%}", f"{x['value_share']:.1%}"])
    md(ui.panel("All models at this budget", ui.table_html(["Model", "Precision", "Recall", "Fraud money caught"], rows,
                                                           num_cols=("Precision", "Recall", "Fraud money caught"))))

# ---------------- Signals ----------------
elif page == "Signals":
    md(ui.masthead("Signals", "Does the way cards are linked carry any information about fraud?", right_chip))
    if D["lifts"] is None:
        md(ui.empty("Nothing to show yet", "Run `python explore_graph_signal.py --fast`, then refresh."))
    else:
        lt = D["lifts"]
        feat = st.selectbox("Relationship feature", sorted(lt.feature.unique()))
        sub = lt[lt.feature == feat]
        fig = go.Figure(go.Bar(x=sub["bin"], y=sub["lift"], marker_color=[GREEN if v >= 1 else BROWN for v in sub["lift"]],
                               text=[f"{v:.1f}x<br>n={n:,}" for v, n in zip(sub["lift"], sub["n_tx"])]))
        fig.add_hline(y=1, line_dash="dash", line_color="#888")
        fig.update_layout(height=340, yaxis_title="fraud rate vs average (1.0 = average)", plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0, r=0, t=6, b=0))
        st.plotly_chart(fig, use_container_width=True)
        md(ui.callout("Bars above 1.0 mean a higher fraud rate than average in that group. Check <b>n</b> before trusting a bar."))
    md('<div class="panel"><h4>Groups of cards linked by shared devices or addresses</h4>'
       '<div class="sub">Built from the 28 days before the test period, without looking at fraud labels.</div></div>')
    if D["clusters"] is None or D["members"] is None:
        md(ui.empty("No clusters yet", "Run `python explore_graph_signal.py --fast`, then refresh."))
    else:
        rank_by = st.selectbox("Rank groups by", ["n_cards", "tx_per_card", "cards_per_entity"],
                               format_func=lambda x: {"n_cards": "number of cards", "tx_per_card": "transactions per card",
                                                       "cards_per_entity": "cards per shared device or address"}[x])
        ct = D["clusters"].sort_values(rank_by, ascending=False).head(25)
        st.dataframe(ct, use_container_width=True, hide_index=True)
        cid = st.selectbox("Open a group", ct["cluster"].tolist())
        nodes, edges = L.cluster_graph(D["members"], cid)
        draw_graph(nodes, edges)
        md(ui.callout("Which ranking works best is decided on the validation window (see the output of the explore script), not here."))

# ---------------- Decisions ----------------
elif page == "Decisions":
    md(ui.masthead("Decisions", "Everything you confirmed, dismissed or escalated.", right_chip))
    if lat.empty:
        md(ui.empty("No decisions yet", "Open a case and choose Confirm fraud, Dismiss or Escalate."))
    else:
        counts = lat["status"].value_counts()
        k = st.columns(3)
        k[0].markdown(ui.kpi("Confirmed fraud", f"{int(counts.get('confirmed fraud', 0))}", "", "warn"), unsafe_allow_html=True)
        k[1].markdown(ui.kpi("Dismissed", f"{int(counts.get('dismissed', 0))}", ""), unsafe_allow_html=True)
        k[2].markdown(ui.kpi("Escalated", f"{int(counts.get('escalated', 0))}", ""), unsafe_allow_html=True)
        st.write("")
        log = lat.reset_index().sort_values("time", ascending=False)
        rows = [[ui.esc(r["time"]), str(int(r["TransactionID"])), st_pill(r["status"]), ui.esc(r["note"] if isinstance(r["note"], str) else ""),
                 ui.esc(r["analyst"])] for _, r in log.head(50).iterrows()]
        md(ui.panel("Decision log", ui.table_html(["Time", "Transaction", "Decision", "Note", "Analyst"], rows)))
        st.download_button("Download decision log (CSV)", log.to_csv(index=False), "my_decisions.csv", "text/csv")
        if eval_mode:
            sc = L.scorecard(decisions, scored)
            st.write("")
            k = st.columns(3)
            k[0].markdown(ui.kpi("Confirmed and really fraud", f"{sc['confirmed_right']} of {sc['confirmed']}", "uses true labels"), unsafe_allow_html=True)
            k[1].markdown(ui.kpi("Dismissed but was fraud", f"{sc['dismissed_but_fraud']}", "missed"), unsafe_allow_html=True)
            k[2].markdown(ui.kpi("Fraud money you confirmed", f"{sc['money_confirmed']:,.0f}", ""), unsafe_allow_html=True)
    with st.expander("Delete my decisions"):
        sure = st.checkbox("I want to delete all my decisions")
        if sure and st.button("Delete decisions"):
            if os.path.exists(DEC_PATH):
                os.remove(DEC_PATH)
            st.rerun()

# ---------------- Guided demo ----------------
elif page == "Guided demo":
    md(ui.masthead("Guided demo", "Three cases worth opening, picked automatically from the highest-priority queue.", right_chip))
    picks = get_demo(view, col)
    if not picks:
        md(ui.empty("Nothing to show", "There are no high-priority cases in this replay window. Move the replay slider later."))
    else:
        cols_ = st.columns(len(picks))
        for c, (title, r, text) in zip(cols_, picks):
            with c:
                md(f'<div class="demo"><h4>{ui.esc(title)}</h4><p>{ui.esc(text)}</p>'
                   f'<p>Transaction <b>{int(r["TransactionID"])}</b>, amount <b>{r["TransactionAmt"]:,.2f}</b>, rank <b>{int(r["rank"]):,}</b>.</p></div>')
                st.button("Open this case", on_click=go_case, args=(int(r["TransactionID"]),), key=f"demo_{int(r['TransactionID'])}",
                          use_container_width=True, type="primary")
    md(ui.panel("A 90-second walkthrough", '<ol style="margin:0;padding-left:20px;font-size:0.93rem;line-height:1.7">'
                "<li>Home: the queue is ranked, and the experiment's findings are stated plainly.</li>"
                "<li>Open the connected-group case and read the reasons, then the network.</li>"
                "<li>Click Confirm fraud and watch the next case open.</li>"
                "<li>Budget: pick 3 analysts, 12 cases an hour, 8 hours, and read what that catches.</li>"
                "<li>Evidence: show the validation and test tables and the limitations.</li></ol>"))

# ---------------- Evidence ----------------
else:
    md(ui.masthead("Evidence", "What we tested, how we kept it fair, and what it showed.", right_chip))
    bullets = "".join(f"<li>{ui.esc(m)}</li>" for m in L.honest_sentences(res))
    md(ui.panel("Findings", f'<ul style="margin:0;padding-left:18px">{bullets}</ul>'))
    c1, c2 = st.columns(2)
    c1.markdown("**Validation** (used to decide)")
    c1.dataframe(L.results_table(res, "va").style.format("{:.4f}"), use_container_width=True)
    c2.markdown("**Test** (report only)")
    c2.dataframe(L.results_table(res, "te").style.format("{:.4f}"), use_container_width=True)
    md(ui.panel("How to read the models", "<ul style='margin:0;padding-left:18px;font-size:0.92rem;line-height:1.7'>"
                "<li><b>A</b> base model on the dataset's own columns.</li>"
                "<li><b>B</b> adds our relationship features: shared devices and addresses, bursts, group size.</li>"
                "<li><b>C</b> removes the dataset's engineered count columns (C1 to C14). <b>D</b> adds our features to C. "
                "Comparing C with D checks whether our features add anything beyond those counts.</li>"
                "<li><b>E</b> and <b>F</b> (if shown) add each client's own history.</li></ul>"))
    md(ui.panel("How we kept it fair", "<ul style='margin:0;padding-left:18px;font-size:0.92rem;line-height:1.7'>"
                "<li>Split by time: train 70%, validation 10%, test 20%. Choices are made on validation; test is only reported.</li>"
                "<li>Relationship features use only earlier transactions. A test in the repository checks that flipping labels or removing the future changes nothing.</li>"
                "<li>Precision@1% is the share of frauds among the 1% highest-scored transactions.</li></ul>"))
    md(ui.panel("Limitations", "<ul style='margin:0;padding-left:18px;font-size:0.92rem;line-height:1.7'>"
                "<li>The dataset has no account or UPI IDs, so a card proxy (card1 to card6 plus address) stands in for a client.</li>"
                "<li>Device information exists for only about 25% of transactions, so links are sparse.</li>"
                "<li>It is 2017 e-commerce card data, not Indian UPI traffic. We show a method, not Indian fraud rates.</li>"
                "<li>A score is a reason to look, not proof of fraud.</li></ul>"))
