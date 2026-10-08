"""All dashboard calculations live here (plain pandas, no Streamlit) so they can be tested.
Reads the files that run_experiment.py and explore_graph_signal.py save in outputs/.
Case/network views use ONLY transactions up to the one being investigated (no future)."""
import json
import os
import pickle

import numpy as np
import pandas as pd


# ---------- loading ----------
def load_outputs(folder="outputs"):
    """Returns a dict; anything missing is None (the app then shows a friendly hint)."""
    def p(name):
        return os.path.join(folder, name)

    out = {"results": None, "scored": None, "lifts": None, "clusters": None, "members": None}
    if os.path.exists(p("results.json")):
        out["results"] = json.load(open(p("results.json")))
    if os.path.exists(p("test_scored.pkl")):
        out["scored"] = pd.read_pickle(p("test_scored.pkl")).sort_values("TransactionDT").reset_index(drop=True)
    if os.path.exists(p("lift_tables.csv")):
        out["lifts"] = pd.read_csv(p("lift_tables.csv"))
    if os.path.exists(p("clusters_top.csv")):
        out["clusters"] = pd.read_csv(p("clusters_top.csv"))
    if os.path.exists(p("clusters.pkl")):
        out["members"] = pickle.load(open(p("clusters.pkl"), "rb"))
    return out


def score_col(variant_name):
    """'B A+G' -> 'score_B'"""
    return "score_" + variant_name.split()[0]


def available_models(results, scored):
    """Variant names that have both metrics and a score column."""
    if not results or scored is None:
        return []
    return [n for n in results["results"] if score_col(n) in scored.columns]


def pretty(name):
    """Human label for a variant name."""
    nice = {"A": "A  Base model", "B": "B  Base + graph", "E": "E  Base + graph + client history",
            "C": "C  No Vesta counts", "D": "D  No Vesta counts + graph", "F": "F  No Vesta counts + graph + history"}
    return nice.get(name.split()[0], name)


def results_table(results, part="te"):
    rows = {pretty(n): v[part] for n, v in results["results"].items()}
    return pd.DataFrame(rows).T


def honest_sentences(results):
    """Plain-language claims that follow the numbers (never overclaim)."""
    r = results["results"]
    msgs = []
    by_letter = {k.split()[0]: k for k in r}  # match by the leading letter, so exact names don't matter
    pairs = [("A", "B", "Adding graph features to the base model"),
             ("C", "D", "Adding graph features to the model without Vesta counts")]
    for a_, b_, label in pairs:
        ref, new = by_letter.get(a_), by_letter.get(b_)
        if ref is None or new is None:
            continue
        if ref in r and new in r:
            d = r[new]["te"]["PR-AUC"] - r[ref]["te"]["PR-AUC"]
            dv = r[new]["va"]["PR-AUC"] - r[ref]["va"]["PR-AUC"]
            if d > 0.005 and dv > 0:
                word = "improved"
            elif d < -0.005 and dv < 0:
                word = "reduced"
            else:
                word = "made no clear difference to"
            msgs.append(f"{label} {word} PR-AUC (test: {r[ref]['te']['PR-AUC']:.3f} -> {r[new]['te']['PR-AUC']:.3f}; "
                        f"validation: {r[ref]['va']['PR-AUC']:.3f} -> {r[new]['va']['PR-AUC']:.3f}).")
    msgs.append(f"Based on {results.get('rows', '?'):,} transactions, {results.get('seeds', '?')} seed(s), "
                f"model: {results.get('model', '?')}. Test numbers are report-only.")
    return msgs


# ---------- review budget ----------
def budget_row(scored, col, pct):
    """If the team can review the top `pct`% of transactions by score, what happens?"""
    n = max(1, int(round(len(scored) * pct / 100)))
    top = scored.sort_values(col, ascending=False).head(n)
    tf = scored["isFraud"].sum()
    tv = scored.loc[scored.isFraud == 1, "TransactionAmt"].sum()
    caught = int(top["isFraud"].sum())
    return {"reviewed": n, "frauds_caught": caught, "precision": caught / n,
            "recall": caught / tf if tf else 0.0,
            "value_caught": float(top.loc[top.isFraud == 1, "TransactionAmt"].sum()),
            "value_share": float(top.loc[top.isFraud == 1, "TransactionAmt"].sum() / tv) if tv else 0.0,
            "random_recall": pct / 100}


def recall_curve(scored, col, pcts=None):
    pcts = pcts if pcts is not None else np.round(np.arange(0.5, 20.5, 0.5), 1)
    s = scored.sort_values(col, ascending=False)["isFraud"].to_numpy()
    cum = np.cumsum(s)
    tf = max(1, s.sum())
    pts = []
    for p in pcts:
        k = max(1, int(round(len(s) * p / 100)))
        pts.append((p, cum[k - 1] / tf))
    return pd.DataFrame(pts, columns=["pct_reviewed", "recall"])


# ---------- queue + reasons ----------
def queue(scored, col, top_n=50):
    q = scored.sort_values(col, ascending=False).head(top_n).copy()
    q.insert(0, "rank", np.arange(1, len(q) + 1))
    q["risk_pct"] = (scored[col].rank(pct=True).loc[q.index] * 100).round(1)
    return q


def _past(scored, row):
    return scored[(scored["TransactionDT"] <= row["TransactionDT"]) & (scored["TransactionID"] != row["TransactionID"])]


def reasons(scored, row, hub_cap=30):
    """Plain-language reasons using only earlier transactions (and the row itself)."""
    past = _past(scored, row)
    out = []
    amt = row["TransactionAmt"]
    pct = (scored["TransactionAmt"] < amt).mean() * 100
    if pct >= 90:
        out.append(f"Amount {amt:,.2f} is higher than {pct:.0f}% of transactions.")
    same_card = past[past.card_id == row["card_id"]]
    day = 86400
    recent = same_card[same_card.TransactionDT >= row["TransactionDT"] - day]
    if len(recent) >= 2:
        out.append(f"This card made {len(recent)} other transactions in the previous 24 hours.")
    if len(same_card) == 0:
        out.append("First time this card appears in the data window.")
    for key, label in (("device_fp", "device"), ("email_addr", "email + address")):
        v = row.get(key)
        if isinstance(v, str):
            others = past[past[key] == v]["card_id"].unique()
            others = [c for c in others if c != row["card_id"]]
            if 1 <= len(others) <= hub_cap:
                out.append(f"This {label} was also used by {len(others)} other card(s) earlier.")
            elif len(others) > hub_cap:
                out.append(f"This {label} is shared by {len(others)} cards (very common, so ignored as a link).")
    if not out:
        out.append("No simple rule explains the score; it comes from many small model signals combined.")
    return out


# ---------- case network ----------
def case_graph(scored, row, hub_cap=30, max_nodes=60, hops=2):
    """Bipartite graph around the transaction's card: cards <-> devices / email-addresses.
    Past only. Entities used by more than hub_cap cards are dropped. Returns (nodes, edges)."""
    past = _past(scored, row)
    past = pd.concat([past, scored[scored["TransactionID"] == row["TransactionID"]]])
    keys = ["device_fp", "email_addr"]
    start = row["card_id"]
    cards, ents = {start}, set()
    frontier = {start}
    for _ in range(hops):
        new_cards = set()
        for k in keys:
            sub = past[past.card_id.isin(frontier)][["card_id", k]].dropna()
            for e in sub[k].unique():
                ent = (k, e)
                users = past.loc[past[k] == e, "card_id"].unique()
                if len(users) < 2 or len(users) > hub_cap:
                    continue
                ents.add(ent)
                new_cards.update(users.tolist())
        frontier = new_cards - cards
        cards |= new_cards
        if len(cards) + len(ents) >= max_nodes:
            break
    # trim to max_nodes: keep start card, then cards with most links
    edges = []
    for k, e in ents:
        for c in past.loc[past[k] == e, "card_id"].unique():
            if c in cards:
                edges.append((c, f"{k}:{e}"))
    edges = list(dict.fromkeys(edges))
    deg = pd.Series([c for c, _ in edges]).value_counts() if edges else pd.Series(dtype=int)
    ranked = [start] + [c for c in deg.index if c != start]
    keep_cards = set(ranked[: max(2, max_nodes // 2)])
    edges = [(c, e) for c, e in edges if c in keep_cards]
    ent_names = {e for _, e in edges}
    fraud = past.groupby("card_id")["isFraud"].max()
    nodes = [{"id": c, "kind": "card", "start": c == start, "fraud": int(fraud.get(c, 0)),
              "n_tx": int((past.card_id == c).sum())} for c in sorted(keep_cards)]
    nodes += [{"id": e, "kind": e.split(":", 1)[0], "start": False, "fraud": 0, "n_tx": 0} for e in sorted(ent_names)]
    return nodes, edges


def cluster_graph(members, cid, max_edges=250):
    """Nodes and edges of one saved cluster (edges were saved by explore_graph_signal.py)."""
    m = members[int(cid)]
    edges = list(m.get("edges", []))[:max_edges]
    cards = sorted({c for c, _ in edges} or set(m["cards"][:60]))
    ents = sorted({e for _, e in edges})
    nodes = [{"id": c, "kind": "card", "start": False, "fraud": 0, "n_tx": 0} for c in cards]
    nodes += [{"id": e, "kind": e.split(":", 1)[0], "start": False, "fraud": 0, "n_tx": 0} for e in ents]
    return nodes, edges


def timeline(scored, card_id):
    t = scored[scored.card_id == card_id].sort_values("TransactionDT")
    return t[["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]]


# ---------- analyst workflow (decisions are saved to a CSV so they survive restarts) ----------
DECISION_COLS = ["time", "TransactionID", "status", "note", "analyst"]
STATUSES = ["open", "confirmed fraud", "dismissed", "escalated"]


def load_decisions(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return pd.DataFrame(columns=DECISION_COLS)


def latest_decisions(dec):
    """Newest decision per transaction."""
    if dec.empty:
        return dec.set_index("TransactionID") if "TransactionID" in dec else dec
    return dec.sort_values("time").drop_duplicates("TransactionID", keep="last").set_index("TransactionID")


def save_decision(path, tid, status, note="", analyst="analyst"):
    assert status in STATUSES
    row = pd.DataFrame([{"time": pd.Timestamp.now().isoformat(timespec="seconds"), "TransactionID": int(tid),
                         "status": status, "note": note, "analyst": analyst}])
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def priority_of(rank, n):
    """High = top 0.5%, Medium = next 1.5%, Low = rest (simple, explainable tiers)."""
    f = rank / max(1, n)
    return np.where(f <= 0.005, "High", np.where(f <= 0.02, "Medium", "Low"))


def ranked_queue(scored, col, decisions=None, min_amt=0.0, status="all", priority="all", search=""):
    """Whole test set ranked by score, with priority and analyst status; filters applied afterwards.
    Rank is always the global rank, so filtering never changes it."""
    r = scored.sort_values(col, ascending=False).reset_index(drop=True)
    r["rank"] = np.arange(1, len(r) + 1)
    r["priority"] = priority_of(r["rank"].to_numpy(), len(r))
    r["status"] = "open"
    if decisions is not None and len(decisions):
        st_ = latest_decisions(decisions)["status"]
        r["status"] = r["TransactionID"].map(st_).fillna("open")
    r = r[r["TransactionAmt"] >= min_amt]
    if status != "all":
        r = r[r["status"] == status]
    if priority != "all":
        r = r[r["priority"] == priority]
    if search.strip():
        s = search.strip()
        r = r[(r["TransactionID"].astype(str) == s) | r["card_id"].str.contains(s, regex=False)]
    return r


def workload(scored, col, analysts, cases_per_hour, hours):
    """Capacity planning: with this many analysts and hours, how many cases can be reviewed and what do they catch?"""
    cap = int(analysts * cases_per_hour * hours)
    cap = max(1, min(cap, len(scored)))
    pct = cap / len(scored) * 100
    out = budget_row(scored, col, pct)
    out["capacity"] = cap
    out["pct"] = pct
    return out


def scorecard(decisions, scored):
    """Compare analyst decisions with the true labels (evaluation only)."""
    lat = latest_decisions(decisions)
    if lat.empty:
        return None
    j = scored.set_index("TransactionID")[["isFraud", "TransactionAmt"]].join(lat[["status"]], how="inner")
    conf, dis = j[j.status == "confirmed fraud"], j[j.status == "dismissed"]
    return {"decided": len(j), "confirmed": len(conf), "confirmed_right": int(conf.isFraud.sum()),
            "dismissed": len(dis), "dismissed_but_fraud": int(dis.isFraud.sum()),
            "escalated": int((j.status == "escalated").sum()),
            "money_confirmed": float(conf.loc[conf.isFraud == 1, "TransactionAmt"].sum())}


def case_report_md(row, col, rank, reason_list, decision=None):
    lines = [f"# Case report: transaction {int(row['TransactionID'])}", "",
             f"- Amount: {row['TransactionAmt']:,.2f}", f"- Model score: {row[col]:.3f}", f"- Queue rank: {rank}",
             f"- Card (proxy id): {str(row['card_id'])[:40]}", "", "## Why it was flagged"]
    lines += [f"- {r}" for r in reason_list]
    if decision is not None:
        lines += ["", "## Analyst decision", f"- Status: {decision.get('status', 'open')}", f"- Note: {decision.get('note', '')}"]
    lines += ["", "_Prototype on IEEE-CIS data. A score is a reason to look, not proof of fraud._"]
    return "\n".join(lines)


# ---------- replay, alerts, demo picks, comparisons ----------
def day_range(scored):
    d = (scored["TransactionDT"] // 86400).astype(int)
    return int(d.min()), int(d.max())


def replay_view(scored, last_day):
    """Everything that had happened by the END of `last_day` (scores are already point-in-time)."""
    return scored[scored["TransactionDT"] < (last_day + 1) * 86400].reset_index(drop=True)


def linked_cards(scored, row, hub_cap=30):
    nodes, _ = case_graph(scored, row, hub_cap=hub_cap)
    return max(0, sum(1 for n in nodes if n["kind"] == "card") - 1)


def alert_rows(scored, col, decisions, top_pct=0.5, min_amt=0.0):
    """Open cases that match the alert rule: within the top `top_pct`% by score AND amount >= min_amt."""
    r = ranked_queue(scored, col, decisions, min_amt=min_amt, status="open")
    return r[r["rank"] <= max(1, int(len(scored) * top_pct / 100))]


def daily_counts(scored, col):
    """Per day: transactions and High-priority alerts (High = top 0.5% of the whole view)."""
    r = scored.sort_values(col, ascending=False).reset_index(drop=True)
    high = np.arange(len(r)) < max(1, int(len(r) * 0.005))
    day = (r["TransactionDT"] // 86400).astype(int)
    d0 = day.min()
    out = pd.DataFrame({"day": day - d0 + 1, "high": high}).groupby("day").agg(transactions=("high", "size"), high=("high", "sum"))
    return out.reset_index()


def demo_cases(scored, col, look=60):
    """Three cases worth showing: the best-connected one, the biggest amount, and a high score with no links."""
    r = ranked_queue(scored, col)
    top = r[r["priority"] == "High"].head(look)
    if top.empty:
        return []
    links = [linked_cards(scored, row) for _, row in top.iterrows()]
    top = top.assign(links=links)
    picks = []
    ring = top.sort_values(["links", col], ascending=False).iloc[0]
    n_links = int(ring["links"])
    plus = "+" if n_links >= 29 else ""
    picks.append(("A connected group", ring,
                  f"This card shares a device or address with {n_links}{plus} other cards seen earlier. "
                  "Open the network to see who they are."))
    used = {int(ring["TransactionID"])}
    big = top[~top["TransactionID"].isin(used)].sort_values("TransactionAmt", ascending=False)
    if len(big):
        b = big.iloc[0]
        picks.append(("The biggest amount at stake", b,
                      f"Amount {b['TransactionAmt']:,.2f}, the largest among the {len(top)} highest-ranked cases."))
        used.add(int(b["TransactionID"]))
    solo = top[(top["links"] == 0) & ~top["TransactionID"].isin(used)].sort_values(col, ascending=False)
    if len(solo):
        s = solo.iloc[0]
        picks.append(("Risky on its own", s,
                      "No shared device or address links this card to others. The model scores it high from the "
                      "transaction details alone, so the network view has nothing to add here."))
    return picks


def overlap(scored, col_a, col_b, pct):
    """How much do two models agree on the top `pct`% and what does each catch that the other misses?"""
    n = max(1, int(round(len(scored) * pct / 100)))
    a = set(scored.sort_values(col_a, ascending=False).head(n)["TransactionID"])
    b = set(scored.sort_values(col_b, ascending=False).head(n)["TransactionID"])
    fraud = set(scored.loc[scored.isFraud == 1, "TransactionID"])
    return {"n": n, "both": len(a & b), "only_a": len(a - b), "only_b": len(b - a),
            "fraud_both": len(a & b & fraud), "fraud_only_a": len((a - b) & fraud), "fraud_only_b": len((b - a) & fraud)}
