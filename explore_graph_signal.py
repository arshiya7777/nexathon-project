"""Does relationship structure carry fraud signal? (descriptive; labels are used ONLY to evaluate)

Part 1  Lift tables: fraud rate for low/high values of each graph feature (whole dataset).
Part 2  Cluster ranking: build clusters of cards linked by shared devices/addresses using ONLY the
        28 days BEFORE the test period, rank them without labels, then check the fraud rate of
        those cards' transactions in the test period (the future).

Run:  python explore_graph_signal.py --fast       (150k rows)
      python explore_graph_signal.py              (all rows)
Outputs: outputs/lift_tables.csv, outputs/clusters_top.csv, outputs/clusters.pkl (for the dashboard)
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from features import (DAY, LINK_KEYS, add_ids, burst_features, component_features,
                      windowed_link_features)
from run_experiment import load


def lift_tables(df, G):
    base = df["isFraud"].mean()
    out = []
    for c in G.columns:
        s = G[c]
        miss = s.isna()
        if miss.any():
            out.append((c, "missing", int(miss.sum()), df.loc[miss, "isFraud"].mean()))
        nz = s[~miss]
        edges = np.unique(nz.quantile([0, .5, .8, .95, .99, 1]).to_numpy())
        if len(edges) < 2:
            continue
        bins = pd.cut(nz, bins=edges, include_lowest=True, duplicates="drop")
        for b, grp in df.loc[nz.index].groupby(bins, observed=True):
            out.append((c, str(b), len(grp), grp["isFraud"].mean()))
    t = pd.DataFrame(out, columns=["feature", "bin", "n_tx", "fraud_rate"])
    t["lift"] = t["fraud_rate"] / base
    return t, base


def build_clusters(df, cut, window_days=28, hub_cap=30, min_cards=3, keys=LINK_KEYS):
    """Clusters of cards connected through shared devices / email+address, built from the
    window just BEFORE `cut`. Entities used by more than `hub_cap` cards are ignored (hubs)."""
    t = df["TransactionDT"].to_numpy()
    sub = df.iloc[np.where((t < cut) & (t >= cut - window_days * DAY))[0]]
    parts = []
    for k in keys:
        v = sub[["card_id", k]].dropna().drop_duplicates()
        deg = v.groupby(k)["card_id"].transform("nunique")
        v = v[(deg >= 2) & (deg <= hub_cap)]  # >=2: it really links cards; <=cap: not a hub
        parts.append(pd.DataFrame({"card_id": v["card_id"].to_numpy(), "entity": k + ":" + v[k].astype(str).to_numpy()}))
    E = pd.concat(parts, ignore_index=True)
    ci, cards = pd.factorize(E["card_id"])
    ei, ents = pd.factorize(E["entity"])
    n = len(cards) + len(ents)
    adj = coo_matrix((np.ones(len(E), dtype=np.int8), (ci, len(cards) + ei)), shape=(n, n))
    ncomp, comp = connected_components(adj, directed=False)
    card_comp, ent_comp = comp[:len(cards)], comp[len(cards):]
    n_cards = np.bincount(card_comp, minlength=ncomp)
    n_ents = np.bincount(ent_comp, minlength=ncomp)
    tx_per_card = sub.groupby("card_id").size()
    tx_of_card = tx_per_card.reindex(cards).fillna(0).to_numpy()
    n_tx = np.bincount(card_comp, weights=tx_of_card, minlength=ncomp)
    keep = np.where(n_cards >= min_cards)[0]
    table = pd.DataFrame({"cluster": keep, "n_cards": n_cards[keep], "n_entities": n_ents[keep], "n_tx_window": n_tx[keep]})
    table["tx_per_card"] = table["n_tx_window"] / table["n_cards"]
    table["cards_per_entity"] = table["n_cards"] / table["n_entities"].clip(lower=1)
    card_df = pd.DataFrame({"comp": card_comp, "card": cards})
    ent_df = pd.DataFrame({"comp": ent_comp, "entity": ents})
    keep_set = set(keep.tolist())
    cmem = {c: g["card"].tolist() for c, g in card_df[card_df.comp.isin(keep_set)].groupby("comp")}
    emem = {c: g["entity"].tolist() for c, g in ent_df[ent_df.comp.isin(keep_set)].groupby("comp")}
    E2 = E.assign(comp=card_comp[ci])
    emap = {c: list(zip(g["card_id"], g["entity"])) for c, g in E2[E2.comp.isin(keep_set)].groupby("comp")}
    members = {int(c): {"cards": cmem.get(c, []), "entities": emem.get(c, []), "edges": emap.get(c, [])} for c in keep}
    return table, members


RANKINGS = {"size (n_cards)": ["n_cards", "n_tx_window"],
            "activity (tx per card)": ["tx_per_card", "n_cards"],
            "density (cards per shared entity)": ["cards_per_entity", "n_cards"]}


def evaluate_clusters(df, table, members, cut, end, ks=(10, 25, 50, 100)):
    t = df["TransactionDT"].to_numpy()
    fut = df.iloc[np.where((t >= cut) & (t < end))[0]]
    base = fut["isFraud"].mean()
    total_fraud = fut["isFraud"].sum()
    print(f"  future window: {len(fut):,} transactions, fraud rate {base:.2%}, {int(total_fraud)} frauds")
    print(f"  {'ranking':<36}{'top K':>6}{'cards':>7}{'fut tx':>8}{'fraud rate':>11}{'lift':>6}{'frauds caught':>16}")
    for name, cols in RANKINGS.items():
        ranked = table.sort_values(cols, ascending=False)
        for k in ks:
            if k > len(ranked):
                continue
            cards = set()
            for c in ranked["cluster"].head(k):
                cards.update(members[int(c)]["cards"])
            f = fut[fut["card_id"].isin(cards)]
            if len(f) == 0:
                continue
            rate = f["isFraud"].mean()
            print(f"  {name:<36}{k:>6}{len(cards):>7}{len(f):>8}{rate:>11.2%}{rate / base:>6.1f}"
                  f"{int(f['isFraud'].sum()):>9} ({f['isFraud'].sum() / total_fraud:.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--hub_cap", type=int, default=30)
    ap.add_argument("--data_dir", default="data")
    a = ap.parse_args()
    rows = a.rows or (150_000 if a.fast else None)
    df = add_ids(load(rows, a.data_dir))
    print(f"Loaded {len(df):,} transactions, fraud rate {df.isFraud.mean():.2%}")

    G = pd.concat([windowed_link_features(df), burst_features(df), component_features(df, hub_cap=a.hub_cap)], axis=1)
    t, base = lift_tables(df, G)
    os.makedirs("outputs", exist_ok=True)
    t.to_csv("outputs/lift_tables.csv", index=False)
    print(f"\n=== PART 1: fraud rate by graph-feature value (base rate {base:.2%}) ===")
    print("(only rows where lift >= 1.5 or <= 0.5 and n_tx >= 200 are shown; everything is in outputs/lift_tables.csv)")
    show = t[(t.n_tx >= 200) & ((t.lift >= 1.5) | (t.lift <= 0.5))]
    print(show.to_string(index=False, formatters={"fraud_rate": "{:.2%}".format, "lift": "{:.1f}x".format}) if len(show) else "  none")

    n = len(df)
    tmax = df["TransactionDT"].max() + 1
    for label, a_, b_ in (("VALIDATION (use this to choose a ranking)", 0.7, 0.8), ("TEST (report only)", 0.8, 1.0)):
        cut = df["TransactionDT"].iloc[int(n * a_)]
        end = df["TransactionDT"].iloc[int(n * b_)] if b_ < 1 else tmax
        table, members = build_clusters(df, cut, hub_cap=a.hub_cap)
        print(f"\n=== PART 2 {label}: {len(table)} clusters (>=3 cards) built from the 28 days before the window ===")
        if len(table) == 0:
            print("  no clusters found: try --hub_cap 100")
            continue
        if a_ == 0.8:
            print("Largest clusters:")
            print(table.sort_values("n_cards", ascending=False).head(8).to_string(
                index=False, formatters={"tx_per_card": "{:.1f}".format, "n_tx_window": "{:.0f}".format, "cards_per_entity": "{:.1f}".format}))
        evaluate_clusters(df, table, members, cut, end)
        if a_ == 0.8:
            table.to_csv("outputs/clusters_top.csv", index=False)
            pickle.dump(members, open("outputs/clusters.pkl", "wb"))
    print("\nSaved outputs/lift_tables.csv, outputs/clusters_top.csv, outputs/clusters.pkl")

if __name__ == "__main__":
    main()
