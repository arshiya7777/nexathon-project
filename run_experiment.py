"""Baseline vs graph-enhanced fraud model on IEEE-CIS (strict time-based split).

Quick test (2-3 min):   python run_experiment.py --fast
Full run:               python run_experiment.py
No xgboost installed?:  python run_experiment.py --fast --model hgb

Four model variants are compared on the same split:
  A  base (all columns)         = normal model, includes Vesta's own C1-C14 count columns
  B  base + our graph features
  C  base WITHOUT C1-C14        = model that looks at each payment alone
  D  base WITHOUT C1-C14 + our graph features
Decisions (what to keep / change) must be taken on the VALIDATION numbers.
TEST numbers are for reporting only.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import (add_ids, burst_features, component_features, giant_component_report,
                      windowed_link_features)

DATA = "data"
C_COLS = [f"C{i}" for i in range(1, 15)]
TX_COLS = (["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD",
            "card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2", "dist1",
            "P_emaildomain", "R_emaildomain"]
           + C_COLS + [f"D{i}" for i in range(1, 16)] + [f"M{i}" for i in range(1, 10)])
ID_COLS = ["TransactionID", "DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"]
CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"] + [f"M{i}" for i in range(1, 10)]


def load(rows=None, data_dir=DATA):
    tx = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"), usecols=TX_COLS)
    idn = pd.read_csv(os.path.join(data_dir, "train_identity.csv"), usecols=ID_COLS)
    df = tx.merge(idn, on="TransactionID", how="left")
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    if rows:
        df = df.iloc[:rows].reset_index(drop=True)  # earliest N rows: keeps time order intact
    return df


def base_features(df):
    X = pd.DataFrame(index=df.index)
    num = ["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2", "dist1"] \
        + C_COLS + [f"D{i}" for i in range(1, 16)]
    for c in num:
        X[c] = df[c]
    for c in CAT_COLS:  # integer codes; NaN -> NaN
        codes = pd.factorize(df[c])[0].astype("float64")
        codes[codes < 0] = np.nan
        X[c] = codes
    X["hour"] = (df["TransactionDT"] // 3600) % 24
    X["dow"] = (df["TransactionDT"] // 86400) % 7
    X["log_amt"] = np.log1p(df["TransactionAmt"])
    return X


def metrics(y, score, amt, top=0.01):
    k = max(1, int(len(y) * top))
    idx = np.argsort(-score)[:k]
    return {
        "PR-AUC": average_precision_score(y, score),
        "ROC-AUC": roc_auc_score(y, score),
        f"Precision@{top:.0%}": y[idx].mean(),
        f"Recall@{top:.0%}": y[idx].sum() / y.sum(),
        f"FraudValueCaught@{top:.0%}": (amt[idx] * y[idx]).sum() / (amt * y).sum(),
    }


def fit_predict(Xtr, ytr, Xva, yva, Xte, seed, model):
    """Returns (valid scores, test scores, feature importances)."""
    if model == "xgb":
        import xgboost as xgb
        clf = xgb.XGBClassifier(n_estimators=1500, learning_rate=0.05, max_depth=8, subsample=0.8,
                                colsample_bytree=0.7, tree_method="hist", eval_metric="aucpr",
                                early_stopping_rounds=50, random_state=seed, n_jobs=-1)
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1], dict(zip(Xtr.columns, clf.feature_importances_))
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=8,
                                         early_stopping=True, validation_fraction=0.1, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1], {}


def drift_report(G, i1, i2):
    """Stationarity check: a feature whose test-period mean is far from its train-period mean
    will mislead the model (ratio near 1 is good)."""
    print("\nFeature drift check (mean in test / mean in train; near 1.0 is good):")
    for c in G.columns:
        a, b = G[c].iloc[:i1].mean(), G[c].iloc[i2:].mean()
        r = b / a if a else float("nan")
        flag = "   <-- drifting" if (np.isfinite(r) and (r > 1.8 or r < 0.55)) else ""
        print(f"  {c:28s} {r:6.2f}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="first 150k rows, 1 seed")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--model", choices=["xgb", "hgb"], default="xgb")
    ap.add_argument("--hub_cap", type=int, default=30)
    ap.add_argument("--data_dir", default=DATA)
    a = ap.parse_args()
    rows = a.rows or (150_000 if a.fast else None)
    seeds = 1 if a.fast else a.seeds

    t0 = time.time()
    df = add_ids(load(rows, a.data_dir))
    print(f"Loaded {len(df):,} transactions, fraud rate {df.isFraud.mean():.2%}  ({time.time()-t0:.0f}s)")
    giant_component_report(df, hub_caps=(None, a.hub_cap))

    Xb = base_features(df)
    G = pd.concat([windowed_link_features(df), burst_features(df),
                   component_features(df, hub_cap=a.hub_cap)], axis=1)
    variants = {
        "A base": Xb,
        "B base+graph": pd.concat([Xb, G], axis=1),
        "C noC": Xb.drop(columns=C_COLS),
        "D noC+graph": pd.concat([Xb.drop(columns=C_COLS), G], axis=1),
    }
    print(f"Features: base {Xb.shape[1]}, graph {G.shape[1]}  ({time.time()-t0:.0f}s)")

    y = df["isFraud"].to_numpy()
    amt = df["TransactionAmt"].to_numpy()
    n = len(df)
    i1, i2 = int(n * 0.70), int(n * 0.80)  # train | valid | test, strictly in time order
    sl = {"tr": slice(0, i1), "va": slice(i1, i2), "te": slice(i2, n)}
    print(f"Split by time -> train {i1:,}  valid {i2-i1:,}  test {n-i2:,} (test fraud rate {y[sl['te']].mean():.2%})")
    drift_report(G, i1, i2)

    res = {name: {"va": [], "te": []} for name in variants}
    scored, imp = None, {}
    for seed in range(seeds):
        for name, X in variants.items():
            pv, pt, fi = fit_predict(X.iloc[sl["tr"]], y[sl["tr"]], X.iloc[sl["va"]], y[sl["va"]],
                                     X.iloc[sl["te"]], seed, a.model)
            res[name]["va"].append(metrics(y[sl["va"]], pv, amt[sl["va"]]))
            res[name]["te"].append(metrics(y[sl["te"]], pt, amt[sl["te"]]))
            if seed == 0:
                if scored is None:
                    scored = df.iloc[sl["te"]][["TransactionID", "TransactionDT", "TransactionAmt", "isFraud",
                                                "card_id", "device_fp", "email_addr"]].copy()
                scored["score_" + name.split()[0]] = pt
                if name.startswith("D") or (name.startswith("B") and not imp):
                    imp = fi
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    names = list(variants)
    for part, title in (("va", "VALIDATION (use this for decisions)"), ("te", "TEST (report only, do not tune on it)")):
        print(f"\n=== {title}; mean over {seeds} seed(s) ===")
        print(f"{'metric':<24}" + "".join(f"{n:>15}" for n in names))
        for k in res[names[0]][part][0]:
            print(f"{k:<24}" + "".join(f"{np.mean([r[k] for r in res[n][part]]):>15.4f}" for n in names))
        for k in ("PR-AUC", "Recall@1%"):
            dB = np.mean([r[k] for r in res["B base+graph"][part]]) - np.mean([r[k] for r in res["A base"][part]])
            dD = np.mean([r[k] for r in res["D noC+graph"][part]]) - np.mean([r[k] for r in res["C noC"][part]])
            print(f"  {k}: graph effect with all columns {dB:+.4f} | graph effect without C columns {dD:+.4f}")
    if imp:
        top = sorted(imp.items(), key=lambda kv: -kv[1])[:12]
        print("\nTop features:", ", ".join(f"{k} ({v:.3f})" for k, v in top))

    os.makedirs("outputs", exist_ok=True)
    json.dump({"rows": len(df), "seeds": seeds, "model": a.model, "hub_cap": a.hub_cap,
               "results": {n: {p: {k: float(np.mean([r[k] for r in res[n][p]])) for k in res[n][p][0]}
                               for p in ("va", "te")} for n in names}},
              open("outputs/results.json", "w"), indent=2)
    scored.to_pickle("outputs/test_scored.pkl")
    print("\nSaved outputs/results.json and outputs/test_scored.pkl")


if __name__ == "__main__":
    main()
