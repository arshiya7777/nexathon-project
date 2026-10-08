"""Graph / relationship features for the fraud-ring project.

Everything here is POINT-IN-TIME: a transaction's features use only transactions
that happened before it (or in the same row), never the future and never labels.
"""
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

DAY = 86400
CARD_COLS = ["card1", "card2", "card3", "card4", "card5", "card6", "addr1"]
# Things that can link two different cards together (graph "entities").
LINK_KEYS = ["device_fp", "email_addr"]


def _text(series):
    """Text version of a column where a missing value becomes the text 'NA'
    (works the same in old and new pandas; new pandas would otherwise keep it missing)."""
    return series.astype(object).where(series.notna(), "NA").astype(str)


def _join(df, cols):
    s = _text(df[cols[0]])
    for c in cols[1:]:
        s = s + "|" + _text(df[c])
    return s


def add_ids(df):
    """card_id = proxy for a client (IEEE-CIS has no account id).
    device_fp = device fingerprint (only where device info exists).
    email_addr = email domain + billing address."""
    df = df.copy()
    df["card_id"] = _join(df, CARD_COLS)
    assert df["card_id"].notna().all()  # every transaction must have a card_id
    has_dev = df["DeviceInfo"].notna()
    fp = _join(df, ["DeviceInfo", "id_30", "id_31", "id_33"])
    df["device_fp"] = fp.where(has_dev)
    has_mail = df["P_emaildomain"].notna() & df["addr1"].notna()
    df["email_addr"] = _join(df, ["P_emaildomain", "addr1", "addr2"]).where(has_mail)
    return df


def cumulative_link_features(df, keys=LINK_KEYS):
    """For each link key: how many DIFFERENT cards have used the same device/email-address
    up to and including this transaction, and how many transactions so far.
    Needs df sorted by TransactionDT."""
    out = {}
    for k in keys:
        valid = df[k].notna()
        sub = df.loc[valid, ["card_id", k]]
        first_time_pair = (~sub.duplicated(subset=[k, "card_id"])).astype(int)
        out[f"{k}_n_cards"] = first_time_pair.groupby(sub[k]).cumsum().reindex(df.index)
        out[f"{k}_n_tx"] = (sub.groupby(k).cumcount() + 1).reindex(df.index)
    return pd.DataFrame(out, index=df.index)


def windowed_link_features(df, keys=LINK_KEYS, lags=7):
    """Stationary version of the link counts (they do NOT keep growing with time).
    For every link key (device / email+address):
      *_cards_today / *_tx_today : different cards / transactions on the same key so far TODAY
      *_card_days_prev / *_tx_prev : card-days / transactions on the same key in the previous `lags` COMPLETED days
    Only past information is used."""
    day = (df["TransactionDT"] // DAY).astype(int)
    out = {}
    for k in keys:
        valid = df[k].notna()
        sub = pd.DataFrame({"k": df.loc[valid, k], "day": day[valid], "card": df.loc[valid, "card_id"]})
        first_today = (~sub.duplicated(subset=["k", "day", "card"])).astype(int)
        out[f"{k}_cards_today"] = first_today.groupby([sub["k"], sub["day"]]).cumsum().reindex(df.index)
        out[f"{k}_tx_today"] = (sub.groupby(["k", "day"]).cumcount() + 1).reindex(df.index)
        cards_per_day = sub.drop_duplicates(["k", "day", "card"]).groupby(["k", "day"]).size()
        tx_per_day = sub.groupby(["k", "day"]).size()
        cards_prev = np.zeros(len(sub))
        tx_prev = np.zeros(len(sub))
        for lag in range(1, lags + 1):
            idx = pd.MultiIndex.from_arrays([sub["k"], sub["day"] - lag])
            cards_prev += cards_per_day.reindex(idx).fillna(0).to_numpy()
            tx_prev += tx_per_day.reindex(idx).fillna(0).to_numpy()
        out[f"{k}_card_days_prev"] = pd.Series(cards_prev, index=sub.index).reindex(df.index)
        out[f"{k}_tx_prev"] = pd.Series(tx_prev, index=sub.index).reindex(df.index)
    return pd.DataFrame(out, index=df.index)


def burst_features(df):
    """How fast is the same card transacting? (seconds since previous / 3 transactions ago)."""
    g = df.groupby("card_id")["TransactionDT"]
    day = (df["TransactionDT"] // DAY).astype(int)
    return pd.DataFrame({
        # capped at 7 days so very long gaps do not make the feature drift over time
        "card_gap_prev": (df["TransactionDT"] - g.shift(1)).clip(upper=7 * DAY),
        "card_span_last3": (df["TransactionDT"] - g.shift(3)).clip(upper=7 * DAY),
        "card_n_today": df.groupby(["card_id", day]).cumcount(),
    }, index=df.index)


def _snapshot_components(card_codes, key_codes_list, rows, n_cards, hub_cap):
    """Connected components of the bipartite graph (cards <-> entities) built from `rows`.
    Entities touched by more than `hub_cap` different cards are dropped (hub filtering)."""
    seen_cards = np.zeros(n_cards, dtype=bool)
    seen_cards[card_codes[rows]] = True
    src, dst, offset = [], [], n_cards
    for kc in key_codes_list:
        ok = kc[rows] >= 0
        pair = np.unique(card_codes[rows][ok].astype(np.int64) * (int(kc.max()) + 2) + kc[rows][ok])
        c = pair // (int(kc.max()) + 2)
        e = pair % (int(kc.max()) + 2)
        if hub_cap is not None:
            deg = np.bincount(e, minlength=int(kc.max()) + 2)
            keep = deg[e] <= hub_cap
            c, e = c[keep], e[keep]
        src.append(c)
        dst.append(e + offset)
        offset += int(kc.max()) + 2
    src, dst = np.concatenate(src), np.concatenate(dst)
    n_nodes = offset
    adj = coo_matrix((np.ones(len(src), dtype=np.int8), (src, dst)), shape=(n_nodes, n_nodes))
    n_comp, comp = connected_components(adj, directed=False)
    cards_in_comp = np.bincount(comp[:n_cards], weights=seen_cards, minlength=n_comp)
    active_entity = np.zeros(n_nodes, dtype=bool)
    active_entity[dst] = True
    ents_in_comp = np.bincount(comp, weights=active_entity, minlength=n_comp)
    return comp, seen_cards, cards_in_comp, ents_in_comp


def component_features(df, keys=LINK_KEYS, snapshot_days=7, hub_cap=30, window_days=28):
    """Weekly snapshots. For a transaction in week j, use the graph built from transactions
    BEFORE week j started (strictly past), looking back `window_days` only so values stay comparable over time. Features: how many cards sit in this card's
    connected component, and how many shared devices/addresses hold it together."""
    t = df["TransactionDT"].to_numpy()
    card_codes = pd.factorize(df["card_id"])[0]
    n_cards = int(card_codes.max()) + 1
    key_codes = [pd.factorize(df[k])[0] for k in keys]
    size = np.full(len(df), np.nan)
    ents = np.full(len(df), np.nan)
    t0, W = t.min(), snapshot_days * DAY
    n_weeks = int((t.max() - t0) // W) + 1
    for j in range(1, n_weeks):
        cut = t0 + j * W
        past = np.where((t < cut) & (t >= cut - window_days * DAY))[0]  # sliding window, not all history
        cur = np.where((t >= cut) & (t < cut + W))[0]
        if len(cur) == 0 or len(past) == 0:
            continue
        comp, seen, cards_in_comp, ents_in_comp = _snapshot_components(card_codes, key_codes, past, n_cards, hub_cap)
        cc = card_codes[cur]
        known = seen[cc]
        size[cur[known]] = cards_in_comp[comp[cc[known]]]
        ents[cur[known]] = ents_in_comp[comp[cc[known]]]
    return pd.DataFrame({"comp_n_cards": size, "comp_n_entities": ents}, index=df.index)


def giant_component_report(df, keys=LINK_KEYS, hub_caps=(None, 30)):
    """Diagnostic: does one giant blob swallow everything? (If yes, component features are useless.)"""
    t = df["TransactionDT"].to_numpy()
    card_codes = pd.factorize(df["card_id"])[0]
    n_cards = int(card_codes.max()) + 1
    key_codes = [pd.factorize(df[k])[0] for k in keys]
    rows = np.arange(len(df))
    print("Giant-component check (last snapshot = all data):")
    for cap in hub_caps:
        comp, seen, cards_in_comp, _ = _snapshot_components(card_codes, key_codes, rows, n_cards, cap)
        big = cards_in_comp.max()
        multi = int((cards_in_comp >= 2).sum())
        print(f"  hub_cap={cap!s:>4}: largest component = {int(big)} cards "
              f"({big / seen.sum():.1%} of all cards), components with >=2 cards: {multi}")
