"""Point-in-time test: features of early transactions must not change when future data is removed or labels change."""
import numpy as np
import pandas as pd
from features import add_ids, burst_features, component_features, windowed_link_features
from run_experiment import load


def feats(df):
    return pd.concat([windowed_link_features(df), burst_features(df), component_features(df, hub_cap=30)], axis=1)


def test_features_do_not_depend_on_future_or_labels():
    df = add_ids(load(rows=60_000))
    full = feats(df)
    cut = int(len(df) * 0.6)
    part = feats(df.iloc[:cut].copy())
    pd.testing.assert_frame_equal(full.iloc[:cut].reset_index(drop=True), part.reset_index(drop=True), check_dtype=False)
    flipped = df.copy()
    flipped["isFraud"] = 1 - flipped["isFraud"]
    pd.testing.assert_frame_equal(full, feats(flipped), check_dtype=False)


def test_time_split_order():
    df = load(rows=10_000)
    assert df["TransactionDT"].is_monotonic_increasing
