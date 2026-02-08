import numpy as np
import pandas as pd


def test_hourly_embedding_handles_sparse_series(forecasting_module):
    idx = pd.date_range("2024-01-01", periods=48, freq="H")
    series = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    emb = forecasting_module.hourly_embedding_per_day(series, horizon_days=1)
    assert not emb.empty
    # Two distinct days
    assert len(emb) == 2
    assert "hret_mean" in emb.columns


def test_hourly_embedding_returns_empty_for_no_data(forecasting_module):
    empty = pd.Series(dtype=float)
    emb = forecasting_module.hourly_embedding_per_day(empty, horizon_days=1)
    assert emb.empty


def test_compute_fxvol_rv_pc1_has_finite_values(forecasting_module):
    idx = pd.date_range("2023-01-01", periods=200, freq="B")
    fx_map = {}
    rng = np.random.default_rng(0)
    for pair in ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"]:
        base = np.linspace(1.0, 1.2, len(idx))
        noise = rng.normal(0, 0.001, len(idx))
        fx_map[pair] = pd.Series(base + noise, index=idx)
    pc1 = forecasting_module.compute_fxvol_rv_pc1(fx_map, horizon_days=1, pca_window_days=60)
    assert "fxvol_rv" in pc1.columns
    assert np.isfinite(pc1["fxvol_rv"].dropna()).any()


def test_build_labels_creates_multiindex(forecasting_module):
    df = pd.DataFrame({"close": np.arange(10.0)}, index=pd.date_range("2024-01-01", periods=10, freq="D"))
    labels = forecasting_module.build_labels(df, horizons=[1, 3])
    assert labels.columns.nlevels == 2
    assert ("target", 1) in labels.columns
