import numpy as np
import pandas as pd


def test_apply_calendar_filters_weekends_for_equities(forecasting_module):
    idx = pd.date_range("2024-01-05", periods=4, freq="D")  # Fri-Mon
    df = pd.DataFrame({"close": [1, 2, 3, 4]}, index=idx)
    filtered = forecasting_module.apply_calendar("AAPL", df)
    # Saturday/Sunday removed
    assert all(ts.weekday() < 5 for ts in filtered.index)


def test_apply_calendar_keeps_crypto_weekends(forecasting_module):
    idx = pd.date_range("2024-01-06", periods=2, freq="D")  # Sat-Sun
    df = pd.DataFrame({"close": [1, 2]}, index=idx)
    filtered = forecasting_module.apply_calendar("BTCUSDT", df)
    assert len(filtered) == 2


def test_logdiff_handles_zero_entries(forecasting_module):
    s = pd.Series([1.0, 0.0, 2.0, 4.0], index=pd.date_range("2024-01-01", periods=4))
    out = forecasting_module.logdiff(s)
    assert np.isfinite(out.dropna()).all()


def test_causal_sg_fallback_without_scipy(monkeypatch, forecasting_module):
    monkeypatch.setattr(forecasting_module, "SCIPY_OK", False)
    series = pd.Series(np.linspace(1, 5, 10), index=pd.date_range("2024-01-01", periods=10, freq="D"))
    out = forecasting_module.causal_sg(series, window=5, poly=2)
    assert len(out) == len(series)


def test_revin_returns_per_symbol_stats(forecasting_module):
    df = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "feat1": [1.0, 2.0, 3.0, 4.0],
            "feat2": [2.0, 4.0, 6.0, 8.0],
        },
        index=pd.date_range("2024-01-01", periods=4, freq="D"),
    )
    normalized, stats = forecasting_module.revin(df.copy(), by="symbol")
    assert set(stats.keys()) == {"AAA", "BBB"}
    for sym_stats in stats.values():
        assert "feat1" in sym_stats["last_mu"]
    tail = normalized.groupby("symbol").tail(1)[["feat1", "feat2"]]
    assert tail.isna().sum().sum() == 0
