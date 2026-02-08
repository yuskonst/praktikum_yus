import sys
import types

import pandas as pd


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _binance_payload():
    candle = [
        1704067200000,  # open_time (2024-01-01 UTC)
        "43000.0",
        "44000.0",
        "42000.0",
        "43500.0",
        "100.0",
        1704153600000,
        "0",
        10,
        "0",
        "0",
        "0",
    ]
    candle2 = [
        1704153600000,  # 2024-01-02 UTC
        "43500.0",
        "45000.0",
        "43000.0",
        "44500.0",
        "120.0",
        1704240000000,
        "0",
        11,
        "0",
        "0",
        "0",
    ]
    return [candle, candle2]


def _install_requests_stub(monkeypatch, get_callable):
    stub = types.ModuleType("requests")
    stub.get = get_callable
    monkeypatch.setitem(sys.modules, "requests", stub)


def test_binance_download_1d_produces_naive_datetime_index(forecasting_module, monkeypatch):
    def fake_get(url, params, timeout):
        assert "klines" in url
        assert params["interval"] == "1d"
        return DummyResponse(_binance_payload())

    _install_requests_stub(monkeypatch, fake_get)

    df = forecasting_module.binance_download_1d("BTCUSDT", start="2024-01-02", end=None)
    assert not df.empty
    assert df.index[0].tzinfo is None
    # first candle filtered by start date
    assert df.index.min().day == 2
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)


def test_binance_download_1h_shared_index_logic(forecasting_module, monkeypatch):
    def fake_get(url, params, timeout):
        assert params["interval"] == "1h"
        return DummyResponse(_binance_payload())

    _install_requests_stub(monkeypatch, fake_get)
    df = forecasting_module.binance_download_1h("BTCUSDT", start=None, end=None)
    assert not df.empty
    assert df.index[0].tzinfo is None


def test_yf_download_falls_back_to_history(forecasting_module, monkeypatch):
    sample = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.5, 2.5],
            "Low": [0.5, 1.5],
            "Close": [1.2, 2.2],
            "Adj Close": [1.1, 2.1],
            "Volume": [100, 150],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    def boom(*args, **kwargs):
        raise RuntimeError("download failed")

    class DummyTicker:
        def history(self, **kwargs):
            return sample

    monkeypatch.setattr(forecasting_module.yf, "download", boom)
    monkeypatch.setattr(forecasting_module.yf, "Ticker", lambda symbol: DummyTicker())

    df = forecasting_module.yf_download_1d("AAPL", "2024-01-01", "2024-01-03")
    assert not df.empty
    assert "adj_close" in df.columns
    assert df.index.tzinfo is None
