import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ts_forecast.volatility_prediction import (  # noqa: E402
    build_sequences,
    iterative_xgb_forecast,
    make_future_index,
    replace_nonfinite,
)


def test_build_sequences_without_leakage():
    data = np.arange(10, dtype=float) / 10.0
    X, y = build_sequences(data, seq_len=3)
    assert X.shape == (7, 3)
    assert y.shape == (7,)
    assert np.allclose(X[0], data[:3])
    assert pytest.approx(y[0]) == data[3]


def test_build_sequences_raises_on_short_series():
    with pytest.raises(ValueError):
        build_sequences([1.0, 2.0], seq_len=5)


def test_iterative_xgb_forecast_rolls_window():
    class DummyModel:
        def __init__(self):
            self.calls = []

        def predict(self, arr):
            self.calls.append(arr.copy())
            return np.array([arr.mean()])

    model = DummyModel()
    seed = np.array([0.0, 0.5, 1.0])
    preds = iterative_xgb_forecast(model, seed, horizon=2)
    assert preds.shape == (2,)
    assert len(model.calls) == 2
    # после первого прогноза окно должно сместиться и использовать предыдущее значение прогноза
    assert model.calls[1][0, -1] == pytest.approx(preds[0])


def test_make_future_index_uses_business_days():
    last = pd.Timestamp("2024-01-05")  # пятница
    idx = make_future_index(last, periods=3)
    assert list(idx.weekday) == [0, 1, 2]  # Пн-Вт-Ср


def test_replace_nonfinite_substitutes_values():
    arr = np.array([1.0, np.nan, np.inf, -np.inf])
    out = replace_nonfinite(arr, fallback=0.5)
    assert np.isfinite(out).all()
    assert out[1] == pytest.approx(0.5)
