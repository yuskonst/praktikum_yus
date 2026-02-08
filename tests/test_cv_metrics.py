import numpy as np
import pandas as pd


def test_purged_kfold_split_respects_embargo(forecasting_module):
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    splits = forecasting_module.purged_kfold_split(idx, k=3, embargo_frac=0.1)
    assert len(splits) == 3
    for tr_idx, va_idx in splits:
        assert set(tr_idx).isdisjoint(set(va_idx))


def test_error_metrics_return_expected_types(forecasting_module):
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = np.array([1.1, 1.9, 3.2, 3.8])
    assert forecasting_module.RMSE(y, yhat) > 0
    assert forecasting_module.MAE(y, yhat) > 0
    assert forecasting_module.MASE(y, yhat, m=1) >= 0
    assert 0 <= forecasting_module.MDA(y, yhat) <= 1


def test_diebold_mariano_outputs_stat_and_pvalue(forecasting_module):
    y = np.linspace(1, 5, 20)
    y1 = y + 0.1
    y2 = y - 0.1
    stat, p = forecasting_module.diebold_mariano(y, y1, y2, h=2)
    assert np.isfinite(stat)
    assert 0 <= p <= 1
