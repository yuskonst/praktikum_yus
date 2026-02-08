import numpy as np
import pandas as pd


def test_assemble_features_produces_targets(forecasting_module):
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    data = pd.DataFrame(
        {
            "open": np.linspace(100, 120, len(idx)),
            "high": np.linspace(101, 121, len(idx)),
            "low": np.linspace(99, 119, len(idx)),
            "close": np.linspace(100, 122, len(idx)),
            "volume": np.linspace(1000, 2000, len(idx)),
        },
        index=idx,
    )
    prep = forecasting_module.assemble_features("AAPL", data, horizons=[1, 5], hourly=None, exog_vol_map=None, fx_close_map=None)
    assert not prep.X.empty
    assert ("target", 1) in prep.Y.columns
