import numpy as np
import pandas as pd
import pytest

try:
    import torch
except ImportError:  # pragma: no cover - executed only if torch missing
    torch = None


def test_seqdataset_and_models_forward(forecasting_module):
    if torch is None or not forecasting_module.TORCH_OK:
        pytest.skip("PyTorch not available")

    df = pd.DataFrame(
        {
            "feat1": np.arange(10, dtype=float),
            "feat2": np.arange(10, dtype=float) * 2,
            "cat": list("abcdefghij"),
        },
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )
    y = pd.Series(np.linspace(0, 1, 10), index=df.index)
    ds = forecasting_module.SeqDataset(df, y, seq_len=3)
    assert len(ds) == 8
    xb, yb = ds[0]
    assert xb.shape == (3, 2)  # only numeric cols

    lstm = forecasting_module.LSTMForecaster(in_dim=2, hidden=8, layers=1, dropout=0.0)
    informer = forecasting_module.InformerTS(in_dim=2, d_model=8, n_heads=2, layers=1, ff_mult=1, dropout=0.0)
    vae = forecasting_module.VAE(in_dim=2, latent=4, hidden=8, layers=1, dropout=0.0)

    batch = torch.from_numpy(np.stack([xb, xb])).float()
    lstm_out = lstm(batch)
    informer_out = informer(batch)
    recon, mu, lv = vae(batch[:, -1, :])

    assert lstm_out.shape == (2,)
    assert informer_out.shape == (2,)
    assert recon.shape[-1] == 2
    assert mu.shape[-1] == 4
    assert lv.shape[-1] == 4
