import numpy as np
import optuna
import pandas as pd
import pytest


def _tiny_series(length=40):
    idx = pd.date_range("2024-01-01", periods=length, freq="D")
    data = {
        "feat1": np.sin(np.linspace(0, 3, length)),
        "feat2": np.cos(np.linspace(0, 3, length)),
    }
    X = pd.DataFrame(data, index=idx)
    y = pd.Series(np.linspace(0, 1, length), index=idx)
    return X, y


@pytest.fixture
def tiny_config(forecasting_module):
    return {
        "optimizer": "Adam",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "clip_grad_norm": 1.0,
        "amp": False,
        "grad_accum_steps": 1,
        "scheduler": {"warmup_frac": 0.1},
        "batch_size": 4,
        "epochs": 2,
        "seq_len": 4,
    }


@pytest.fixture
def cpu_loader(monkeypatch, forecasting_module):
    if not forecasting_module.TORCH_OK:
        pytest.skip("PyTorch not available")

    base_loader = forecasting_module.DataLoader

    class _PatchedLoader(base_loader):
        def __init__(
            self,
            dataset,
            batch_size,
            shuffle,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            **kwargs,
        ):
            super().__init__(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=False,
                drop_last=drop_last,
                **kwargs,
            )

    monkeypatch.setattr(forecasting_module, "DataLoader", _PatchedLoader)

    def _loader(X, y, seq_len, batch, shuffle):
        ds = forecasting_module.SeqDataset(X, y, seq_len)
        return forecasting_module.DataLoader(
            ds, batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=False, drop_last=False
        )

    monkeypatch.setattr(forecasting_module, "make_loader", _loader)


def test_train_one_model_runs(forecasting_module, tiny_config, cpu_loader):
    if not forecasting_module.TORCH_OK:
        pytest.skip("PyTorch not available")
    X, y = _tiny_series()
    model = forecasting_module.LSTMForecaster(in_dim=X.shape[1], hidden=8, layers=1, dropout=0.0)
    trained, rmse = forecasting_module.train_one_model(model, X, y, X, y, tiny_config, device="cpu", use_ema=False)
    assert rmse < 1.0
    assert any(p.requires_grad for p in trained.parameters())


def test_objective_trial_executes(forecasting_module, tiny_config, cpu_loader, monkeypatch):
    if not forecasting_module.TORCH_OK:
        pytest.skip("PyTorch not available")
    monkeypatch.setitem(forecasting_module.PROD_PROFILE["train"]["seq_len"], "1", tiny_config["seq_len"])
    X, y = _tiny_series(80)
    splits = [(np.arange(0, 50), np.arange(50, 64))]
    trial = optuna.trial.FixedTrial(
        {
            "model": "lstm",
            "lr": tiny_config["lr"],
            "weight_decay": 1e-5,
            "dropout": 0.1,
            "hidden": 8,
            "layers": 1,
        }
    )
    scaler = forecasting_module.compute_target_scaler(y)
    y_norm = forecasting_module.normalize_target(y, scaler)
    result = forecasting_module.objective_trial(
        trial,
        X,
        y_norm,
        y,
        horizon=1,
        cv_splits=splits,
        train_conf=tiny_config,
        model_conf=forecasting_module.PROD_PROFILE["models"],
        scaler=scaler,
    )
    assert isinstance(result, tuple) and len(result) == 2


def test_run_training_for_symbol_minimal(forecasting_module, tiny_config, cpu_loader, monkeypatch, tmp_path):
    if not forecasting_module.TORCH_OK:
        pytest.skip("PyTorch not available")
    monkeypatch.setattr(optuna.trial.Trial, "report", lambda self, value, step: None)
    monkeypatch.setattr(optuna.trial.Trial, "should_prune", lambda self: False)

    class DummyTrial:
        def __init__(self):
            self.params = {
                "model": "lstm",
                "lr": tiny_config["lr"],
                "weight_decay": tiny_config["weight_decay"],
                "dropout": 0.1,
                "hidden": 16,
                "layers": 1,
            }
            self.user_attrs = {}
            self.values = None

        def suggest_categorical(self, name, choices):
            return self.params[name]

        def suggest_float(self, name, low, high, log=False):
            return self.params[name]

        def suggest_int(self, name, low, high, step=1):
            return self.params[name]

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

        def report(self, value, step):
            pass

        def should_prune(self):
            return False

    class DummyStudy:
        def __init__(self):
            self.best_trials = []

        def optimize(self, func, n_trials, show_progress_bar=False):
            trial = DummyTrial()
            trial.values = func(trial)
            self.best_trials = [trial]

    monkeypatch.setattr(forecasting_module.optuna, "create_study", lambda *args, **kwargs: DummyStudy())
    profile = forecasting_module.PROD_PROFILE
    monkeypatch.setitem(profile["hpo"], "n_trials", 1)
    monkeypatch.setitem(profile["hpo"], "pareto_topk", 1)
    monkeypatch.setitem(profile["hpo"], "budget_epochs_fraction", 0.2)
    monkeypatch.setitem(profile["cv"], "purged_kfold", 2)
    monkeypatch.setitem(profile["hpo"]["constraints"], "sMAPE_max", 10.0)
    monkeypatch.setitem(profile["hpo"]["constraints"], "MASE_max", 10.0)
    monkeypatch.setitem(profile["train"], "batch_size", tiny_config["batch_size"])
    monkeypatch.setitem(profile["train"], "grad_accum_steps", tiny_config["grad_accum_steps"])
    monkeypatch.setitem(profile["train"]["epochs"], "1", 2)
    monkeypatch.setitem(profile["train"]["seq_len"], "1", tiny_config["seq_len"])
    monkeypatch.setitem(profile["train"], "lr", tiny_config["lr"])
    monkeypatch.setitem(profile["train"], "weight_decay", tiny_config["weight_decay"])
    monkeypatch.setitem(profile["train"], "dropout", tiny_config["dropout"])

    idx = pd.date_range("2023-01-01", periods=260, freq="B")
    prices = pd.DataFrame(
        {
            "open": np.linspace(100, 110, len(idx)),
            "high": np.linspace(101, 111, len(idx)),
            "low": np.linspace(99, 109, len(idx)),
            "close": np.linspace(100, 112, len(idx)),
            "volume": np.linspace(1000, 1500, len(idx)),
        },
        index=idx,
    )
    out_dir = tmp_path / "runs"
    res = forecasting_module.run_training_for_symbol(
        "AAPL",
        prices,
        hourly=None,
        exog_map={},
        fx_map=None,
        out_dir=str(out_dir),
        horizons=[1],
    )
    assert 1 in res
