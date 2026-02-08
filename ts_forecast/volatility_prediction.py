"""Volatility forecasting pipeline with Holt-Winters + XGBoost ensemble.

Этот модуль выделяет функциональность из черновика Volatility Prediction,
чтобы её можно было переиспользовать и покрыть тестами.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date
from math import sqrt
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - зависит от окружения
    raise RuntimeError("yfinance не установлен в окружении") from exc


DEFAULT_START_DATE = "2018-01-01"
PLOTS_DIR = Path("artifacts_prod") / "volatility_forecast"


@dataclass
class ForecastConfig:
    symbol: str = "AAPL"
    start: str = DEFAULT_START_DATE
    end: Optional[str] = None
    sequence_length: int = 30
    horizon_days: int = 30
    volatility_window: int = 20


def _normalize_end(end: Optional[str]) -> str:
    if end:
        return end
    return date.today().isoformat()


def download_price_history(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """Загружает дневные котировки и нормализует индекс."""

    end_date = _normalize_end(end)
    data = yf.download(symbol, start=start, end=end_date)
    if data.empty:
        raise ValueError(f"Не удалось загрузить данные для {symbol} в диапазоне {start} - {end_date}")
    data.index = pd.to_datetime(data.index).tz_localize(None)

    if isinstance(data.columns, pd.MultiIndex):
        # Если MultiIndex вида (Price, Ticker) и тикер один, убираем уровень.
        last_level = data.columns.get_level_values(-1)
        if len(set(last_level)) == 1:
            data.columns = data.columns.droplevel(-1)
        else:
            data.columns = ["_".join(str(part) for part in col if part) for col in data.columns]

    required = {"Close", "Open", "High", "Low", "Volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Не найдены необходимые поля {missing} в загружаемых данных для {symbol}")

    return data


def prepare_volatility_frame(data: pd.DataFrame, volatility_window: int) -> pd.DataFrame:
    """Строит DataFrame с доходностями и исторической волатильностью."""

    frame = data.copy()
    frame["Returns"] = frame["Close"].pct_change().fillna(0.0)
    frame["Volatility"] = frame["Returns"].rolling(window=volatility_window).std()
    frame = frame.dropna(subset=["Volatility"]).copy()
    if frame.empty:
        raise ValueError("Недостаточно данных для расчета волатильности — увеличьте диапазон загрузки")
    return frame


def scale_target(series: pd.Series) -> Tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    values = series.values.reshape(-1, 1)
    scaled = scaler.fit_transform(values).ravel()
    return scaled, scaler


def build_sequences(values: Sequence[float], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Готовит обучающие сэмплы без утечек из будущего."""

    if seq_len <= 0:
        raise ValueError("sequence_length должен быть положительным")
    arr = np.asarray(values, dtype=float)
    if arr.size <= seq_len:
        raise ValueError("Недостаточно данных для построения последовательностей")

    samples, targets = [], []
    for idx in range(seq_len, arr.size):
        samples.append(arr[idx - seq_len : idx])
        targets.append(arr[idx])
    return np.stack(samples), np.asarray(targets)


def train_best_holt_winters(series: pd.Series) -> Tuple[ExponentialSmoothing, float]:
    """Перебирает конфигурации Holt-Winters и возвращает модель с минимальным RMSE."""

    best_model = None
    best_rmse = float("inf")
    trend_opts = ["add", "mul", None]
    seasonal_opts = ["add", "mul", None]
    seasonal_periods = [10, 20]

    for trend in trend_opts:
        for seasonal in seasonal_opts:
            for period in seasonal_periods:
                if seasonal is None and period is not None:
                    continue
                try:
                    model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=period)
                    fitted = model.fit(optimized=True)
                    fitted_values = fitted.fittedvalues
                    rmse = sqrt(mean_squared_error(series.iloc[-len(fitted_values) :], fitted_values))
                except Exception:
                    continue
                if np.isnan(rmse):
                    continue
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = fitted

    if best_model is None:
        raise RuntimeError("Не удалось подобрать устойчивую модель Holt-Winters")
    return best_model, best_rmse


def train_xgb_with_grid(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[XGBRegressor, float]:
    """Подбирает гиперпараметры XGBoost на данных обучающей выборки."""

    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=random_state,
        tree_method="hist",
        enable_categorical=False,
        verbosity=0,
    )
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    search = GridSearchCV(
        base_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    best_model = search.best_estimator_
    best_rmse = sqrt(abs(search.best_score_))
    return best_model, best_rmse


def iterative_xgb_forecast(model: XGBRegressor, seed_sequence: np.ndarray, horizon: int) -> np.ndarray:
    """Формирует многошаговый прогноз, подавая на вход предсказания прошлых шагов."""

    if horizon <= 0:
        return np.array([])
    window = np.asarray(seed_sequence, dtype=float).copy()
    preds = []
    for _ in range(horizon):
        features = window.reshape(1, -1)
        next_pred = float(model.predict(features))
        preds.append(next_pred)
        window = np.roll(window, -1)
        window[-1] = next_pred
    return np.asarray(preds)


def make_future_index(last_timestamp: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    if periods <= 0:
        return pd.DatetimeIndex([])
    start = (last_timestamp + BDay()).normalize()
    return pd.bdate_range(start, periods=periods)


def replace_nonfinite(values: Sequence[float], fallback: float) -> np.ndarray:
    """Заменяет NaN/inf значения на безопасный скаляр."""

    arr = np.asarray(values, dtype=float)
    if np.isfinite(arr).all():
        return arr
    return np.where(np.isfinite(arr), arr, fallback)


def _ensure_plots_dir() -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR


def _plot_forecast(
    history_idx: pd.Index,
    history_values: np.ndarray,
    future_idx: pd.Index,
    future_values: np.ndarray,
    label: str,
    symbol: str,
) -> Path:
    out_dir = _ensure_plots_dir()
    path = out_dir / f"{symbol}_{label.lower()}_forecast.png"
    plt.figure(figsize=(12, 6))
    plt.plot(history_idx, history_values, label="Историческая волатильность")
    plt.plot(future_idx, future_values, label=f"Прогноз {label}")
    plt.axvline(history_idx[-1], color="gray", linestyle="--", label="Граница прогноза")
    plt.title(f"{symbol}: прогноз волатильности ({label})")
    plt.xlabel("Дата")
    plt.ylabel("Волатильность")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def generate_forecast_plots(
    volatility: pd.Series,
    future_idx: pd.Index,
    hw: np.ndarray,
    xgb: np.ndarray,
    ensemble: np.ndarray,
    symbol: str,
) -> Dict[str, str]:
    history_idx = volatility.index
    history_values = volatility.values
    plot_paths = {}
    plot_paths["HW"] = str(_plot_forecast(history_idx, history_values, future_idx, hw, "HW", symbol))
    plot_paths["XGB"] = str(_plot_forecast(history_idx, history_values, future_idx, xgb, "XGB", symbol))
    plot_paths["Ensemble"] = str(
        _plot_forecast(history_idx, history_values, future_idx, ensemble, "Ensemble", symbol)
    )
    return plot_paths


def run_pipeline(cfg: ForecastConfig) -> Dict[str, object]:
    """Полный цикл обучения, оценки и построения прогноза."""

    raw = download_price_history(cfg.symbol, cfg.start, cfg.end)
    prepared = prepare_volatility_frame(raw, cfg.volatility_window)

    volatility = prepared["Volatility"]
    scaled, scaler = scale_target(volatility)
    sequences, targets = build_sequences(scaled, cfg.sequence_length)

    if len(sequences) < 2:
        raise ValueError("Недостаточно выборки для обучения")

    split_idx = max(int(len(sequences) * 0.8), 1)
    if split_idx >= len(sequences):
        split_idx = len(sequences) - 1

    X_train = sequences[:split_idx]
    y_train = targets[:split_idx]
    X_test = sequences[split_idx:]
    y_test = targets[split_idx:]

    hw_train_cutoff = cfg.sequence_length + len(X_train)
    hw_train_series = volatility.iloc[:hw_train_cutoff]
    hw_model, _ = train_best_holt_winters(hw_train_series)
    hw_forecast = hw_model.forecast(len(X_test))
    hw_forecast = replace_nonfinite(hw_forecast, fallback=float(hw_train_series.iloc[-1]))

    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    hw_rmse = sqrt(mean_squared_error(y_test_actual, hw_forecast))

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    best_xgb, _ = train_xgb_with_grid(X_train_flat, y_train)
    xgb_pred_scaled = best_xgb.predict(X_test_flat)
    xgb_pred = scaler.inverse_transform(xgb_pred_scaled.reshape(-1, 1)).ravel()
    xgb_pred = replace_nonfinite(xgb_pred, fallback=float(y_test_actual[-1]))
    xgb_rmse = sqrt(mean_squared_error(y_test_actual, xgb_pred))

    ensemble_pred = (hw_forecast + xgb_pred) / 2
    ensemble_pred = replace_nonfinite(ensemble_pred, fallback=float(y_test_actual[-1]))
    ensemble_rmse = sqrt(mean_squared_error(y_test_actual, ensemble_pred))

    # Для финального прогноза обучаем модели заново на всей доступной истории
    hw_full_model, _ = train_best_holt_winters(volatility)
    best_params = best_xgb.get_params()
    final_xgb = XGBRegressor(**best_params)
    final_xgb.fit(sequences.reshape(len(sequences), -1), targets)

    future_hw = hw_full_model.forecast(cfg.horizon_days)
    future_hw = replace_nonfinite(future_hw, fallback=float(volatility.iloc[-1]))
    seed = scaled[-cfg.sequence_length :]
    future_xgb_scaled = iterative_xgb_forecast(final_xgb, seed, cfg.horizon_days)
    future_xgb = scaler.inverse_transform(future_xgb_scaled.reshape(-1, 1)).ravel()
    future_xgb = replace_nonfinite(future_xgb, fallback=float(volatility.iloc[-1]))
    future_ensemble = (future_hw + future_xgb) / 2
    future_ensemble = replace_nonfinite(future_ensemble, fallback=float(volatility.iloc[-1]))
    future_idx = make_future_index(volatility.index[-1], cfg.horizon_days)

    future_frame = pd.DataFrame(
        {
            "HW_Volatility": future_hw,
            "XGB_Volatility": future_xgb,
            "Ensemble_Volatility": future_ensemble,
        },
        index=future_idx,
    )

    plot_paths = generate_forecast_plots(
        volatility=volatility,
        future_idx=future_idx,
        hw=future_hw,
        xgb=future_xgb,
        ensemble=future_ensemble,
        symbol=cfg.symbol,
    )

    metrics = {
        "HW_RMSE": hw_rmse,
        "XGB_RMSE": xgb_rmse,
        "Ensemble_RMSE": ensemble_rmse,
    }

    return {"metrics": metrics, "forecast": future_frame, "config": cfg, "plot_paths": plot_paths}


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    cfg = ForecastConfig()
    result = run_pipeline(cfg)

    print("=== Итоги backtest-валидации ===")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.6f}")

    print("\n=== Прогноз волатильности на ближайшие 5 торговых дней ===")
    print(result["forecast"])
    print("\nПути к графикам:")
    for label, path in result["plot_paths"].items():
        print(f"{label}: {path}")


if __name__ == "__main__":  # pragma: no cover - точка входа для скрипта
    main()
