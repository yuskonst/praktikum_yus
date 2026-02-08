import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def forecasting_module():
    script_path = Path(__file__).resolve().parents[3] / "financial TS forecasting_06.10.py"
    spec = importlib.util.spec_from_file_location("financial_ts_forecasting", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["financial_ts_forecasting"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
