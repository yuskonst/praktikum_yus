# Praktikum_yus
проект-4

## Development environment
- Activate the bundled virtual environment before working: .venv\Scripts\Activate.ps1 (PowerShell) or .venv\Scripts\activate (cmd).
- If you use VS Code, select .venv\Scripts\python.exe via Python: Select Interpreter.
- In PyCharm, go to Settings > Project > Python Interpreter, click the gear icon, choose Add, and point to .venv\Scripts\python.exe.
- After switching the interpreter, reload the window or invalidate caches so warnings clear and the analyzer picks up installed packages.

## Installed packages
The virtual environment reuses global site packages (created with --system-site-packages), so existing torch and optuna installations are visible inside it. Reinstall with python -m pip install --upgrade torch optuna if you need different versions.