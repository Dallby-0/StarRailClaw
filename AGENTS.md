# Development environment

- Use `.venv\Scripts\python.exe` as the project Python interpreter on Windows.
- Run tests with `.venv\Scripts\python.exe -m pytest --basetemp=.pytest-tmp` so Codex does not depend on the user-level temporary directory.
- Install development dependencies with `.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`.
- Do not create another virtual environment or reinstall dependencies unless the existing environment is missing a required package.
- Invoke the virtual environment's Python executable directly; do not rely on activation persisting between commands.
