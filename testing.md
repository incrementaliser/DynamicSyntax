# Testing guide

Tests live under [`tests/`](tests/) and are run with [pytest](https://pytest.org/). Configuration is in [`pyproject.toml`](pyproject.toml) (`[tool.pytest.ini_options]`: `testpaths`, `pythonpath`).

From the repository root, use **uv** so the same environment as development is used:

```bash
uv sync --group dev
```

## See each test and whether it passed

Verbose mode prints one line per test (file, function name, and `PASSED` / `FAILED`):

```bash
uv run pytest -v
```

## Shorter tracebacks on failures

```bash
uv run pytest -v --tb=short
```

Use `--tb=long` or `--tb=auto` if you need more context for a specific failure.

## Stop at the first failure

```bash
uv run pytest -v -x
```

## Re-run only tests that failed last time

```bash
uv run pytest -v --lf
```

Run the **full** suite but execute previously failed tests first (uses the last run’s cache):

```bash
uv run pytest -v --ff
```

## Run a subset

Single file:

```bash
uv run pytest -v tests/formula/test_ttr_record_type.py
```

Single test:

```bash
uv run pytest -v tests/formula/test_ttr_record_type.py::test_parse_empty_record
```

Match names with `-k`:

```bash
uv run pytest -v -k "ttr_record"
```

## Quiet summary (CI-style)

```bash
uv run pytest -q
```

## Optional HTML report

If you add a plugin such as `pytest-html` to your dev dependencies, you can generate a self-contained report:

```bash
uv run pytest -v --html=report.html --self-contained-html
```

## Continuous integration

The [GitHub Actions workflow](.github/workflows/ci.yml) runs `uv sync --group dev`, `ruff check tests`, and `pytest -q` on pushes and pull requests.
