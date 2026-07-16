# AGENTS.md

## Cursor Cloud specific instructions

`dynamicsyntax` (a.k.a. DyLan) is a single Python package (`>=3.11`) managed with **uv**.
There is no database, container stack, or backend API — every surface runs as a local
Python process or static files. The startup update script already runs `uv sync --group dev`,
so a `.venv` with dev tools is present when a session begins.

### Product surfaces
- **Core library** (`dynamicsyntax` / `dylan`): programmatic incremental parser producing TTR semantics.
- **Induction CLI** (`dsttr-induction`): YAML-driven EM lexicon learning + evaluation.
- **Flet GUI** (`dylan-gui`): desktop parser UI (also runnable in a browser).
- **Pyodide web shell** (`web/`): static browser app running the parser via WebAssembly.

### Running / testing (standard commands live in `README.md`, `pyproject.toml` `[project.scripts]`, and `.github/workflows/ci.yml`)
- Lint (CI scope): `uv run ruff check tests` (locally you can also lint `src`).
- Tests: `uv run pytest -q` — full suite takes ~1 min; a few induction tests use `@pytest.mark.timeout`.
- Build wheel/sdist: `uv build`.
- Library hello-world: `uv run python -c "import dynamicsyntax as ds; print(ds.parse('a man knows you','ttr').semantics)"`.
- Induction pipeline: `uv run dsttr-induction --config configs/induction/holdout.yaml` (writes to `out/runs/<timestamp>_<name>/`; ~2 min for `class1.txt`).

### Non-obvious gotchas
- **Prefer `uv run <cmd>`** rather than activating the venv; it always uses the project `.venv`.
- **Flet GUI in this headless VM must use web mode**: run `DYLAN_FLET_WEB=1 uv run dylan-gui`
  and open `http://127.0.0.1:8550/`. Two caveats in web mode:
  - It logs a non-fatal `TimeoutException ... Window(...).center` traceback at startup
    (`page.window.center()` is desktop-only); the UI still renders and works.
  - The "Load grammar" button uses a **native directory picker** that does not work in a
    browser, so grammar loading (and therefore parsing) can't be completed via the web GUI.
    Use the library API / `dsttr-induction` CLI for end-to-end parsing in this environment,
    or run the GUI in native desktop mode where the picker works.
- **Pyodide web shell**: build + copy the wheel first (`uv build` then
  `uv run python scripts/sync_web_wheel.py`, producing `web/dist/package.whl`), then serve
  with `uv run dylan-serve-web` (port 8000). It needs network access to the pinned Pyodide
  CDN (`cdn.jsdelivr.net/pyodide/v0.26.4`). Note: the bundled `micropip` rejects the wheel
  because it is served as `package.whl` (`InvalidWheelFilename: wrong number of parts`), so
  Python fails to boot in the browser — a pre-existing code-level issue, not an env/setup one.
- Conventional Commits are required (see `docs/CONTRIBUTING.md`); the `commit-msg` hook in
  `scripts/git-hooks/` is optional and not enabled by default.
