# dynamicsyntax

`dynamicsyntax` is the python implementation of the **[DyLan](https://github.com/Dynamics-of-Language)** the Dynamic Syntax parser. It has been (vibe) translated from Java to Python by [@incrementaliser](https://github.com/incrementaliser) using Cursor, and is still under development.

## Installation

You need [uv](https://docs.astral.sh/uv/) and a Python that satisfies **`>=3.12`** (see `pyproject.toml`).

1. Create a virtual environment with a fixed Python version (run this in the directory where you want `.venv`, e.g. the repo root for a clone):

   ```bash
   uv venv --python 3.12
   ```

2. Activate it. On Linux or macOS:

   ```bash
   source .venv/bin/activate
   ```

   On Windows, use `.venv\Scripts\activate` (cmd) or `.\.venv\Scripts\Activate.ps1` (PowerShell).

3. Install into that environment, using the published package from PyPI:

   ```bash
   uv pip install dynamicsyntax
   ```

   or editable install from a git clone:

   ```bash
   uv pip install -e .
   ```

## Example

```python
import dynamicsyntax as ds

semantics = ds.parse("a man arrives", "ttr")
print(semantics)
```

The above loads the 2015-english-ttr grammar and parses the utterance "a man arrives".

## Testing

See [testing.md](testing.md) for how to run pytest with **uv**, show per-test pass/fail output, and useful options (`-v`, `--lf`, subsets, and CI).

## GitHub Codespaces

This repository includes a [dev container](.devcontainer/devcontainer.json). After the container finishes `uv sync --group dev`, run the GUI in the browser (port **8550** is forwarded automatically):

```bash
export PATH="$HOME/.local/bin:$PATH"
export DYLAN_FLET_WEB=1
uv run dylan-gui
```

Open the forwarded URL for port 8550 in your browser. You can override the port with `DYLAN_FLET_PORT`.

Contributions (via PRs) are very welcome! Since this project has been translated using Cursor, there is a chance errors have been introduced. We would be grateful if you could report them by opening an issue. Although, it is important to mention that the output of (some of) the methods here have been verified via unit tests.
