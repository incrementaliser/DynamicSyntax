# dynamicsyntax

`dynamicsyntax` is the python implementation of the **[DyLan](https://github.com/Dynamics-of-Language)** the Dynamic Syntax parser. It has been (vibe) translated from Java to Python by [@incrementaliser](https://github.com/incrementaliser) using Cursor, and is still under development.

## Installation

You need [uv](https://docs.astral.sh/uv/) and a Python that satisfies **`>=3.11`** (see `pyproject.toml`).

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

## Examples

For some code examples, please see [this Google Colab notebook](https://colab.research.google.com/drive/1ofpCLwLOE88AvbtR3gh5uisaYkpWGNP9?usp=sharing).

## Testing

See [testing.md](testing.md) for how to run pytest with **uv**, show per-test pass/fail output, and useful options (`-v`, `--lf`, subsets, and CI).

## Get Involved

Contributions (issues, PRs, [donations](https://buymeacoffee.com/incrementaliser)) are very welcome! Since this project has been translated using Cursor and tested only on Windows, there is a small chance errors exist. I would be grateful if you could report them to me through the above channels. Although, it is important to mention that some of the most critical methods here have been verified via unit tests, so there is a high chance the migration from Java has been corrrect so far.
