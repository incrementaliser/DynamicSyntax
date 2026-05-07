# dynamicsyntax

`dynamicsyntax` is the python implementation of the **[DyLan](https://github.com/Dynamics-of-Language)** the Dynamic Syntax parser. It has been (vibe) translated from Java to Python by [@incrementaliser](https://github.com/incrementaliser) using Cursor, and is still under development.

## Installation

You need [uv](https://docs.astral.sh/uv/) and a Python that satisfies **`>=3.13`** (see `pyproject.toml`).

1. Create a virtual environment with a fixed Python version (run this in the directory where you want `.venv`, e.g. the repo root for a clone):

   ```bash
   uv venv --python 3.13
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

`dynamicsyntax.parse` returns a **`ParseResult`** (with `.semantics`, `.ok`, `.tree`, `.vis()`, and `.address_order`). This replaces older versions that returned `TTRRecordType | None` directly—use `.semantics` for the final TTR record.

List bundled grammars and (placeholder) datasets:

```python
import dynamicsyntax as ds

print(ds.get_grammars())   # e.g. ['2015-english-ttr', 'ttr', ...]
print(ds.get_datasets()) # [] until datasets ship with the package
```

One-shot parse with a grammar id or alias (`"ttr"` maps to `2015-english-ttr`):

```python
import dynamicsyntax as ds

p = ds.parse("a man arrives", "ttr")
print(p.ok, p.semantics)
p.vis()  # prints the same address-order tree view as the GUI
```

Load a grammar once, then parse without repeating the grammar argument:

```python
import dynamicsyntax as ds

ds.load_grammar("ttr")  # or ds.load_grammar("2015-english-ttr")
p = ds.parse("a man arrives")
print(p.semantics)
p.vis()
```

Use a filesystem path to a grammar directory (as in the GUI “load folder” flow):

```python
import dynamicsyntax as ds
from pathlib import Path

ds.load_grammar(Path("/path/to/grammar-dir"))
p = ds.parse("go to the red box")
```

## Testing

See [testing.md](testing.md) for how to run pytest with **uv**, show per-test pass/fail output, and useful options (`-v`, `--lf`, subsets, and CI).

## Get Involved

Contributions (issues, PRs, [donations](https://buymeacoffee.com/incrementaliser)) are very welcome! Since this project has been translated using Cursor, there is a small chance errors have been introduced. I would be grateful if you could report them to me through the above channels. Although, it is important to mention that some of the most critical methods here have been verified via unit tests, so there is a high chance the migration from Java has been corrrect so far.
