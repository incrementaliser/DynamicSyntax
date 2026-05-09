# dynamicsyntax

`dynamicsyntax` is the python implementation of **[DyLan](https://github.com/Dynamics-of-Language)** the Dynamic Syntax parser. It has been (vibe) translated from Java to Python by [@incrementaliser](https://github.com/incrementaliser), and is still under development.

## Installation

You need [uv](https://docs.astral.sh/uv/) and a Python that satisfies `>=3.11`.

1. Create a virtual environment with a fixed Python version (run this in the directory where you want `.venv`, e.g. the repo root for a clone):
  ```bash
   uv venv --python 3.11 # or 3.12 or 3.13
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

## Examples

For some code examples of how to use the parser in action, check out this notebook: [Open In Colab](https://colab.research.google.com/drive/1ofpCLwLOE88AvbtR3gh5uisaYkpWGNP9?usp=sharing)

### Quickstart

```python
import dynamicsyntax as ds

p = ds.parse("a man knows you", "ttr")
print(p.semantics)  # TTR semantics based on the default grammar/lexicon
print(p.vis())      # Visualise the parse tree
```

## Get Involved

Contributions (issues, PRs, [donations (spent on coffee/LLM credits!)](https://buymeacoffee.com/incrementaliser)) are very welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

**Note**
Since this project has been translated using Cursor and tested only on Windows, there is a small chance errors exist. I would be grateful if you could report them to me through the above channels. Although, it is important to mention that some of the most critical methods here have been verified via unit tests, so there is a high chance the migration from Java has been correct so far.