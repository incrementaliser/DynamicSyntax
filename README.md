# dynamicsyntax

`dynamicsyntax` is the python implementation of the **[DyLan](https://github.com/Dynamics-of-Language)** the Dynamic Syntax parser. It has been (vibe) translated from Java to Python by @incrementaliser using Cursor, and is still under development.

## Installation

You can run the following commands to install the package:

```bash
uv sync
uv pip install dynamicsyntax
```

or the older way:

```bash
pip install dynamicsyntax
```

## Example

```python
import dynamicsyntax as ds

semantics = ds.parse("a man arrives", "ttr")
print(semantics)
```

The above loads the 2015-english-ttr grammar and parses the utterance "a man arrives".

Contributions (via PRs) are very welcome!
