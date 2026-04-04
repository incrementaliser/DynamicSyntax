# dynamicsyntax

Python 3.13 port of the **DyLan** Dynamic Syntax parser core (Java `qmul.ds`). Installable import package: **`dylan`**.

## Install (development)

```bash
cd dynamicsyntax
uv sync
uv run pytest
```

Optional NLP tokenizers:

```bash
uv sync --extra nlp
```

## Scope

- Working **parser** path: grammar/lexicon resource loading, `InteractiveContextParser`, `parseUtterance`.
- **Not** ported: probabilistic generators, `Feature`, learner GUI stack.

## License

DyLan upstream license applies to ported logic; see `LICENSE` if copied from the Java project.
