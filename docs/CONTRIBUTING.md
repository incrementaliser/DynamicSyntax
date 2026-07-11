# Contributing

Here are some guidelines for contributing to this project.

## Commit style

This project uses [Conventional Commits](https://www.conventionalcommits.org).

Format: `type(scope): short description` — e.g. `fix(parser): handle empty token list`.

Types: feat · fix · docs · refactor · perf · test · chore · style

Suggested scopes: `parser` · `lexicon` · `learner` · `generator` · `ttr` · `gui` · `tests` · `ci`

Use imperative mood in the description: "add X", not "adds X" or "added X".

## Local commit hook (optional)

To validate commit messages before they are created, point Git at the tracked hooks directory from the repo root:

```bash
git config core.hooksPath scripts/git-hooks
```

On Unix, ensure the hook is executable (`chmod +x scripts/git-hooks/commit-msg`). Git for Windows runs this hook via `sh` when using `core.hooksPath`.

## Releasing

Releases are triggered automatically by pushing a version tag, for example:

```bash
git tag v0.2.2 && git push origin v0.2.2
```

GitHub Actions builds the package, generates the changelog for the GitHub Release, and publishes to PyPI.

The first time you publish from CI, configure **Trusted Publishing** on PyPI so no API token is required: PyPI → project **dynamicsyntax** → Publishing → Add a new publisher → GitHub Actions → select this repository and workflow **`release.yml`**.
