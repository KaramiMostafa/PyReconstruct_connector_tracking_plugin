# Contributing

Use Python 3.11, install both repositories in editable mode, and run the test
suite before committing:

```bash
python -m unittest discover -v tests
```

## Commit format

Every customized commit must follow
[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):

```text
type(scope): description
```

Use lowercase `type`/`scope`, add `!` before `:` for a breaking change, and
include a `BREAKING CHANGE: ...` footer when migration instructions are
needed. Common types are `feat`, `fix`, `docs`, `test`, `refactor`, `perf`,
`build`, `ci`, `chore`, `style`, and `revert`.

Enable the local hook once:

```bash
git config core.hooksPath .githooks
```

Validate the whole customized branch manually with:

```bash
python scripts/check_conventional_commits.py --range main..HEAD
```

GitHub Actions enforces the same rule. Prefer squash merging with a
Conventional Commit title.
