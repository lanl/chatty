# Contributing to chatty

Thanks for your interest in improving `chatty`. This file covers what we ask of
every pull request.

## Certificate of origin

Contributions are accepted under the project's [MIT license](LICENSE), the same
terms chatty is distributed under.

Please sign off on your commits, certifying that you wrote the contribution or
otherwise have the right to submit it under that license — the
[Developer Certificate of Origin](https://developercertificate.org). Add `-s`
when you commit:

```sh
git commit -s -m "your message"
```

which appends a `Signed-off-by:` line using the name and email from your
`git config`. To sign off commits already on a branch:

```sh
git rebase --signoff main
git push --force-with-lease
```

A pull request whose commits are not signed off cannot be merged.

## Development setup

`chatty` uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management, and requires Python ≥ 3.12.

```bash
git clone https://github.com/lanl/chatty.git
cd chatty
uv sync --dev                     # add --extra litkit or --extra rag for RAG work
uv run pre-commit install
```

## Running tests

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

The suite must not reach a real LLM endpoint. Mock HTTP with `respx`, and use the
fixtures in `tests/conftest.py`, which also keep your own config out of the
tests. UI changes may need the Textual snapshots in `tests/__snapshots__/`
regenerated (`uv run pytest --snapshot-update`). Look at the snapshot report
before you commit regenerated snapshots.

chatty has no CI yet, so say in the PR what you ran and on which platform. If
some tests fail on your platform before your change too (Windows is
[untested](README.md#windows-support-experimental)), name them.

## Fixing bugs

- **Reference the issue.** Put `Fixes #N` in the PR body. If there is no issue for
  the bug, describe it and how to reproduce it.
- **A test that fails without the fix.** Every bug fix comes with a regression
  test that fails on `main` and passes on your branch. Test the behaviour the user
  sees, not the line that changed.
- **Fix review findings in the PR.** A finding from review that is in code the PR
  touches, or is the same bug in a neighbouring path, is fixed in that PR, not
  filed as a new issue. A finding that is unrelated or needs a design decision
  goes in a new issue.
- **Keep it focused.** One issue, or one family of issues with a shared root
  cause, per PR. Do not reformat or refactor code the fix does not need.

## Before opening a PR

- Match the surrounding style. `ruff` and `mypy --strict` must pass.
- Add or update tests for behaviour changes.
- Update `README.md` or `chatty.toml.example` when you add or change a config
  option or command.
- **Do not edit [`CHANGELOG.md`](CHANGELOG.md).** Write a one- or two-sentence
  changelog entry in the PR body instead. The maintainer adds it at release time.
- Pull requests written with AI coding agents are welcome. You are responsible
  for them: you should understand the change, you should have run the tests, and
  you should be able to answer review questions about it.

## Credit

Everyone whose code is merged is listed in [`CONTRIBUTORS.md`](CONTRIBUTORS.md),
by name with GitHub handle — `Ada Lovelace (@alovelace)`. You are added when your
first contribution merges. We try to get your name right from what you have
already made public: the name on your commits, your GitHub profile, and how you
signed the pull request.

That is a best effort, and it is your name. If you want to be listed
differently, say so on your pull request or open an issue, and we will change
it. You can ask for another form of your name, your handle only, a different
handle, or no listing at all.
