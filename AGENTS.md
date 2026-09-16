# Working on csiphon

Conventions for anyone (human or agent) changing this library.

## Mechanical baseline

Run `./check_style.sh` before you consider a change done. It runs ruff, mypy,
pylint, and pytest, and stops at the first failure. Fix what it reports; silence
a check only where truly required, and don't add ignore rules without asking.

CI runs everything the script does except pylint, so a local run is the only
place pylint catches things.

- **ruff** (`E, F, I, UP, B, SIM, RUF, ANN`): import sorting and full type
  annotations are mandatory in `src/`; tests are exempt from arg annotations.
- **mypy `--strict`** type-checks `src/`. A local pass is not conclusive if your
  environment differs from CI's (dependency versions especially).
- Config lives in [pyproject.toml](pyproject.toml). Don't change configs or add
  tools without being asked.

Some things the tools don't catch:

- Prefer aligned comments and colons. Isolate blocks (e.g. in dataclasses) with
  `# fmt: off` and `# fmt: on`.
- Add `from __future__ import annotations` only when a file needs forward
  references. Don't add it by default. Actually this goes for any import.

## Simplicity

Prefer the simplest solution that solves the problem. Reach for the library you
already depend on before writing the same thing by hand. Add a helper, a wrapper,
or an abstraction when a caller needs it, not in anticipation. Delete code that
no longer earns its place.

Importantly, don't be smart when it's not required. Code must be readable before
anything else. Deep indentations, complex nested calls, sophisticated axis contractions
and other odd contraptions are for code-golfing and not for this library. Such
complexity is only allowed when performance demands it, and must then be commented
properly (see below).

## Naming and helpers

Helper functions can be great for reuse. However, a single line operation does not
need a helper. Use them where it truly makes sense.

Everything is written in snake_case and abbreviations only used sparingly where
they follow an established convention. For example, "pca" is fine. However, in
general prefer extensive and descriptive names. Code is prose.

Functions ideally have one job, and it's in the name, and a brief docstring.

## Generality

Write an operation as generally as the problem allows. Act on the parts you
care about and leave the rest untouched, rather than assuming a fixed shape,
rank, or axis count you don't actually depend on. Constrain the input only where
the operation genuinely requires it, and enforce that constraint explicitly.

## Don't duplicate

If a new thing is an existing thing plus a trivial variation, parameterize or
extend the existing one instead of copying it. Before writing something, check
whether it already exists in a form you can reuse.

## Steps

Adding or changing a step? Read
[src/csiphon/steps/README.md](src/csiphon/steps/README.md) first. It covers the
file template, the spec contract, input guards, streaming, and registering the
step so the tests find it.

## Types

- Use the array aliases `SignalArray` (`RealArray | ComplexArray`) and the
  converters `as_real_array`, `as_signal_array`, `as_complex_array`.
- Narrow a type at the point where the code depends on the narrower type, and
  note why when it isn't obvious.

## Comments

Comments should explain why code is written the way it is. A non-obvious trick,
the need for an operation, explanations of semantics (e.g. axes). Comments can
also just summarize the "combined action" of a few lines of code for structure.
Don't overdo it though by adding comments that add nothing.

Prefer comments _before_ a statement or LOC, not behind it. Example:

```python
# NOTE: Array shapes are (time, antenna, subcarrier)
some_result = np.array(...)
```

## Breathing room

Give code room to read: a blank line after a function or method docstring, and
blank lines between logical groups.

## Tests state requirements

A test pins down one expectation and reads as its spec.

- State the expectation in the docstring, one requirement per test.
- Cover positive and negative behavior. For valid input, assert what it
  produces: axes, types, shapes, metadata, value semantics, the full contract.
  For invalid input, assert it is rejected with the right error and message.
- Check that parameters do what they claim, and that input maps to output as
  specified.
- Check batch/stream equivalence wherever a step claims it.
- When a test guards against a real failure, say so, so a later reader keeps it.

## Writing

- No em-dashes. Use a comma, a period, parentheses, or a colon.
- State the point directly. A contrast is fine when it instructs ("use X instead
  of Y"). Avoid the empty contrastive punchline where the negation carries no
  information, like "this isn't a bug, it's a feature".
- Don't argue against a position no one holds.
- Plain English. Use jargon only where it genuinely helps.
- Keep it short. Say the thing and stop.
- Cut redundancy. Drop a sentence that adds nothing, don't restate a point, and
  don't summarize after explaining. A codebase is not a medium.com article.
- No metawriting: don't announce what the next sentence will do.
- Write general rules, not anecdotes. Don't anchor a point to a past incident or
  "the most common mistake here".
- No marketing language or emphatic filler: skip "powerful", "robust",
  "seamless", bold IMPORTANT, ALL-CAPS, and exclamation marks. A natural, low-key
  joke is welcome; hype is not.
- Use fillers only where they help a sentence read. Cut the verbose ones ("it is
  worth noting that", "in order to").
- Name the subject (this is what active voice buys you). Use "we" or "csiphon"
  for what the code does, "you" for the reader or implementer, and keep one of
  them per passage.
- Don't let docstrings grow longer than the functions they describe. Be brief.

## Process

- Don't push or publish on the maintainer's behalf. Commits, pushes, and
  releases are their action. The package is on PyPI
  (https://pypi.org/project/csiphon/); publishing happens by cutting a GitHub
  Release (Trusted Publishing), so don't `twine upload` or push tags yourself.
- Record user-facing changes in [CHANGELOG.md](CHANGELOG.md) (Keep a Changelog
  format) and bump the version in `pyproject.toml` per semantic versioning.
- Be concise in answers. Match the length to the question.
- Don't break an existing .venv with uv or poetry.
