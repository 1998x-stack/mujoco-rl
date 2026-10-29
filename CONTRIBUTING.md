# Contributing

## Engineering baseline

- Base changes on `main` and open a reviewable pull request.
- Run `python -m unittest discover -s tests -v`,
  `python -m compileall -q src tests`, and
  `ruff check src tests` before requesting review.
- A physics smoke test is `python -m src.check_env --steps 32`.
- For any SAC lifecycle change, run short train, resume, evaluate and
  `--fresh` archive tests on the supported platforms.
- Never merge benchmark claims based only on 260 or 2048 steps.
  Use multiple independent training seeds, fixed evaluation seeds,
  identical environment and policy configurations, and report uncertainty.
- Do not silently overwrite historical models, logs, or benchmark outputs.
- Avoid external data downloads in unit tests. Mark expensive or graphics-
  dependent integration testing separately.
- Record limitations, failure modes, environment versions and hardware.

## Changes that require special attention

Checkpoint format, trained policy deserialization, rewards/termination,
action scaling, random seed management, and evaluation protocols
affect correctness of published research results. Accompany changes
with regression tests and an explicit migration note.

## Legal

No license has been assigned by this change. The repository owner must
choose and publish an appropriate license before describing the project
as permissively reusable.
