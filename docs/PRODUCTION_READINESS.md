# Production readiness and release gates

This project is an **experiment platform under active development**, not yet
a certified robot controller or an independently established RL leaderboard.
It should not be described as fully production-ready just because CI is green.

## Implemented and independently checkable

| Capability | Implementation | Objective gate |
| --- | --- | --- |
| Ant-v5 physics | Gymnasium + MuJoCo smoke checks | Finite observations/rewards, no headless display dependency |
| SAC lifecycle | Train, restore and evaluate | Linux/macOS/Windows integration CI |
| Artifact integrity | Immutable replay/model snapshot and SHA256 pointer | Corruption and crash-injection unit tests |
| Experiment separation | MUJOCO_RL_OUTPUT_ROOT and idempotent suite target budget | No cross-seed checkpoint/log contamination |
| Provenance | Experiment manifest, model SHA256, environment and config fingerprints | Abort on modified resume configuration |
| Benchmarking | Independent training seeds, equal-budget validation, bootstrap CI | Reject duplicated seeds, budget mismatch, protocol mismatch |
| Artifact audit | python -m src.verify | Verify saved model, replay, metadata and evaluated model |

Read the current Actions workflow before treating the gates as passed.
A successful CI smoke run establishes only software integration quality,
not sample efficiency or meaningful quadruped locomotion.

## Release blockers

**P0: experimental validity.** Publish a frozen Ant-v5 benchmark protocol
with at least five independent training seeds per algorithm, identical
training budgets, independent evaluation episodes, and confidence intervals.
Compare against RL Baselines3 Zoo baselines. Include forward displacement
and failure rate, not reward alone. Publish raw results, complete hardware
and library versions and reproducible commands. No scores are supplied here
until measurements actually exist.

**P0: restart semantics.** Persist/restore Python, NumPy, torch, environment,
callback and sampling states, or explicitly label a resumed run as non-bitwise
replicable. Current restart state is not bit-for-bit deterministic.

**P0: artifact trust.** SHA256 checksums are not signatures. Introduce
signed release manifests, immutable artifact storage and a trust policy
before accepting external models or replay-buffer pickle files.

**P1: reproducible dependencies.** Commit tested platform/architecture
specific lockfiles with hashes and verify dependency vulnerabilities and
SBOMs in a release workflow.

**P1: checkpoint lifecycle.** Implement and fault-test bounded retention,
atomic best-model/score pairing, disk-full failure injection and restart
recovery. Replaying snapshots consumes substantial storage.

**P1: benchmark coverage.** Compare SAC, PPO and TD3 with fixed evaluation
and budget; test on at least Ant-v5, HalfCheetah-v5 and Hopper-v5 before
claiming generalizable algorithm quality.

**P1: infrastructure.** Benchmark CPU/GPU throughput and variance;
run stress tests, cancellation tests, and capacity estimates for multi-seed
jobs under realistic storage and compute budgets.

**P2: community publication.** Choose a compatible open-source LICENSE
(the maintainer must authorize it), publish releases/changelogs, sample
videos with seed/config provenance, a documentation site, model cards and
issue triage guidelines. GitHub Trending is a community outcome rather
than a verifiable engineering acceptance criterion.

## What the quantitative evidence means

- A 260-step or 2,048-step SAC job is **only** a training pipeline smoke test.
- The mean of episode rewards across different seeds is not a substitute
  for a confidence interval across independent training runs.
- A 95% percentile bootstrap interval is an uncertainty summary, not a
  guarantee of coverage with very small sample counts.
- A model that earns survival reward without meaningful displacement should
  not be described as having learned effective walking.
- A simulation policy must not be assumed safe on real actuators.
