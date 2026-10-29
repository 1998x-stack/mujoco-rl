# Ant-v5 benchmark protocol — P0 quality evidence

Status: **implemented, automated smoke-tested; long-run learning results not yet generated**.
No scores from the 260-step CI smoke suite may be used to claim locomotion quality.

## Why episode return alone is insufficient

Gymnasium Ant-v5 includes a positive healthy/survival reward and action/contact
costs. A policy can survive and accumulate reward without learning a useful
forward gait. Therefore report reward together with signed x-displacement,
net x-velocity, episode duration, early termination and time-limit completion.

Reference: [official Gymnasium Ant-v5](https://gymnasium.farama.org/environments/mujoco/ant/).

## Frozen evaluation rules

| Item | Protocol |
|---|---|
| Environment | Gymnasium `Ant-v5`, default task configuration |
| Algorithm | Stable-Baselines3 SAC, repository training contract |
| Seeds | Exactly `0,1,2,3,4` in GitHub publication workflow (at least five in CLI) |
| Budget | At least `1,000,000` environment timesteps **per training seed** |
| Policy selection | Best deterministic training-time EvalCallback checkpoint; tracked by source hash |
| Evaluation | 10 complete independent episodes per seed |
| Evaluation seed series | `1000 + episode index`, identical across all five policies |
| Inference | `model.predict(..., deterministic=True)` |
| Report version | `evaluation.json` schema 3 |
| Progress proxy | Net displacement in +x >= 1 metre for an episode |
| Confidence interval | 95% percentile bootstrap over **per-training-seed means** |
| Build parity | Identical algorithm/config fingerprint, episode count, evaluation protocol, Gymnasium/MuJoCo/SB3 versions |
| Publication guard | `--publication` on aggregator or sweep |

The +1 m progress proxy is deliberately named **positive progress** rather than
`success`. It is not a proof of stable walking; inspect rendered trajectories,
reward decomposition and termination cases. Reward comparisons use the same task,
time limit and code/package versions. GPU vs CPU device is part of the shared
configuration fingerprint, so results are not silently mixed.

### Per-episode definitions

- `forward_displacement_x`: terminal x-position minus reset x-position, in metres.
- `mean_forward_speed_x`: net signed displacement divided by simulated elapsed seconds,
  **not** average absolute speed, and not a gait stability measure.
- `early_termination_rate`: fraction where `terminated and not truncated`.
  Ant-v5 termination should not automatically be called a "fall" without checking
  the environment termination reason.
- `time_limit_rate`: fraction with `truncated` true.
- `positive_progress_rate`: fraction of episodes with net +x >= 1 m.
- `mean_episode_length_steps`: mean actual episode length.

Each returned episode includes `steps`, `elapsed_simulation_seconds`,
`terminated` and `truncated`, so independent verifiers can recompute all
reported quantities. Aggregation refuses fabricated/inconsistent derived metrics.

## Running the complete benchmark

### Option 1: Manual GitHub Actions (five independent CI jobs)

Navigate to **Actions → Ant-v5 five-seed publication candidate → Run workflow**.
This triggers five independent, expensive CPU runs, each with a six-hour runner
limit. All five training jobs must succeed before aggregation; no partial
benchmark is marked as publishable.

Each seed uploads an evidence bundle with its evaluated policy ZIP, full
episode-level evaluation, training contract, resolved Python packages and
local integrity audit. The final job independently validates those bundles
and generates `benchmark.json`, `verification.json` and `SCORECARD.md`.

Artifacts expire according to retention settings (30 days per-seed, 90 days
for the aggregate); download/backup verified outputs for long-term research.
Bundles intentionally exclude pickle replay buffers, which are potentially
unsafe from untrusted sources and large. They allow evaluation reproducibility,
not exact training resumption.

### Option 2: Local sequential command

```bash
# Use project Python environment: Linux/macOS: .venv/bin/python
.venv/bin/python -m src.sweep --name ant-v5-paper-v1 \
  --seeds 0,1,2,3,4 --steps 1000000 --episodes 10 \
  --eval-seed 1000 --device cpu --publication

# Retry toward the same TOTAL training budget after a partial interruption.
.venv/bin/python -m src.sweep --name ant-v5-paper-v1 \
  --seeds 0,1,2,3,4 --steps 1000000 --episodes 10 \
  --eval-seed 1000 --device cpu --publication --resume

# Re-aggregate previously generated evaluation JSONs.
.venv/bin/python -m src.benchmark \
  runs/ant-v5-paper-v1/seed-*/logs/evaluation.json \
  --publication --min-seeds 5 --output runs/ant-v5-paper-v1/benchmark.json

.venv/bin/python -m src.scorecard runs/ant-v5-paper-v1/benchmark.json \
  --output runs/ant-v5-paper-v1/SCORECARD.md
```

Windows: use `.venv\\Scripts\\python.exe` and supply explicit five JSON
paths instead of depending on a Bash glob.

### How to interpret outcomes

Only a successfully verified report with `publication_protocol_verified=true`
meets the **minimum experimental protocol**. This does NOT mean the policy
learned walking, that the confidence interval is narrow, or that SAC beats
another algorithm. Examine reward, net displacement, velocity, termination and
video together. With just five seeds, confidence intervals may be wide.

Algorithm comparisons (PPO, TD3) remain a separate step: their own tuning
budgets, model-selection rules, action scaling and evaluation episodes must be
standardized before drawing head-to-head claims. Do not invent scores.

## Evidence and adversarial tests

The repository tests that the aggregator rejects repeated seeds, unequal
training budgets, fewer than 10 episodes, missing physical measurements,
inconsistent reward summaries, falsified displacements/speeds, mismatched
MuJoCo/Gymnasium/SB3 versions and unverified model ZIP hashes.

`src.verify` audits same-run model/replay snapshot and evaluation claims;
`src.evidence` verifies the portable final-policy ZIP and audit/manifest.
These checks detect accidental corruption and inconsistent bookkeeping, but
not a malicious publisher with the ability to rewrite *all* evidence.
Cryptographic signatures/attestations remain a production-release requirement.

This benchmark implementation follows the independent-seed statistical
reporting principles explained by
[google-research/rliable](https://github.com/google-research/rliable)
and uses the official [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo)
as an external reference for future hyperparameter and algorithm comparisons.
