# MuJoCo Ant-v5 SAC — Windows / Linux / macOS

Cross-platform Python project for training a simulated quadruped with Gymnasium, MuJoCo and Stable-Baselines3 SAC. Includes automated environment setup, physics checks, training, evaluation, reward plots, checkpoint recovery, interactive playback, and optional MP4 output. **No pretrained walking policy is bundled:** the default 2,048-step run is a software pipeline smoke test, not a guarantee of learned locomotion.

[![CI](https://github.com/1998x-stack/mujoco-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/1998x-stack/mujoco-rl/actions/workflows/ci.yml)

## Reproducible multi-seed benchmark

A 2,048-step training run is only a software smoke test. A serious evaluation uses
independent training seeds, identical configurations and evaluation seeds, and
reports uncertainty over **training seeds**, not correlated episodes.

After installing project dependencies using `./setup.sh` (macOS/Linux) or
`.\\setup.cmd` (Windows):

```bash
# Isolated experiment directories: runs/ant-sac-baseline/seed-0/...seed-4
python -m src.sweep --name ant-sac-baseline --seeds 0,1,2,3,4 --steps 1000000 --episodes 10

# After an interruption: idempotently reach the requested cumulative budget
python -m src.sweep --name ant-sac-baseline --seeds 0,1,2,3,4 --steps 1000000 --resume

# No additional training; reevaluate saved snapshots with common episode seeds
python -m src.sweep --name ant-sac-baseline --seeds 0,1,2,3,4 --evaluate-only

# Offline report verification and seed-level bootstrap statistics
python -m src.benchmark runs/ant-sac-baseline/seed-*/logs/evaluation.json --output runs/ant-sac-baseline/summary.json
```

On Windows, use `.\\.venv\\Scripts\\python.exe` in place of `python` if your
system interpreter is not the project venv.

Each seed writes independent `models/`, `logs/`, and `videos/` beneath its own
output root using `MUJOCO_RL_OUTPUT_ROOT`. New runs refuse to overwrite
existing outputs. `--resume` checks immutable experiment configuration
(including algorithm, hyperparameters, seed and device); legacy checkpoints
without provenance warn that the original configuration cannot be verified.
The `models/experiment.json` manifest captures a training-contract fingerprint,
Python/package versions, platform and Git SHA. Evaluations include training
seed, episode scores and an SHA-256 digest of the evaluated model. Benchmark
aggregation requires at least two independent seeds, the same evaluation
protocol and a matching seed-independent configuration fingerprint.

`benchmark.json` contains per-seed scores, across-seed mean/standard
deviation, and a deterministic percentile bootstrap 95% confidence interval
over training seed means. Five seeds provide a useful starting point, not a
guarantee of tight confidence intervals or algorithm superiority. This
repository does not currently include validated locomotion scores, a
multi-algorithm comparison or signed/pinned releases.

## Engineering quality and security

CI verifies source compilation, Ruff static checks, unit/fault-injection
tests and physics/SAC lifecycle tests across Linux, macOS and Windows; a
Linux integration job also exercises independent seed isolation and
benchmark aggregation. Check the latest Actions results before claiming
that any specific revision passed.

Validate the resulting artifacts using \`python -m src.verify --require-evaluation\` in each experiment's output root. The multi-seed suite resumes to a **target total** training budget; unlike \`src.train --steps\`, it does not repeat completed steps on retry. The aggregator refuses unequal budgets.\n\nSee [Production readiness](docs/PRODUCTION_READINESS.md), [Contributing](CONTRIBUTING.md) and [Security](SECURITY.md).
**Load only trusted model and replay-buffer artifacts:** a checksum checks
integrity, not authenticity. This is simulated locomotion, not a
safety-certified real-world robot controller.

## Getting started

Download/clone the repository to a writable local directory. First run requires a compatible 64-bit computer, an internet connection to download Python and packages, and sufficient free disk space. Python 3.11 is provisioned locally by `uv` when necessary; no Conda, Unity, or WSL is required.

**macOS / Linux** (Terminal):

```bash
git clone https://github.com/1998x-stack/mujoco-rl.git
cd mujoco-rl
chmod +x run.sh setup.sh watch.sh test.sh
./run.sh
```

**Windows 10/11** (PowerShell or CMD): clone/extract the repository, open a terminal in its root and run:

```powershell
.\run.cmd
```

On a Linux server without a graphical desktop, the runner automatically skips video. On macOS remote SSH, video is disabled by default. On Windows, use `.\run.cmd -NoVideo` when a usable OpenGL context is unavailable. The optional MP4 recording reports a warning rather than failing the already completed training pipeline.

## Training and safe reruns

| Task | macOS / Linux | Windows |
|---|---|---|
| First run: train 2,048 steps | `./run.sh` | `.\run.cmd` |
| Resume with 1,000,000 additional steps | `./run.sh --full --resume` | `.\run.cmd -Full -Resume` |
| Resume with 100,000 additional steps | `./run.sh --steps 100000 --resume` | `.\run.cmd -Steps 100000 -Resume` |
| Start a new experiment, archiving active outputs | `./run.sh --fresh` | `.\run.cmd -Fresh` |
| Disable optional MP4 | `./run.sh --no-video` | `.\run.cmd -NoVideo` |
| Open live simulation | `./watch.sh` | `.\watch.cmd` |
| Unit tests and physics check | `./test.sh` | `.\test.cmd` |

**Existing training artifacts cannot be silently overwritten.** After the first run, choose `--resume`/`-Resume` to continue, or `--fresh`/`-Fresh` to archive existing active artifacts and begin a separate experiment. `--full` does not implicitly resume. Both launchers also offer `--help` / `-Help`.

## Checkpoint storage and reproducibility

Each completed training invocation writes `models/snapshots/snapshot-*/model.zip` and `replay_buffer.pkl`, calculates SHA-256 checksums and atomically publishes `models/latest.json` only after both files have been saved. A failed save leaves the previous pointer in place. `--resume` validates the model and replay buffer pair; it refuses to silently resume from an incomplete pair. Historical `models/ant_sac_latest.zip` and `models/ant_sac_replay_buffer.pkl` are accepted for migration when no new snapshot is present. Older immutable snapshots stay on disk until explicitly cleaned up, so longer experiments can consume substantial disk space. A fresh run archives **all** active contents of `models/` (except historical archives), `logs/`, and `videos/` under `models/archive/<UTC timestamp>-<id>/`, including old snapshots and periodic checkpoints. Empty scaffolding and `.gitkeep` files do not count as previous experiments.

`models/best/best_model.zip` is the best *evaluated* model; `models/best/score.json` persists its selection score across resume operations. Monitor episode CSV files append on resume instead of being truncated, and each training invocation keeps its own `logs/eval_history/<run-id>/evaluations.npz`. `src.evaluate`, `src.play` and `src.record` prefer the best model and fall back to the latest snapshot. To use an explicit model instead, pass `--model /path/to/model.zip` to those Python modules.

These snapshots protect against a partial application-level save; they do not guarantee bitwise-identical continuation after a process restart, nor durability against disk failure or power loss. Python package requirements specify ranges rather than an immutable per-platform lockfile. Record package versions, seeds, hardware and evaluation protocol for comparative experiments.

## Project layout

```text
run.sh, run.ps1, run.cmd        end-to-end OS-native launchers
setup.sh, setup.ps1, setup.cmd  project-local Python + dependencies
watch.sh, watch.ps1, watch.cmd  interactive replay
test.sh, test.ps1, test.cmd     unit tests + physics check
requirements.txt              Python dependencies
src/checkpoint.py             integrity-checked snapshots + best score
src/provenance.py             immutable training contracts + environment manifest
src/sweep.py, src/benchmark.py independent multi-seed runs + statistical reports
src/train.py                  SAC training, evaluation callbacks, resume
src/check_env.py              non-graphical physics smoke test
src/evaluate.py               independent evaluation + JSON summary
src/play.py, src/record.py    live viewer / optional MP4
src/common.py, src/plot.py    shared configuration / reward plot
tests/test_project.py         dependency-free contract/failure tests
docs/REVIEW.md                engineering review and validation matrix
.github/workflows/ci.yml      Windows/Linux/macOS checks
models/, logs/, videos/        generated run artifacts (gitignored)
```

Detailed documentation: [Architecture](docs/ARCHITECTURE.md), [Commands](docs/COMMANDS.md), [Troubleshooting](docs/TROUBLESHOOTING.md), [Engineering review](docs/REVIEW.md). The physics environment is Gymnasium `Ant-v5`, using eight continuous actuator controls; the robot is entirely simulated, with no real hardware, API keys or LLM provider required.
