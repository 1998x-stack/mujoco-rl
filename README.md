# MuJoCo Ant-v5 SAC — Windows / Linux / macOS

Cross-platform Python project for training a simulated quadruped with Gymnasium, MuJoCo and Stable-Baselines3 SAC. Includes automated environment setup, physics checks, training, evaluation, reward plots, checkpoint recovery, interactive playback, and optional MP4 output. **No pretrained walking policy is bundled:** the default 2,048-step run is a software pipeline smoke test, not a guarantee of learned locomotion.

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

Each completed training invocation writes `models/snapshots/snapshot-*/model.zip` and `replay_buffer.pkl`, calculates SHA-256 checksums and atomically publishes `models/latest.json` only after both files have been saved. A failed save leaves the previous pointer in place. `--resume` validates the model and replay buffer pair; it refuses to silently resume from an incomplete pair. Historical `models/ant_sac_latest.zip` and `models/ant_sac_replay_buffer.pkl` are accepted for migration when no new snapshot is present. Older immutable snapshots stay on disk until explicitly cleaned up, so longer experiments can consume substantial disk space.

`models/best/best_model.zip` is the best *evaluated* model; `models/best/score.json` persists its selection score across resume operations. `src.evaluate`, `src.play` and `src.record` prefer the best model and fall back to the latest snapshot. To use an explicit model instead, pass `--model /path/to/model.zip` to those Python modules.

These snapshots protect against a partial application-level save; they do not guarantee bitwise-identical continuation after a process restart, nor durability against disk failure or power loss. Python package requirements specify ranges rather than an immutable per-platform lockfile. Record package versions, seeds, hardware and evaluation protocol for comparative experiments.

## Project layout

```text
run.sh, run.ps1, run.cmd        end-to-end OS-native launchers
setup.sh, setup.ps1, setup.cmd  project-local Python + dependencies
watch.sh, watch.ps1, watch.cmd  interactive replay
test.sh, test.ps1, test.cmd     unit tests + physics check
requirements.txt              Python dependencies
src/checkpoint.py             integrity-checked snapshots + best score
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
