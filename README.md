# MuJoCo Ant SAC — Windows, Linux, macOS

Cross-platform source project for training a simulated quadruped on Gymnasium **Ant-v5** using MuJoCo physics and Stable-Baselines3 **SAC**. Includes automatic local Python setup, a small training smoke test, longer experiments, checkpoint/replay-buffer save and resume, model evaluation, TensorBoard, live viewing, and optional MP4 recording.

**The repository does not contain a pretrained model or bundled third-party binaries.** Initial setup needs internet, sufficient free space, a compatible 64-bit system, and (for video/live visualization) working graphics drivers. The default **2,048-step smoke test checks the pipeline; it does not teach the robot to walk.** Training quality is not guaranteed by any fixed number of steps.

## Quick start

### Windows 10/11 (native PowerShell / CMD; WSL not required)

```powershell
.\run.cmd
.\run.cmd -Full -Resume
.\watch.cmd
```

To skip graphics on a remote host: `.\run.cmd -NoVideo -NoOpen`. A PowerShell-native alternative is `.\run.ps1`, provided your session permits running scripts. `run.cmd` invokes PowerShell in a separate process using a process-scoped execution policy; it does not change persistent settings.

### Linux/macOS

```bash
chmod +x run.sh setup.sh watch.sh test.sh
./run.sh
./run.sh --full --resume
./watch.sh
```

On a headless Linux server use `./run.sh --no-video --no-open`; when no display is detected, the default Linux runner skips video automatically. Optional `./run.sh --headless-video --resume --steps 1024` requires OSMesa/OpenGL system runtime libraries. On macOS the live-viewing launcher prefers MuJoCo's `mjpython` when installed.

## What one command does

1. Installs official uv to local `.tools/bin` if needed; creates local Python 3.11 `.venv` and installs `requirements.txt` packages. No WSL or Conda required; Linux graphics drivers/system libraries may require approved OS-level installation.
2. Runs a non-graphical MuJoCo Ant-v5 physics/control check (`src.check_env`).
3. Trains SAC for 2,048 environment steps by default (`src.train`), writing model/replay buffer and periodic checkpoints; logs TensorBoard metrics.
4. Evaluates the policy on three episodes and writes `logs/evaluation.json`; generates a PNG rewards chart when episodes are available.
5. Renders and optionally opens `videos/ant_demo.mp4` when a working graphics backend exists; use `--no-video`/`-NoVideo` to skip rendering.

The success message indicates a completed **software pipeline**, not a learned walking gait.

## Commands

| Action | Linux/macOS | Windows |
|---|---|---|
| First run | `./run.sh` | `.\run.cmd` |
| Train 1M additional steps | `./run.sh --full --resume` | `.\run.cmd -Full -Resume` |
| Train 100k more | `./run.sh --steps 100000 --resume` | `.\run.cmd -Steps 100000 -Resume` |
| Disable video | `./run.sh --no-video` | `.\run.cmd -NoVideo` |
| Live viewer | `./watch.sh` | `.\watch.cmd` |
| Tests | `./test.sh` | `.\test.cmd` |
| TensorBoard | `./.venv/bin/tensorboard --logdir logs` | `.\.venv\Scripts\tensorboard.exe --logdir logs` |

See [Commands](docs/COMMANDS.md), [Architecture](docs/ARCHITECTURE.md), and [Troubleshooting](docs/TROUBLESHOOTING.md). For a different viewing model, use `--model models/ant_sac_latest.zip` (POSIX) or `-Model models/ant_sac_latest.zip` (Windows). A first run with `--resume` or `-Resume` is invalid until the latest model exists.

## Project layout

```text
run.sh / run.ps1 / run.cmd          end-to-end platform launchers
setup.sh / setup.ps1 / setup.cmd    local Python and dependency setup
watch.sh / watch.ps1 / watch.cmd    live viewer
 test.sh / test.ps1 / test.cmd     unit and physics checks
requirements.txt / .python-version
src/common.py, check_env.py, train.py, evaluate.py, play.py, record.py, plot.py
tests/test_project.py
docs/ARCHITECTURE.md, COMMANDS.md, TROUBLESHOOTING.md
models/, logs/, videos/            generated artifacts (ignored by Git)
```

## Reproducibility and limits

Ant-v5 has eight continuous actuator actions and usually 105 observation values in its default configuration. Its task rewards forward locomotion, not navigation to arbitrary goal coordinates. SAC training uses CPU for portability. `--resume` loads the latest model plus replay buffer when available, but does not restore every random/environment/process state, so it is not bitwise identical to an uninterrupted run. Checkpoints, old evaluations, and stored buffers may take disk space. For independent experiments, archive previous models and logs first. This is a pure RL project, not an LLM/MCP or real-robot controller.

The setup scripts download the official uv installer from astral.sh when necessary. Review downloaded installers and follow your device's security policies; the scripts do not request elevation or change persistent OS security settings.

Official sources: [MuJoCo](https://mujoco.readthedocs.io/en/stable/python.html), [Gymnasium Ant](https://gymnasium.farama.org/environments/mujoco/ant/), [SB3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html), [uv](https://docs.astral.sh/uv/getting-started/installation/).

The project-specific code is an educational example; third-party libraries retain their own licenses.
