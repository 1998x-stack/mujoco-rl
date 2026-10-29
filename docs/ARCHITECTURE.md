# Architecture and training controls

## End-to-end flow

```text
Linux/macOS run.sh          Windows run.cmd -> run.ps1
          \                    /
              setup.sh / setup.ps1
                     |
       official uv + managed Python 3.11 + dependencies
                     |
          src.check_env (Gymnasium Ant-v5)
                     |
      src.train (Stable-Baselines3 SAC, CPU)
          |                  |
        Gymnasium          SAC policy + critics
          |                  |
        MuJoCo physics <-> observations/actions/rewards
                     |
    immutable latest snapshot + replay buffer + periodic checkpoints
                     |
             src.evaluate -> JSON
             src.plot     -> PNG if training episodes
             src.record   -> MP4 (graphics support needed)
             src.play     -> live 3D (separate viewer script)
```

## Platform launchers

`run.sh` runs in Bash on Linux/macOS. `run.ps1` runs on native Windows (PowerShell 5.1+), and `run.cmd` forwards arguments to it from either CMD or PowerShell without requiring users to change their persistent execution policy. All paths are anchored to the project directory; each OS gets its own `.venv` layout (`bin/python` on POSIX, `Scripts/python.exe` on Windows). The scripts do not assume WSL or Xcode/Unity.

For macOS live playback, `watch.sh` prefers MuJoCo's `mjpython` launcher to satisfy its native viewer main-thread requirement. Linux headless training is supported; OSMesa software MP4 is optional and depends on additional system libraries. Windows graphics need a working local OpenGL desktop context.

## Reinforcement learning details

The existing Gymnasium `Ant-v5` task provides a default 105-element observation, eight continuous actuator actions, forward-locomotion reward, and episode-reset/termination mechanics. SAC uses replay-buffer sampling and neural networks. `src.train` has an independent monitoring/evaluation environment, writes checkpoints, and saves the latest network state plus its replay buffer on normal completion and Python-handled interruption.

The short 2048-step default validates the stack; the 1M-step mode is an experiment, not a guarantee of gait quality. CPU training is selected deliberately to avoid assuming CUDA on Windows/Linux or Metal on macOS.

## Resume and reproducibility

Both `--resume` and `-Resume` require a matching SAC model and replay buffer; an incomplete pair is rejected. New snapshots are immutable and published through an integrity-checked latest pointer. `--fresh` / `-Fresh` archives old active models, snapshots, checkpoint directories, logs and videos beneath `models/archive/`, without moving previous archives. Monitor episode CSVs append on resume; callback evaluation histories are stored in distinct per-invocation directories. Restarting a process will not in general exactly reproduce an uninterrupted run, due to RNG/environment state and evaluation timing. Hard kills/power loss may preempt the `finally` save; check periodic checkpoints.

The `requirements.txt` supplies compatible version ranges, not an exact lockfile. For comparative experiments record the installed package versions, Python version, platform, random seeds, and any modifications to training configuration.

An LLM planner or MCP interface is out of scope: this project demonstrates a standalone RL-powered quadruped policy on a virtual robot, with no real robot hardware or external API keys.

## Reproducible benchmark architecture (v2)

```text
src.sweep: validated seed plan / target training budget
     |
     +-- seed-0 subprocess ---> src.train ---> src.provenance experiment.json
     |                                |-- immutable model/replay checkpoints
     |                                +-- evaluation/monitor history
     |                       ---> src.evaluate ---> evaluation.json schema 2
     |                       ---> src.verify   ---> checkpoint/eval audit
     +-- seed-1 subprocess ---> ... independent directory ...
     +-- seed-N subprocess ---> ...
     |
     +--> src.benchmark
             |-- verify seed uniqueness, config, episodes and training budget
             |-- bootstrap across independent per-seed mean scores
             +-- benchmark.json / CI integrity checks
```

Paths are rooted in `MUJOCO_RL_OUTPUT_ROOT` if explicitly provided, or the
repository root for backward compatibility. The multi-seed driver sets this
environment variable per training/evaluation subprocess and never uses shell
interpolation. A suite resuming to a target total reads snapshot metadata
before launching any training; `src.train` then validates the model/buffer
SHA-256 pair and immutable training contract.

This is reproducibility of **configuration, seed provenance, and comparable
experiment budgets**, not bit-exact replay after restart. That would additionally
need RNG, environment, callback and sampler state. Any evaluation's chosen
best model may have been saved at a prior time within its verified training
budget; the budget measures experimental resources, not the selected model's
last gradient step.
