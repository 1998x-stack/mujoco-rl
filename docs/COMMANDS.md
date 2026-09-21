# Commands for macOS, Linux and Windows

Run from the extracted project folder. These commands install packages on the first run; use a trusted internet connection.

## macOS / Linux (Terminal)

```bash
chmod +x run.sh setup.sh watch.sh test.sh
./run.sh                                  # 2048-step end-to-end smoke test
./run.sh --full --resume                  # 1M additional steps
./run.sh --steps 100000 --resume          # 100k more steps
./run.sh --no-video --no-open             # no graphics required
./watch.sh                                # local live 3D viewer
./test.sh                                 # unit + physics checks
./.venv/bin/tensorboard --logdir logs     # monitor training
./.venv/bin/python -m src.evaluate --episodes 10
./.venv/bin/python -m src.record --frames 500
```

Linux only: `./run.sh --headless-video --resume --steps 1024` tries software OSMesa rendering; install corresponding OS packages first. Under macOS, `./watch.sh` uses `mjpython` to handle platform viewer requirements.

## Windows 10/11 (PowerShell / CMD)

```powershell
.\run.cmd                                  # 2048-step end-to-end smoke test
.\run.cmd -Full -Resume                    # 1M more steps
.\run.cmd -Steps 100000 -Resume            # 100k more steps
.\run.cmd -NoVideo -NoOpen                 # no graphics required
.\watch.cmd                                # local live viewer
.\test.cmd                                 # unit + physics checks
.\.venv\Scripts\tensorboard.exe --logdir logs
.\.venv\Scripts\python.exe -m src.evaluate --episodes 10
.\.venv\Scripts\python.exe -m src.record --frames 500
```

The `*.cmd` wrappers use PowerShell in a process-scoped execution-policy override; no persistent OS settings need changing. On systems where scripts are blocked by organizational policy, contact the device administrator rather than weakening security controls.

## Manage models and artifacts

- `models/ant_sac_latest.zip` — latest SAC model; `models/ant_sac_replay_buffer.pkl` — experience needed for resume.
- `models/best/best_model.zip` — highest mean periodic evaluation reward so far; available only after a completed callback evaluation.
- `models/checkpoints/` — periodic model and buffer checkpoints. Old buffers may occupy substantial disk space.
- `logs/evaluation.json` — evaluation metrics, `logs/rewards.png` — plot if enough episodes completed.
- `videos/ant_demo.mp4` — optional rendered preview.

From scratch without previous run's weights: use `./run.sh --full` or `.\run.cmd -Full` after archiving your older `models/` and `logs/` if you want to compare independent experiments. `--full` and `-Full` request a million *additional* steps for that invocation, not a wall-clock duration or guaranteed gait quality.
