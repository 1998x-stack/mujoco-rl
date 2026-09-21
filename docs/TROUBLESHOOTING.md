# Troubleshooting (Windows / Linux / macOS)

## First-run prerequisites

Internet access and sufficient free space are required for uv, Python 3.11, PyTorch, MuJoCo, and related packages. Use a compatible 64-bit OS and a supported architecture. The repository includes project source, **not** all installers, runtime binaries, or a trained model. Fresh OS images may be missing graphics drivers or system OpenGL libraries. A real graphics desktop is needed for ordinary video/viewer modes.

`setup.sh` / `setup.ps1` download the official uv installer to a temporary file when uv is absent. They do not request `sudo`, administrator privilege, or disable OS security prompts. Review the script and installer provenance before running if required.

## Windows: `run.ps1 cannot be loaded because running scripts is disabled`

Use the included **`run.cmd`** from PowerShell or Command Prompt (or `watch.cmd` / `setup.cmd`), which invokes the supplied script in a fresh process without permanently changing your settings. If an organizational policy forbids scripts or your IT administrator blocks downloads, do not bypass that policy: request an approved development environment.

## Windows: PowerShell not found or native dependency / DLL error

Normal Windows 10/11 installations include `powershell.exe`. For unusual stripped-down systems, ask your administrator to provision PowerShell. Install Windows system updates and any OS-approved Microsoft Visual C++ runtime that your Python/PyTorch wheel explicitly requires. Windows uses native Python/MuJoCo; no WSL is needed. Run `.\run.cmd -NoVideo` to separate core environment issues from display issues.

## macOS: `Permission denied: ./run.sh`

```bash
chmod +x run.sh setup.sh watch.sh test.sh
./run.sh
```

If macOS asks for Apple Command Line Tools or security approval, follow the official dialog if you trust the downloaded project. Do not disable Gatekeeper or apply blanket quarantine-removal commands. Older Intel Macs/macOS releases may not have compatible current PyTorch wheels.

## Linux: `curl: command not found` or missing shared library

Install `curl` and standard TLS/CA certificates through your distribution's approved package manager. For GUI and offscreen rendering, ensure that your graphics driver and appropriate GLFW/OpenGL runtime libraries are installed. Typical Ubuntu/Debian systems might need a compatible `libglfw3`, `libgl1` and `libosmesa6` runtime depending on your setup; exact package names and requirements vary by distribution. The launcher intentionally does not perform unattended privileged OS package installation.

To test simulation and training without a graphics stack, use `./run.sh --no-video`. If Python crashes **on import** due to a missing library, install the library reported by the error before trying again.

## Linux headless video (SSH / CI)

The default `./run.sh` detects the absence of `DISPLAY` and `WAYLAND_DISPLAY` and skips video, while still training and evaluating. To request headless CPU video, first install the OSMesa/OpenGL libraries for your distribution, then:

```bash
./run.sh --headless-video --resume --steps 1024
```

This sets `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa` for the child Python process. If OSMesa is not installed, rendering may fail. For headless **training only**, `./run.sh --no-video` requires neither an X server nor live viewer. On remote GPU Linux systems EGL is an alternative, but configuration is hardware/driver dependent and is not assumed by this script.

## `No module named mujoco` or incompatible Python

Finish `./run.sh` or `.\run.cmd` first; use the project interpreter rather than your system Python. `setup.*` expects Python 3.11. If your old `.venv` contains an incompatible interpreter, back it up or remove it only after confirming it has no needed files, then rerun setup.

## Video/render fails but training succeeds

The pipeline reports a rendering failure explicitly instead of silently claiming full success. Core training/evaluation works without video:

- macOS/Linux: `./run.sh --no-video`; Windows: `.\run.cmd -NoVideo`.
- Run the live viewer only within a logged-in desktop session (`./watch.sh` or `.\watch.cmd`).
- For macOS, `watch.sh` uses MuJoCo's `mjpython` launcher when present. GUI rendering may not work via SSH on macOS.
- The MP4 relies on a working MuJoCo GL backend **and** imageio-ffmpeg (which installs its own ffmpeg binary). Windows Remote Desktop/VM sessions sometimes lack the graphics context required by MuJoCo; use desktop graphics or disable rendering.

## `Cannot resume` / models not found

Run a first experiment without `--resume` / `-Resume`. Resume needs `models/ant_sac_latest.zip`. `models/ant_sac_replay_buffer.pkl` improves continuity but is not guaranteed to reproduce an uninterrupted run. If a hard crash preempted a latest save, inspect `models/checkpoints/`. For explicit model selection use `--model models/ant_sac_latest.zip` (Unix) or `-Model models/ant_sac_latest.zip` (`watch.ps1`).

## Training seems slow, robot does not walk, or out of memory/disk

The 2,048-step default is **only a systems test**. Train longer and inspect `logs/evaluation.json`, TensorBoard, and the rendered behavior. Neither one million steps nor any fixed hyperparameters guarantee natural gait. CPU-only training is deliberate for portability. Repeated checkpoint replay buffers accumulate disk use; archive/delete only obsolete checkpoints you don't need. Keep the latest buffer if you intend to resume.

## Useful diagnostics

macOS/Linux:

```bash
uname -sm
./.venv/bin/python -m src.check_env --steps 1
./test.sh
```

Windows PowerShell:

```powershell
[System.Environment]::OSVersion.VersionString
.\.venv\Scripts\python.exe -m src.check_env --steps 1
.\test.cmd
```

References: https://mujoco.readthedocs.io/en/stable/python.html and https://docs.astral.sh/uv/getting-started/installation/.
