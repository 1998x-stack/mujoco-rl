#!/usr/bin/env bash
# macOS/Linux: setup, physics check, SAC training, evaluation, optional MP4.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
usage() {
  cat <<'HELP'
Usage: ./run.sh [--quick|--full|--steps N] [--resume] [--no-video] [--no-open]
                [--headless-video] [--help]
  Default      2048 training steps + evaluation + video when display is available.
  --full       1,000,000 additional training steps (long experiment).
  --steps N    Additional steps, N >= 1.
  --resume     Load latest SAC weights and replay buffer.
  --no-video   Skip video rendering (always safe for remote/headless sessions).
  --no-open    Save MP4 without opening a video player.
  --headless-video  Linux only: try CPU OSMesa rendering without a display.
                    Requires libOSMesa/OpenGL runtime packages already installed.
On Windows use run.cmd or run.ps1 (native PowerShell; WSL is not required).
HELP
}
STEPS=2048; RESUME=0; VIDEO=1; OPEN=1; HEADLESS_VIDEO=0; VIDEO_EXPLICIT=0
while (($#)); do
  case "$1" in
    --quick) STEPS=2048; shift ;;
    --full) STEPS=1000000; shift ;;
    --steps)
      if (($# < 2)) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then echo 'ERROR: --steps needs a positive integer.' >&2; exit 2; fi
      STEPS="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --no-video) VIDEO=0; shift ;;
    --no-open) OPEN=0; shift ;;
    --headless-video) HEADLESS_VIDEO=1; VIDEO=1; VIDEO_EXPLICIT=1; OPEN=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
OS="$(uname -s)"
case "$OS" in Darwin|Linux) ;; *) echo 'Unsupported OS: use run.ps1 on Windows.' >&2; exit 2 ;; esac
if ((HEADLESS_VIDEO)) && [[ "$OS" != Linux ]]; then echo '--headless-video requires Linux.' >&2; exit 2; fi
if [[ "$OS" == Linux && -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" && $VIDEO_EXPLICIT -eq 0 ]]; then
  VIDEO=0; OPEN=0
  echo '[display] No graphical session detected. Skipping MP4; run with --headless-video to try OSMesa.'
fi
if ((HEADLESS_VIDEO)); then export MUJOCO_GL=osmesa; export PYOPENGL_PLATFORM=osmesa; fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
echo '[1/5] Setup Python + packages.'
bash "$ROOT/setup.sh"
PYTHON="$ROOT/.venv/bin/python"
echo '[2/5] MuJoCo environment check.'
"$PYTHON" -m src.check_env --steps 32
echo "[3/5] SAC training: $STEPS additional steps."
TRAIN_ARGS=(--steps "$STEPS")
if ((RESUME)); then TRAIN_ARGS+=(--resume); fi
"$PYTHON" -m src.train "${TRAIN_ARGS[@]}"
echo '[4/5] Evaluate and plot.'
"$PYTHON" -m src.evaluate --episodes 3
"$PYTHON" -m src.plot || echo 'WARN: Reward plot unavailable (no finished training episodes).'
if ((VIDEO)); then
  echo '[5/5] Record MP4.'
  if ! "$PYTHON" -m src.record --frames 250; then
    echo 'ERROR: Training succeeded but video rendering failed. See docs/TROUBLESHOOTING.md; use --no-video on headless systems.' >&2
    exit 5
  fi
  if ((OPEN)); then
    if [[ "$OS" == Darwin ]]; then
      open "$ROOT/videos/ant_demo.mp4" || echo 'MP4 saved but could not launch the viewer.'
    elif command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$ROOT/videos/ant_demo.mp4" || echo 'MP4 saved but could not launch the viewer.'
    else
      echo 'MP4 saved. Open videos/ant_demo.mp4 with a video player.'
    fi
  fi
else
  echo '[5/5] Video skipped.'
fi
echo 'SUCCESS: models/ant_sac_latest.zip and logs/evaluation.json'
if ((VIDEO)); then echo 'SUCCESS: videos/ant_demo.mp4'; fi
