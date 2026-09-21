# Engineering review and validation report

Baseline: `1998x-stack/mujoco-rl` at `67c6434420b1e04e304f28aabb6592583bddb60c`. Reviewed OS-native launchers, Python environment boundary, SAC training lifecycle, checkpoint/replay consistency, best-model selection, evaluation, rendering, dependency management and tests. This report separates demonstrated local checks from pending three-OS integration checks.

## Findings and fixes

| Severity | Finding | Change |
|---|---|---|
| P0 | Separate mutable model/replay saves can leave a mixed pair after an interruption | Immutable snapshot directories; SHA-256 checksums; atomic `latest.json` pointer published after both writes |
| P0 | New EvalCallback instance resets best score, allowing worse resumed evaluations to overwrite the previous best | Persist best score across invocations; back up a historic best model that lacks score metadata |
| P1 | Re-running from scratch silently discards progress | Fail without `--resume` or `--fresh` when active outputs exist; fresh mode archives active outputs and logs |
| P1 | Optional video failure makes successfully completed training appear failed | Video failure becomes warning; remote/headless mode can skip rendering |
| P1 | Tests claimed standard-library-only operation while importing PyTorch and Stable-Baselines3 | Dependency-free tests cover atomic save, failure injection, file corruption, model resolution and score persistence |
| P2 | Fixed playback FPS could misrepresent motion speed | MP4 playback FPS derived from actual environment time step |
| P2 | `curl` required even with preinstalled uv | Only check `curl` when uv must be downloaded |

## Verified locally

- `bash -n run.sh setup.sh watch.sh test.sh` — passed in a Linux container.
- `python -m compileall -q src tests` — passed in the Linux container.
- `python -m unittest discover -s tests -v` — six contract tests passed, including fault-injection and integrity checks.

The container did not have MuJoCo, Gymnasium or Stable-Baselines3 installed and had no network access, so full physical environment initialization, real SAC training, live macOS/Windows execution and OpenGL video rendering were **not** tested there. GitHub CI is configured to attempt dependency installation and physics checks on Windows, Linux and macOS; inspect an actual Actions run and logs before claiming success.

## Reproducibility boundaries

The checkpoint pointer is an application-level consistency mechanism on a local filesystem, not a guarantee against filesystem corruption or sudden power loss. Restarting SAC with its weights and replay buffer does not restore the exact environment state, Python/NumPy/PyTorch RNG streams, callback counters or scheduling; resumed and uninterrupted runs may diverge. Dependencies are specified by compatibility ranges, not a hashed lockfile. The default 2,048 steps are a pipeline test, not a learned walking policy. Compare long experiments with multiple independent seeds, and report rewards together with forward displacement and video observations.

## Future work (priority order)

1. Verify complete short train/save/resume/evaluate flow on all three operating systems, including native Windows PowerShell and macOS graphical sessions.
2. Establish platform-specific tested dependency locks and CI coverage for a short training+resume integration test.
3. Add configurable replay/snapshot retention and manual recovery selection for historic snapshots.
4. Track experiments by run ID, add multi-seed evaluation and uncertainty summaries.
5. For exact resume, serialize relevant environment, RNG, callback and optimizer state; verify interrupted-versus-uninterrupted runs experimentally.
