"""SAC training, crash-consistent snapshots, persistent best-model selection."""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from .checkpoint import (LATEST, load_best_score, load_checkpoint_paths,
                         save_best_score, save_checkpoint)
from .common import (BEST_MODEL, ENV_ID, FINAL_MODEL, LOGS, MODELS,
                     REPLAY_BUFFER, ensure_dirs, make_env)

BEST_SCORE = BEST_MODEL.parent / "score.json"


def positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a positive integer") from exc
    if value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return value


class PersistentEvalCallback(EvalCallback):
    """Keep best-model comparison across separate --resume invocations."""

    def __init__(self, *args, resume: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if resume and BEST_MODEL.is_file():
            previous = load_best_score(BEST_SCORE)
            if previous is not None:
                self.best_mean_reward = previous
            else:
                # Historic versions did not record a score. Preserve that model.
                legacy = BEST_MODEL.parent / "best_model_before_score_tracking.zip"
                if not legacy.exists():
                    import shutil
                    shutil.copy2(BEST_MODEL, legacy)
                print(f"Preserved historic best model: {legacy}", flush=True)

    def _on_step(self) -> bool:
        old = self.best_mean_reward
        result = super()._on_step()
        if self.best_mean_reward > old:
            save_best_score(BEST_SCORE, self.best_mean_reward)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC on MuJoCo Ant-v5")
    parser.add_argument("--steps", type=positive_int, default=2048, help="Additional env steps for this invocation")
    parser.add_argument("--seed", type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Resume latest model and replay buffer")
    mode.add_argument("--fresh", action="store_true", help="Explicitly begin a new experiment, archiving previous artifacts")
    parser.add_argument("--eval-freq", type=positive_int, default=None)
    parser.add_argument("--checkpoint-freq", type=positive_int, default=None)
    args = parser.parse_args()
    ensure_dirs()

    previous_exists = LATEST.is_file() or FINAL_MODEL.is_file() or BEST_MODEL.is_file()
    if previous_exists and not (args.resume or args.fresh):
        parser.error("Existing training artifacts detected; use --resume or --fresh (archives previous artifacts).")
    if args.fresh and previous_exists:
        from datetime import datetime, timezone
        import shutil
        import uuid
        archive = MODELS / "archive" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
        archive.mkdir(parents=True)
        for path in (LATEST, FINAL_MODEL, REPLAY_BUFFER, BEST_MODEL, BEST_SCORE):
            if path.exists():
                dest = archive / path.relative_to(MODELS)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest))
        if LOGS.exists():
            for old in LOGS.iterdir():
                dest = archive / "logs" / old.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old), str(dest))
        print(f"Archived previous experiment metadata, logs and models: {archive}", flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_env = make_env(seed=args.seed, monitor_file=LOGS / "train_monitor")
    eval_env = make_env(seed=args.seed + 100, monitor_file=LOGS / "eval_monitor")
    model = None
    try:
        if args.resume:
            paths = load_checkpoint_paths()
            if paths is not None:
                model_file, buffer_file, metadata = paths
                if metadata["seed"] != args.seed:
                    parser.error(f"Checkpoint seed={metadata['seed']}; pass --seed {metadata['seed']} or start --fresh")
            elif FINAL_MODEL.is_file():
                model_file, buffer_file = FINAL_MODEL, REPLAY_BUFFER
            else:
                parser.error("Cannot resume: no checkpoint found; run without --resume first")
            model = SAC.load(str(model_file), env=train_env, device="cpu", tensorboard_log=str(LOGS))
            if not buffer_file.is_file():
                parser.error(f"Missing replay buffer: {buffer_file}. Recovery requires a matching model and buffer.")
            model.load_replay_buffer(str(buffer_file))
            print(f"Resumed model={model_file}, replay={buffer_file}", flush=True)
        else:
            model = SAC(
                "MlpPolicy", train_env,
                learning_rate=3e-4, buffer_size=100_000,
                learning_starts=256, batch_size=128,
                tau=0.005, gamma=0.99,
                train_freq=1, gradient_steps=1,
                ent_coef="auto", policy_kwargs={"net_arch": [256, 256]},
                seed=args.seed, device="cpu", verbose=1,
                tensorboard_log=str(LOGS),
            )

        eval_freq = args.eval_freq or max(1024, args.steps // 4)
        checkpoint_freq = args.checkpoint_freq or max(2048, args.steps // 2)
        callbacks = CallbackList([
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(MODELS / "checkpoints"),
                name_prefix="ant_sac", save_replay_buffer=True, verbose=1,
            ),
            PersistentEvalCallback(
                eval_env, best_model_save_path=str(BEST_MODEL.parent),
                log_path=str(LOGS / "eval_history"),
                eval_freq=eval_freq, n_eval_episodes=2,
                deterministic=True, render=False, verbose=1,
                resume=args.resume,
            ),
        ])
        print(f"Training {ENV_ID}: {args.steps} additional steps; seed={args.seed}; resume={args.resume}", flush=True)
        print(f"Monitor: tensorboard --logdir {LOGS}", flush=True)
        try:
            model.learn(
                total_timesteps=args.steps, callback=callbacks,
                log_interval=10, tb_log_name="Ant_SAC",
                reset_num_timesteps=not args.resume,
            )
        finally:
            # Publish the pointer only once the entire snapshot has been saved.
            saved = save_checkpoint(model, seed=args.seed)
            print(f"Saved consistent SAC snapshot: {saved}", flush=True)
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
