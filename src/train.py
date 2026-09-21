"""SAC training with independent evaluation and optional replay-buffer resume."""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from .common import (BEST_MODEL, ENV_ID, FINAL_MODEL, LOGS, MODELS,
                     REPLAY_BUFFER, ensure_dirs, make_env)


def positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC on MuJoCo Ant-v5")
    parser.add_argument("--steps", type=positive_int, default=2048, help="Additional env steps for this invocation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Continue from models/ant_sac_latest.zip and replay buffer")
    parser.add_argument("--eval-freq", type=positive_int, default=None)
    parser.add_argument("--checkpoint-freq", type=positive_int, default=None)
    args = parser.parse_args()
    ensure_dirs()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_env = make_env(seed=args.seed, monitor_file=LOGS / "train_monitor")
    eval_env = make_env(seed=args.seed + 100, monitor_file=LOGS / "eval_monitor")
    model = None
    try:
        if args.resume:
            if not FINAL_MODEL.is_file():
                parser.error(f"Cannot resume; missing {FINAL_MODEL}")
            model = SAC.load(str(FINAL_MODEL), env=train_env, device="cpu", tensorboard_log=str(LOGS))
            if REPLAY_BUFFER.is_file():
                model.load_replay_buffer(str(REPLAY_BUFFER))
                print(f"Loaded replay buffer: {REPLAY_BUFFER}", flush=True)
            else:
                print("WARNING: Replay buffer is absent. Resuming model weights without previous experiences.", flush=True)
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
                name_prefix="ant_sac", save_replay_buffer=True,
                verbose=1,
            ),
            EvalCallback(
                eval_env, best_model_save_path=str(BEST_MODEL.parent),
                log_path=str(LOGS / "eval_history"),
                eval_freq=eval_freq, n_eval_episodes=2,
                deterministic=True, render=False, verbose=1,
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
            # Ctrl+C during learning still leaves a recoverable model + experience buffer.
            model.save(str(FINAL_MODEL))
            model.save_replay_buffer(str(REPLAY_BUFFER))
            print(f"Saved weights: {FINAL_MODEL}", flush=True)
            print(f"Saved replay buffer: {REPLAY_BUFFER}", flush=True)
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
