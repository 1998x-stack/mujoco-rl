"""Optional simple graph of episode rewards, saved as a PNG."""
import csv
from pathlib import Path

from .common import LOGS


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monitor = LOGS / "train_monitor.monitor.csv"
    if not monitor.is_file():
        raise FileNotFoundError(f"No monitor log at {monitor}; first run training")
    with monitor.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    if not rows:
        print("No completed training episodes yet; skipping reward plot.")
        return
    values = [float(row["r"]) for row in rows]
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(values) + 1), values)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("MuJoCo Ant SAC training rewards")
    plt.tight_layout()
    destination = LOGS / "rewards.png"
    plt.savefig(destination, dpi=150)
    plt.close()
    print(f"Saved plot: {destination}")


if __name__ == "__main__":
    main()
