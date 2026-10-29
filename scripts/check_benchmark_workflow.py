"""CI guardrail against accidentally weakening the expensive publication workflow."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ant-benchmark.yml"


def verify_workflow() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert isinstance(triggers, dict) and set(triggers) == {"workflow_dispatch"}, (
        "Long-run benchmark must remain manual-only"
    )
    jobs = workflow["jobs"]
    train = jobs["train"]
    aggregate = jobs["aggregate"]
    assert train["strategy"]["matrix"]["seed"] == [0, 1, 2, 3, 4]
    assert train["strategy"]["max-parallel"] <= 2
    assert 60 <= train["timeout-minutes"] <= 360
    scripts = "\n".join(str(step.get("run", "")) for step in train["steps"])
    for required in (
        "src.train --steps 1000000",
        "--seed ${{ matrix.seed }}",
        "src.evaluate --episodes 10 --seed 1000",
        "src.verify --require-evaluation",
        "src.evidence evidence",
    ):
        assert required in scripts, f"Missing per-seed verification command: {required}"
    assert aggregate["needs"] == ["train"], "All five seed jobs must complete first"
    aggregate_steps = "\n".join(str(step.get("run", "")) for step in aggregate["steps"])
    for required in ("src.evidence", "src.benchmark", "--publication", "src.scorecard"):
        assert required in aggregate_steps, f"Missing aggregation gate: {required}"
    assert any(step.get("uses", "").startswith("actions/upload-artifact@")
               for step in train["steps"])
    assert any(step.get("uses", "").startswith("actions/upload-artifact@")
               for step in aggregate["steps"])


if __name__ == "__main__":
    verify_workflow()
    print("PASS: manual, five-seed, million-step workflow and publication gates")
