from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src import load_config
from src.training.trainer import train

SCENARIOS = [
    {"name": "A", "use_learnable_pe": False, "use_focal_loss": False},  # vanilla
    {"name": "B", "use_learnable_pe": True,  "use_focal_loss": True},   # proposed
    {"name": "C", "use_learnable_pe": False, "use_focal_loss": True},   # ablasi PE
    {"name": "D", "use_learnable_pe": True,  "use_focal_loss": False},  # ablasi loss
]


def run_all_scenarios(config) -> List[Dict]:
    summaries = []
    for sc in SCENARIOS:
        print(f"=== Scenario {sc['name']} ===")
        summary = train(
            config,
            scenario_name=sc["name"],
            use_learnable_pe=sc["use_learnable_pe"],
            use_focal_loss=sc["use_focal_loss"],
        )
        summaries.append(summary)
    out = Path(config.paths.results_dir) / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out, index=False)
    print(f"Summary saved: {out}")
    return summaries


def main() -> None:
    cfg = load_config("configs/default.yaml")
    run_all_scenarios(cfg)


if __name__ == "__main__":
    main()
