from __future__ import annotations

import csv
import json
from pathlib import Path


def export_metrics(metrics: dict[str, float | str], out_dir: str | Path, stem: str = "ragas_report") -> dict[str, str]:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    json_path = path / f"{stem}.json"
    csv_path = path / f"{stem}.csv"

    json_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])

    return {"json": str(json_path), "csv": str(csv_path)}
