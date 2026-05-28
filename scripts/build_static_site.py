from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIR = ROOT / "public"


def build_static_site() -> Path:
    PUBLIC_DIR.mkdir(exist_ok=True)

    shutil.copyfile(ROOT / "1.html", PUBLIC_DIR / "index.html")

    dashboard = ROOT / "dora-dashboard-placeholder.png"
    if dashboard.exists():
        shutil.copyfile(dashboard, PUBLIC_DIR / dashboard.name)

    (PUBLIC_DIR / ".nojekyll").write_text("", encoding="utf-8")
    (PUBLIC_DIR / "deployment-metadata.json").write_text(
        json.dumps(
            {
                "project": "3DGS-Viewer",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "entrypoint": "index.html",
                "health_path": "/health",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PUBLIC_DIR


if __name__ == "__main__":
    print(build_static_site())
