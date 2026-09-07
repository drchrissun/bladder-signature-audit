"""Create the final supplementary Figures S7 and S8 from the candidate plots."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures"


def main() -> None:
    module_path = Path(__file__).with_name("make_nature_candidates.py")
    spec = importlib.util.spec_from_file_location("nature_candidates", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.figure_sensitivity_grid()
    module.figure_pathway_bubble()

    mapping = {
        "NatureCandidate_SensitivityDiagnostics": "FigureS7",
        "NatureCandidate_PathwayBubble": "FigureS8",
    }
    for source, target in mapping.items():
        for extension in ("pdf", "png"):
            source_path = FIGURES / f"{source}.{extension}"
            shutil.copy2(source_path, FIGURES / f"{target}.{extension}")
            source_path.unlink()
    print("Figures S7 and S8 written")


if __name__ == "__main__":
    main()
