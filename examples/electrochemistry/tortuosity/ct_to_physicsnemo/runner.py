"""End-to-end runner: `python -m ct_to_physicsnemo.runner configs/default.yaml`.

Steps mirror the weekly plan in student_project_CT_to_physicsnemo.md:
  1. load + REV-crop CT
  2. segment
  3. metrics (epsilon, av, PSD, tau via TauFactor)
  4. mesh + cleanup -> STL
  5. PhysicsNeMo geometry + sampling
  6. Laplace PINN -> tau_PINN ; compare to TauFactor
  7. EIS sweep over configured ω grid
  8. (optional) E1 / E2 ablations
Each step writes intermediate artefacts to runs/<timestamp>/.
"""

from __future__ import annotations
import sys
from pathlib import Path


def main(config_path: str | Path) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main(sys.argv[1])
