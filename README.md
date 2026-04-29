# eis_fit

`eis_fit` is a reusable Python library for fitting electrochemical impedance
spectroscopy (EIS) curves with the equivalent circuit
`Rs-(Rsei||CPE1)-(Rct||CPE2)-W1`.

The current implementation uses PyTorch to train circuit parameters from random
initializations and keeps the codebase intentionally small, typed, and modular.
The runtime entry point lives in `trainer.py`, and all runtime settings live in
`config.py`.

## Install

Use the existing `chemistry` conda environment and install the package in
editable mode. This installs the runtime dependencies declared in
`pyproject.toml`, including `draccus`, `torch`, and `tqdm`:

```powershell
conda run -n chemistry python -m pip install -e .
```

If you are already inside the `chemistry` environment, run:

```bash
python -m pip install -e .
```

If `python -m eis_fit.trainer` fails with `ModuleNotFoundError: draccus` or
`ModuleNotFoundError: tqdm`, the environment has not installed the project
dependencies yet; run the editable install command above.

## Run

The default input file is configured in `src/eis_fit/config.py`:

```python
input_path: Path = Path("data/EA-2.txt")
```

Fit that file with:

```powershell
conda run -n chemistry python -m eis_fit.trainer
```

CPU is the default device because the local fitting workflow is configured for
this machine. If CUDA is available and you explicitly want GPU training, pass
`--device cuda`.

Configuration values can still be overridden directly from the command line through
`draccus`. For example:

```powershell
conda run -n chemistry python -m eis_fit.trainer `
  --input_path .\data\EA-2.txt `
  --restarts 8 `
  --generate_plots false
```

Training progress is shown with `tqdm` progress bars for the restart loop, the
Adam phase, and the LBFGS refinement phase. The progress postfix reports the
current loss, the best loss for that restart, and the early-stopping wait count.

By default, the fitter also drops a leading high-frequency prefix with positive
imaginary impedance before optimization. That preprocessing is enabled because
the configured circuit does not include an inductive element, so those points
are otherwise impossible to fit well. The output CSV still contains all original
points and marks whether each row was used for fitting.

## Chemistry-informed constraints

Equivalent-circuit fitting is often non-unique. If `Rsei` becomes too large or
`Rct` becomes too small, inject chemistry knowledge directly into the objective
instead of only increasing epochs.

Use hard bounds for impossible values and soft constraints for preferred ranges:

```python
from pathlib import Path

from eis_fit.config import ChemistryConstraints, FitConfig, ParameterBounds

config = FitConfig(
    input_path=Path("data/EA-2.txt"),
    restarts=8,
    bounds=ParameterBounds(
        rsei=(1.0, 120.0),
        rct=(50.0, 600.0),
    ),
    chemistry_constraints=ChemistryConstraints(
        rsei=(10.0, 80.0, 0.2),
        rct=(80.0, 400.0, 0.2),
        minimum_rct_to_rsei_ratio=1.0,
        rct_rsei_ratio_weight=0.5,
        minimum_second_arc_tau_ratio=2.0,
        second_arc_tau_ratio_weight=0.5,
    ),
)
```

This combination does two different jobs:

- Hard bounds prevent impossible parameter values.
- Soft constraints tell the optimizer which region is chemically preferred when
  several fits explain the spectrum similarly well.

The arc-time-constant constraint is useful when the two `R || CPE` branches try
to swap identities. It pushes the second arc to remain slower than the first
arc, which is often the chemically meaningful ordering for `SEI -> charge
transfer`.

An equivalent console script is also installed:

```powershell
conda run -n chemistry eis-fit --input_path .\data\EA-2.txt
```

## Outputs

Each run writes a timestamped directory under `outputs/` with:

- `best_params.json`: best-fit parameters and runtime summary.
- `fit_curve.csv`: measured and fitted impedance values.
- `loss_history.csv`: per-epoch loss records for every restart.
- `restart_summary.csv`: best loss from each random restart.
- `nyquist.png`: Nyquist plot of measured vs fitted data.
- `bode.png`: Bode magnitude and phase plot.

## Circuit intuition

You do not need electrochemistry background to read the fitted curve:

- `Rs` is the high-frequency intercept on the real axis.
- `Rsei || CPE1` forms the first depressed semicircle.
- `Rct || CPE2` forms the second depressed semicircle.
- `W1` is a finite-length open Warburg term, which bends the low-frequency end
  into a diffusion tail.

So the full Nyquist curve usually looks like a starting point on the real axis,
followed by one or two flattened arcs, then a low-frequency tail.
