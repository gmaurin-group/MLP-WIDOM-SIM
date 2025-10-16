# Widom Insertion with Machine Learning Potentials (MLPs)

Efficient Widom insertion simulations for porous materials (e.g., MOFs/COFs) using pre‑trained **ASE-compatible machine learning interatomic potentials** (e.g., Matlantis/PFP). This repository provides:

- A **batch driver** for running Widom insertions across many CIF structures.
- A modular **`AI_Widom`** class that performs insertions, reports **Henry’s constant (K_H)** and **isosteric heat of adsorption (Q<sub>st</sub>)**, and saves minimum‑energy snapshots.

---

## ✨ Features
- Random or grid‑based insertions of a rigid adsorbate into a periodic framework.
- Random rigid‑body rotations per insertion.
- Optional **van der Waals overlap rejection**.
- On‑the‑fly tracking of the **lowest‑energy** configuration and trajectory.
- Robust logging of **K_H** (mol·kg⁻¹·Pa⁻¹) and **Q<sub>st</sub>** (kJ·mol⁻¹).
- Works with **ASE calculators** (examples shown with **Matlantis/PFP**).

---

## 📁 Repository Structure
```
.
├─ run_widom_batch.py        # Batch CLI driver
├─ src/
│  ├─ widom.py            # AI_Widom class (core logic)
│  ├─ utilities.py           # helpers: rotations, overlap checks, etc.
│  └─ __init__.py
├─ requirements.txt
├─ README.md                 # this file
└─ LICENSE
```

> **Note**: Your code may use different filenames (e.g., `AI_Widom` class in `src/widom.py`). Update paths accordingly.

---

## 🔧 Installation
Create and activate an environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`** (adjust versions as needed):
```txt
ase
numpy
scipy
molmod
pfp_api_client
matlantis_features
```

If you plan to use GPU acceleration with Matlantis/PFP, follow the vendor’s instructions for CUDA/toolkit compatibility.

---

## 🚀 Quick Start
Run Widom insertion for all CIFs in a directory using the batch driver:

```bash
python run_widom_batch.py \
  --cif-dir ./structures \
  --adsorbate molecules/CO2.xyz \
  --temperature 298 \
  --pressure 1.0 \
  --mc-steps 50001 \
  --device cuda
```

**Outputs** for each framework are written under:
```
results_widom_<FRAME_NAME>/
  ├─ snapshots/
  │    ├─ widom_<frame>_<ads>_e_min.pdb
  │    └─ widom_<frame>_<ads>_e_min_traj.extxyz
  ├─ e_traj/
  │    └─ widom_<frame>_<ads>_e_traj.npy
  └─ widom_<frame>_<ads>.log
```

---

## 🧭 Command‑Line Usage (`input-widom.py`)

```bash
python input-widom.py \
  --cif-dir <DIR_WITH_CIFS> \
  --adsorbate <ADSORBATE_XYZ_OR_MOL> \
  --temperature <K> \
  --pressure <bar> \
  [--mc-steps 50001] \
  [--seed 12345] \
  [--device {cpu,cuda}] \
  [--min-cell-length 10.0] \
  [--insertion-grid grid.pkl] \
  [--pfp-model-version v6.0.0] \
  [--pfp-calc-mode CRYSTAL_U0_PLUS_D3] \
  [--energy-type TOTALENERGY]
```

**Key options**
- `--min-cell-length`: Small cells are automatically expanded into a supercell so that each lattice vector ≥ this value (Å). Defaults to **10 Å**.
- `--insertion-grid`: Optional **pickled** array/list of precomputed insertion points (Cartesian Å). If omitted, positions are drawn uniformly from the unit cell.
- `--device`: Forwarded to your calculator/workflow (`cpu` or `cuda`).
- `--pfp-*`: Convenience knobs for Matlantis/PFP.

---

## 🧩 API Usage (`AI_Widom`)
Minimal example of using the class directly:

```python
from ase.io import read
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
from molmod.units import kelvin
from src.ai_widom import AI_Widom
from ase.data import vdw_radii as ASE_VDW_RADII

# Build calculator (Matlantis/PFP example)
estimator = Estimator(model_version="v6.0.0", calc_mode=EstimatorCalcMode.CRYSTAL_U0_PLUS_D3)
calc = ASECalculator(estimator)
calc.set_default_properties(["energy"])  # required

atoms_frame = read("structures/MOF_001.cif")
atoms_ads   = read("molecules/C2H4.xyz")

widom = AI_Widom(
    model=calc,
    model_energy="TOTALENERGY",
    name_frame="MOF_001",
    atoms_frame=atoms_frame,
    insertion_grid=None,  # or a list/array of positions in Å
    name_ads="CO2",
    atoms_ads=atoms_ads,
    T=298 * kelvin,
    vdw_radii=ASE_VDW_RADII,
    device="cuda",
    seed=12345,
)

widom.run(iterations=50001)
```

---

## 📐 Units & Conventions
- **Energies** from ASE are in eV and internally converted to **atomic units** via `molmod.units`.
- **Temperature** is passed as `T_K * kelvin`.
- Reported values:
  - **Q<sub>st</sub>** in **kJ·mol⁻¹**
  - **K_H** in **mol·kg⁻¹·Pa⁻¹**
- **vDW overlap check** uses ASE’s `vdw_radii` reduced by 0.35 Å (tunable in `ai_widom.py`).

---

## 🧱 Insertion Grids (Optional)
You can speed up convergence by inserting only at precomputed grid points but not recommended for Widom simulations:
<!-- 
```python
import pickle, numpy as np
points = np.random.rand(1000, 3)  # your own positions in Å (example)
with open("grid.pkl", "wb") as f:
    pickle.dump(points, f)
``` -->

pass `--insertion-grid grid.pkl` to the CLI.

---

## ⚙️ Performance Tips
- Use **supercells** when unit cells are very small (`--min-cell-length` helps automate this).
- Prefer **GPU** (`--device cuda`) if your calculator supports it.
- Increase insertions (`--mc-steps`) for smoother statistics.

---

## 🧪 Reproducibility
- Set `--seed` to make trials repeatable.
- The batch driver writes a `widom_batch_runtime.txt` with timing and failures.

---

## 🩺 Troubleshooting
- **No CIFs found**: check `--cif-dir` and file extensions (`.cif`).
- **Calculator error**: ensure the MLP backend is installed, licensed, and versions match CUDA/toolkit (if using GPU).
- **KH or Qst looks off**: increase `--mc-steps`, verify units, and review the overlap threshold (`vdw_radii - 0.35`).
- **Tiny/anisotropic cells**: raise `--min-cell-length` to expand supercells.

---

## 📚 Citation
If you use this code in academic work, please cite:

```bibtex
@software{repo_widom_mlp,
  title        = {Towards Accurate and Scalable High-throughput MOF Adsorption Screening: Merging Classical Force Fields and Universal Machine Learned Potentials},
  author       = {Satyanarayana Bonakala et al.},
  year         = {2025},
  url          = {},
  note         = {Version V1.0.0}
}
```

And any interatomic potential / MLP framework you use (e.g., Matlantis/PFP) plus standard references on Widom insertion.

---

## 📄 License

---

<!-- ## 🗺️ Roadmap (Optional)
- CSV/Parquet logging of per‑iteration statistics.
- Multi‑adsorbate batch runs and mixtures.
- Optional rigid/flexible framework modes.
- Pore‑space grid generators and visualization helpers. -->

---
MIT License

Copyright (c) 2025 gmaurin-group

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## 🤝 Acknowledgments
Thanks to the developers and maintainers of **ASE**, **molmod**, and the MATLANTIS MLP framework. Widom insetion code is adopted from Goeminne et al., DOI: 10.1021/acs.jctc.3c00495.

