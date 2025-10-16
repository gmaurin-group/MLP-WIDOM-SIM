#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_widom_batch.py
------------------
Batch Widom insertion driver for porous materials using a pre-trained MLP calculator.

This script:
  1) Scans a directory for CIF files,
  2) Builds a minimally sized supercell (e.g., min cell length >= 10 Å),
  3) Runs Widom insertion Monte Carlo using an AI/ML potential (e.g., Matlantis/PFP),
  4) Records wall-clock timing for the full batch.

Example:
    python run_widom_batch.py \
        --cif-dir ./structures \
        --adsorbate CO2.xyz \
        --temperature 298 \
        --pressure 1.0 \
        --mc-steps 50001 \
        --device cuda

Requirements:
    - ASE, NumPy, molmod, pfp_api_client, matlantis_features (if using Matlantis)
    - Your project module exposing `AI_Widom` at `src.widom`

Author: Your Name
License: MIT
"""

from __future__ import annotations

import os
import sys
import glob
import time
import pickle
import argparse
from typing import Optional, List

import numpy as np

from ase.io import read
from ase.build import make_supercell
from ase.data import vdw_radii as ASE_VDW_RADII  # array-like VdW radii per Z

from molmod.units import kelvin
from src.widom import AI_Widom

# --- Optional: Matlantis (PFP) MLP configuration ---
# Comment out these imports if you inject your own ASE calculator externally.
from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode


def build_calculator(model_version: str = "v6.0.0",
                     calc_mode: str = "CRYSTAL_U0_PLUS_D3") -> ASECalculator:
    """
    Construct the ASE calculator for the MLP model (Matlantis/PFP here).

    Args:
        model_version: Version string for the estimator model.
        calc_mode:     Calculation mode (EstimatorCalcMode name).

    Returns:
        ASECalculator instance with default property "energy".
    """
    mode = getattr(EstimatorCalcMode, calc_mode)
    estimator = Estimator(model_version=model_version, calc_mode=mode)
    calc = ASECalculator(estimator)
    calc.set_default_properties(["energy"])
    return calc


def load_insertion_grid(grid_path: Optional[str]) -> Optional[dict]:
    """
    Load optional grid-based insertion data (pickled dict).

    Args:
        grid_path: Path to a .pkl file or None.

    Returns:
        Deserialized grid data dict or None.
    """
    if grid_path is None:
        return None
    if not os.path.isfile(grid_path):
        raise FileNotFoundError(f"Insertion grid file not found: {grid_path}")
    with open(grid_path, "rb") as f:
        return pickle.load(f)


def choose_supercell_scale(cell_lengths: np.ndarray,
                           min_length: float = 10.0) -> np.ndarray:
    """
    Compute diagonal supercell scale factors to guarantee minimal cell length.

    Args:
        cell_lengths: (3,) array of lattice vector lengths in Å.
        min_length:   Minimum target length in Å for each lattice vector.

    Returns:
        3x3 diagonal integer matrix for `make_supercell`.
    """
    scales = [int(np.ceil(min_length / L)) if L < min_length else 1 for L in cell_lengths]
    return np.diag(scales)


def find_cif_files(cif_dir: str) -> List[str]:
    """
    Find and sort all CIF files in a directory.

    Args:
        cif_dir: Directory containing .cif files.

    Returns:
        Sorted list of CIF file paths.
    """
    return sorted(glob.glob(os.path.join(cif_dir, "*.cif")))


def run_single_widom(cif_path: str,
                     adsorbate_path: str,
                     temperature_K: float,
                     pressure_bar: float,
                     mc_steps: int,
                     device: str,
                     seed: int,
                     min_cell_length: float,
                     calculator: ASECalculator,
                     insertion_grid_path: Optional[str] = None,
                     model_energy_type: str = "TOTALENERGY") -> None:
    """
    Run Widom insertion for a single CIF with a built supercell.

    Args:
        cif_path:            Path to framework CIF.
        adsorbate_path:      Path to adsorbate (.xyz/.mol).
        temperature_K:       Simulation temperature (K).
        pressure_bar:        Simulation pressure (bar). (If needed by your AI_Widom.)
        mc_steps:            Number of MC steps.
        device:              'cpu' or 'cuda' (forwarded to AI_Widom).
        seed:                RNG seed for reproducibility.
        min_cell_length:     Minimum supercell length in Å.
        calculator:          ASE-compatible calculator providing energies.
        insertion_grid_path: Optional path to a pickle grid file.
        model_energy_type:   Energy type string expected by AI_Widom.

    Returns:
        None. (Side effects: creates outputs via AI_Widom.)
    """
    name_frame = os.path.splitext(os.path.basename(cif_path))[0]
    print(f"\n[INFO] Structure: {name_frame}")

    # Read framework (unit cell)
    atoms_frame_unit = read(cif_path)
    cell_lengths = np.array(atoms_frame_unit.cell.lengths(), dtype=float)

    # Scale up to ensure adequate sampling box
    S = choose_supercell_scale(cell_lengths, min_length=min_cell_length)
    atoms_frame = make_supercell(atoms_frame_unit, S)

    print(f"[INFO] Supercell scale factors: {np.diag(S).tolist()}")
    print(f"[INFO] Final cell lengths (Å): {atoms_frame.cell.lengths()}")

    # Load adsorbate (only once outside loop if you prefer; kept here for clarity)
    atoms_ads = read(adsorbate_path)

    # Optional insertion grid
    insertion_grid = load_insertion_grid(insertion_grid_path)

    # Initialize Widom runner
    widom = AI_Widom(
        model=calculator,
        model_energy=model_energy_type,
        name_frame=name_frame,
        atoms_frame=atoms_frame,
        insertion_grid=insertion_grid,
        name_ads=os.path.splitext(os.path.basename(adsorbate_path))[0],
        atoms_ads=atoms_ads,
        T=temperature_K * kelvin,
        vdw_radii=ASE_VDW_RADII,   # pass ASE VdW radii array for contact checks
        device=device,
        seed=seed,
        # If your AI_Widom supports pressure directly, pass it here as well:
        # P=pressure_bar * bar  # (convert with molmod.units if needed)
    )

    # Run MC
    print(f"[INFO] MC steps: {mc_steps} | Device: {device} | Seed: {seed}")
    widom.run(mc_steps)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Batch Widom insertion in porous solids using an MLP calculator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cif-dir", required=True, help="Directory containing .cif/.pdb files")
    p.add_argument("--adsorbate", required=True, help="Path to adsorbate (.xyz/.mol/.pdb)")
    p.add_argument("--temperature", type=float, required=True, help="Temperature in K")
    p.add_argument("--pressure", type=float, required=True, help="Pressure in bar")
    p.add_argument("--mc-steps", type=int, default=50001, help="Number of MC steps per structure")
    p.add_argument("--seed", type=int, default=12345, help="RNG seed")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Computation device")
    p.add_argument("--min-cell-length", type=float, default=10.0,
                   help="Minimum target cell length (Å) for supercell construction")
    p.add_argument("--insertion-grid", default=None,
                   help="Optional pickle file with precomputed insertion grid")
    # MLP (Matlantis/PFP) options
    p.add_argument("--pfp-model-version", default="v6.0.0",
                   help="PFP model version string")
    p.add_argument("--pfp-calc-mode", default="CRYSTAL_U0_PLUS_D3",
                   help="PFP EstimatorCalcMode (e.g., CRYSTAL_U0_PLUS_D3)")
    p.add_argument("--energy-type", default="TOTALENERGY",
                   help="Energy type string passed to AI_Widom")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # Validate inputs
    if not os.path.isdir(args.cif_dir):
        print(f"[ERROR] CIF directory not found: {args.cif_dir}")
        return 2
    if not os.path.isfile(args.adsorbate):
        print(f"[ERROR] Adsorbate file not found: {args.adsorbate}")
        return 2

    cif_files = find_cif_files(args.cif_dir)
    if len(cif_files) == 0:
        print(f"[WARN] No CIF files found in: {args.cif_dir}")
        return 0

    # Build calculator (Matlantis/PFP by default)
    try:
        calculator = build_calculator(
            model_version=args.pfp_model_version,
            calc_mode=args.pfp_calc_mode
        )
    except Exception as exc:
        print(f"[ERROR] Failed to initialize MLP calculator: {exc}")
        return 3

    # Batch header
    print("========== Widom Batch ==========")
    print(f"Temperature [K]          : {args.temperature}")
    print(f"Pressure   [bar]         : {args.pressure}")
    print(f"MC steps per structure   : {args.mc_steps}")
    print(f"Min cell length [Å]      : {args.min_cell_length}")
    print(f"Device                   : {args.device}")
    print(f"CIF directory            : {args.cif_dir}")
    print(f"# CIFs found             : {len(cif_files)}")
    print(f"MLP model version        : {args.pfp_model_version}")
    print(f"MLP calc mode            : {args.pfp_calc_mode}")
    print(f"Insertion grid           : {args.insertion_grid or 'None'}")
    print(f"Start time               : {time.ctime()}")
    print("=================================")

    t0 = time.perf_counter()
    failures = []

    for cif_path in cif_files:
        try:
            run_single_widom(
                cif_path=cif_path,
                adsorbate_path=args.adsorbate,
                temperature_K=args.temperature,
                pressure_bar=args.pressure,
                mc_steps=args.mc_steps,
                device=args.device,
                seed=args.seed,
                min_cell_length=args.min_cell_length,
                calculator=calculator,
                insertion_grid_path=args.insertion_grid,
                model_energy_type=args.energy_type,
            )
        except Exception as exc:
            print(f"[ERROR] {os.path.basename(cif_path)} failed: {exc}")
            failures.append((cif_path, str(exc)))

    t1 = time.perf_counter()

    # Write batch timing and a brief summary
    summary_path = os.path.join(args.cif_dir, "widom_batch_runtime.txt")
    with open(summary_path, "w") as f:
        f.write("======= ALL DONE =======\n")
        f.write(f"End time: {time.ctime()}\n")
        f.write(f"Total time: {(t1 - t0):.2f} seconds\n")
        f.write(f"Total time: {(t1 - t0)/60:.2f} minutes\n")
        if failures:
            f.write("\nFailures:\n")
            for fp, msg in failures:
                f.write(f"- {os.path.basename(fp)} : {msg}\n")

    print("\n======= ALL DONE =======")
    print("End time:", time.ctime())
    print(f"Total time: {(t1 - t0):.2f} seconds ({(t1 - t0)/60:.2f} minutes)")
    if failures:
        print(f"[WARN] {len(failures)} structure(s) failed. See details in {summary_path}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())