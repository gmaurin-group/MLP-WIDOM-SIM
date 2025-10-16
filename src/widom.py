import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation

from molmod.units import *
from molmod.constants import *
import torch
from ase import Atoms
from ase.io import write
from ase.calculators.singlepoint import SinglePointCalculator
from time import time

from .utilities import _random_rotation, _random_translation, _random_position, _vdw_overlap


class AI_Widom:

    def __init__(self, model, model_energy, name_frame, atoms_frame, insertion_grid, name_ads, atoms_ads, T, vdw_radii, device, seed=None):

        self.model = model
        self.model_energy = model_energy
        self.name_frame = name_frame

        self.atoms_frame = atoms_frame
        self.n_frame = len(self.atoms_frame)
        self.cell = np.array(self.atoms_frame.get_cell())
        self.V = self.atoms_frame.cell.volume * angstrom**3
        self.mass_frame = np.sum(atoms_frame.get_masses()) * amu
        self.rho = self.mass_frame / self.V

        self.insertion_grid = insertion_grid
        self.n_insertion_grid = len(self.insertion_grid) if np.any(self.insertion_grid) else 0

        self.name_ads = name_ads
        self.atoms_ads = atoms_ads
        self.n_ads = len(self.atoms_ads)
        self.mass_ads = np.sum(atoms_ads.get_masses())

        if self.model_energy == 'BE':
            self.e_framework_intra = 0.0
            self.e_adsorbate_intra = 0.0
        else:
            self.e_framework_intra = self.compute(self.atoms_frame)
            self.e_adsorbate_intra = self.compute(self.atoms_ads)

        self.T = T
        self.beta = 1 / (boltzmann * T)
        self.vdw = (vdw_radii - 0.35)   # modify this value to overcome the atoms overlap
        self.device = device
        self.seed = seed
        self.trajectory = []

        self.widom_e_min = 0.0
        self.snapshot_widom_e_min = []

        self.output_dir = f'results_widom_{name_frame}'
        os.makedirs(self.output_dir, exist_ok=True)

        self.snapshot_dir = os.path.join(self.output_dir, 'snapshots')
        os.makedirs(self.snapshot_dir, exist_ok=True)

        self.e_traj_dir = os.path.join(self.output_dir, 'e_traj')
        os.makedirs(self.e_traj_dir, exist_ok=True)

        self.snapshot_widom_e_min_file_name = os.path.join(self.snapshot_dir, f'widom_{name_frame}_{name_ads}_e_min.pdb')
        self.snapshot_widom_e_min_traj_file_name = os.path.join(self.snapshot_dir, f'widom_{name_frame}_{name_ads}_e_min_traj.extxyz')
        self.widom_e_traj_file_name = os.path.join(self.e_traj_dir, f'widom_{name_frame}_{name_ads}_e_traj')

        # Redirect stdout to a log file
        self.log_file_path = os.path.join(self.output_dir, f'widom_{name_frame}_{name_ads}.log')
        self.log_file = open(self.log_file_path, 'w')
        self.original_stdout = sys.stdout
        sys.stdout = self.log_file

        # Print basic info
        print(self.snapshot_widom_e_min_file_name, flush=True)
        print("\n", flush=True)
        print("FRAMEWORK NAME           :", self.name_frame, flush=True)
        print("FRAMEWORK COMPOSITION    :", self.atoms_frame.symbols, flush=True)
        print("FRAMEWORK N_ATOMS        :", self.n_frame, flush=True)
        print("FRAMEWORK MASS [me]      :", self.mass_frame, flush=True)
        print("FRAMEWORK MASS [amu]     :", self.mass_frame / amu, flush=True)
        print("FRAMEWORK CELL PARAMS    :", self.atoms_frame.cell.cellpar(), flush=True)
        print("FRAMEWORK VOLUME [bohr3] :", self.V, flush=True)
        print("FRAMEWORK VOLUME [ang3]  :", self.V / angstrom**3, flush=True)
        print("", flush=True)
        print("ADSORBATE NAME           :", self.name_ads, flush=True)
        print("ADSORBATE COMPOSITION    :", self.atoms_ads.symbols, flush=True)
        print("ADSORBATE N_ATOMS        :", self.n_ads, flush=True)
        print("ADSORBATE MASS           :", self.mass_ads, flush=True)
        print("", flush=True)
        print("FRAMEWORK ENERGY [H]     :", self.e_framework_intra, flush=True)
        print("ADSORBATE ENERGY [H]     :", self.e_adsorbate_intra, flush=True)
        print("\n", flush=True)

    def compute(self, atoms_trial):
        atoms_trial.calc = self.model
        energy = atoms_trial.get_potential_energy() * electronvolt
        return energy

    def run(self, iterations):
        if self.seed:
            np.random.seed(self.seed)
        atoms = self.atoms_frame.copy()
        energies = []

        for i in range(iterations):
            if self.n_insertion_grid > 0:
                insertion_position = self.insertion_grid[np.random.randint(self.n_insertion_grid)]
            else:
                insertion_position = np.dot(atoms.get_cell(), np.random.rand(3))

            atoms_ads_trial = self.atoms_ads.copy()
            atoms_ads_trial.set_positions(insertion_position + atoms_ads_trial.get_positions())

            phi, theta, psi = Rotation.random().as_euler("zxz", degrees=True)
            atoms_ads_trial.euler_rotate(phi=phi, theta=theta, psi=psi, center="COP")

            atoms_trial = atoms + atoms_ads_trial

            if _vdw_overlap(atoms_trial, self.vdw, len(self.atoms_ads), self.n_frame):
                e_trial = 10**10 * electronvolt
            else:
                e_trial = self.compute(atoms_trial) - self.e_framework_intra - self.e_adsorbate_intra

            if len(energies) > 0 and e_trial < np.min(energies):
                self.widom_e_min = e_trial
                self.snapshot_widom_e_min = atoms_trial.copy()
                self.snapshot_widom_e_min.calc = SinglePointCalculator(self.snapshot_widom_e_min, energy=e_trial/electronvolt)
                write(self.snapshot_widom_e_min_traj_file_name, self.snapshot_widom_e_min, append=True)

            energies.append(e_trial)

            if i != 0 and i % 1000 == 0:         # modify this if you get any 
                self._log_iteration(i, energies)

        np.save(self.widom_e_traj_file_name, np.array(energies))
        self._log_iteration(iterations, energies, final=True)
        write(self.snapshot_widom_e_min_file_name, self.snapshot_widom_e_min)

        sys.stdout = self.original_stdout
        self.log_file.close()

    def _log_iteration(self, i, energies, final=False):
        energies_temp = np.array(energies)
        exp_avg = np.mean(np.exp(-self.beta * energies_temp))
        exp_std = np.std(np.exp(-self.beta * energies_temp))
        vexp_avg = np.mean(energies_temp * np.exp(-self.beta * energies_temp))
        vexp_std = np.std(energies_temp * np.exp(-self.beta * energies_temp))

        henry = self.beta / self.rho * exp_avg
        henry_error = self.beta / self.rho * exp_std / np.sqrt(i)
        e_ads = vexp_avg / (exp_avg)
        enthalpy = e_ads - 1.0 / self.beta
        enthalpy_error = np.abs(enthalpy) / np.sqrt(i) * np.sqrt((vexp_std / vexp_avg)**2 + (exp_std / (exp_avg) )**2)

        label = 'FINAL' if final else f'{i:<10d}'
        # print(f'{label}: ADSORPTION ENERGY         : {e_ads/electronvolt:.2f} eV, {e_ads/kjmol:.2f} kJ/mol', flush=True)
        print(f'{label}: Adsorption enthalpy (Qst) : {enthalpy / kjmol:.2f} kJ/mol ', flush=True)
        print(f'{label}: Henry coefficient (KH)    : {henry / avogadro * kilogram * pascal:.5f} mol/kg/pa', flush=True)
        # print(f'{label}: Henry coefficient (KH)    : {henry / avogadro * kilogram * bar:.5f} +- {henry_error / avogadro * kilogram * bar:.5f} mol/kg/bar\n', flush=True)
