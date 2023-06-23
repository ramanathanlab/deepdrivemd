from pathlib import Path
from typing import Tuple

import MDAnalysis
import numpy as np
import numpy.typing as npt
import pandas as pd
import parmed as pmd

try:
    import openmm.app as app
    import openmm.unit as u
except ImportError:
    pass  # For testing purposes

from MDAnalysis.analysis import distances

from deepdrivemd.ligand_unbinding_openmm_cvae.simulation import (
    MDSimulationInput,
    MDSimulationOutput,
    MDSimulationSettings,
)
from deepdrivemd.utils.openmm import OpenMMSimulationApplication


def get_force_LJ_atomgroup(atmgrp1, atmgrp2, dists) -> float:
    # get parameters
    sigma_i = np.array([atom.sigma for atom in atmgrp1])
    sigma_j = np.array([atom.sigma for atom in atmgrp2])
    eps_i = np.array([atom.epsilon for atom in atmgrp1])
    eps_j = np.array([atom.epsilon for atom in atmgrp2])
    # mesh parameters
    sigma_i, sigma_j = np.meshgrid(sigma_i, sigma_j, sparse=True)
    eps_i, eps_j = np.meshgrid(eps_i, eps_j, sparse=True)
    # combination
    sigma_ij = (sigma_i + sigma_j) / 2
    eps_ij = (eps_i * eps_j) ** 0.5
    c_ij = sigma_ij / dists
    v_lj = 4 * eps_ij * (c_ij**12 - c_ij**6)
    return np.sum(v_lj)


def get_force_Coul_atomgroup(atmgrp1, atmgrp2, dists) -> float:
    f = 139.935485
    # get parameters
    q_i = np.array([atom.charge for atom in atmgrp1])
    q_j = np.array([atom.charge for atom in atmgrp2])
    # mesh parameters
    q_i, q_j = np.meshgrid(q_i, q_j, sparse=True)
    # combination
    q_ij = q_i * q_j
    v_coul = f * q_ij / dists
    return np.sum(v_coul)


class MDSimulationApplication(OpenMMSimulationApplication):
    config: MDSimulationSettings

    def run(self, input_data: MDSimulationInput) -> MDSimulationOutput:
        # Log the input data
        input_data.dump_yaml(self.workdir / "input.yaml")

        if input_data.sim_frame is None:
            # No restart point, starting from initial PDB
            pdb_file = next(input_data.sim_dir.glob("*.pdb"))
            pdb_file = self.copy_to_workdir(pdb_file)
            assert pdb_file is not None
        else:
            # Collect PDB, DCD, and topology files from previous simulation
            pdb_file = self.generate_restart_pdb(
                input_data.sim_dir, input_data.sim_frame
            )

        # Collect an optional topology file
        top_file = self.copy_topology(input_data.sim_dir)
        assert top_file is not None

        # Initialize an OpenMM simulation
        sim = self.configure_simulation(
            pdb_file=pdb_file,
            top_file=top_file,
            solvent_type=self.config.solvent_type,
            gpu_index=0,
            dt_ps=self.config.dt_ps,
            temperature_kelvin=self.config.temperature_kelvin,
            heat_bath_friction_coef=self.config.heat_bath_friction_coef,
        )

        # openmm typed variables
        dt_ps = self.config.dt_ps * u.picoseconds
        report_interval_ps = self.config.report_interval_ps * u.picoseconds
        simulation_length_ns = self.config.simulation_length_ns * u.nanoseconds

        # Steps between reporting DCD frames and logs
        report_steps = int(report_interval_ps / dt_ps)
        # Number of steps to run each simulation
        nsteps = int(simulation_length_ns / dt_ps)

        # Set up reporters to write simulation trajectory file and logs
        traj_file = self.workdir / "sim.dcd"
        sim.reporters.append(app.DCDReporter(traj_file, report_steps))
        sim.reporters.append(
            app.StateDataReporter(
                str(self.workdir / "sim.log"),
                report_steps,
                step=True,
                time=True,
                speed=True,
                potentialEnergy=True,
                temperature=True,
                totalEnergy=True,
            )
        )

        # Run simulation
        sim.step(nsteps)

        # Analyze simulation and collect contact maps and PLC interaction energy
        contact_maps, energy_df = self.analyze_simulation(pdb_file, top_file, traj_file)

        # Save simulation analysis
        np.save(self.workdir / "contact_map.npy", contact_maps)
        energy_df.to_csv(self.workdir / "energy.csv")

        # Return simulation analysis outputs
        output_data = MDSimulationOutput(
            contact_map_path=self.persistent_dir / "contact_map.npy",
            energy_path=self.persistent_dir / "energy.csv",
        )

        # Log the output data
        output_data.dump_yaml(self.workdir / "output.yaml")
        self.backup_node_local()

        return output_data

    def analyze_simulation(
        self, pdb_file: Path, top_file: Path, traj_file: Path
    ) -> Tuple["npt.ArrayLike", pd.DataFrame]:
        mda_u = MDAnalysis.Universe(str(pdb_file), str(traj_file))
        top = pmd.load_file(str(top_file), xyz=str(pdb_file))

        # Setup energy calculation
        protein_atoms = mda_u.select_atoms(self.config.protein_selection)
        ligand_atoms = mda_u.select_atoms(self.config.ligand_selection)
        protein_top = [top.atoms[i] for i in protein_atoms.indices]
        ligand_top = [top.atoms[i] for i in ligand_atoms.indices]

        # Setup contact map calculation
        contact_atoms = mda_u.select_atoms(self.config.contact_selection)
        box = mda_u.atoms.dimensions

        rows, cols, energies = [], [], []
        for ts in mda_u.trajectory:
            # Compute contact map of current frame (scipy lil_matrix form)
            cm = distances.contact_matrix(
                contact_atoms.positions,
                self.config.cutoff_angstrom,
                box=box,
                returntype="sparse",
            )
            coo = cm.tocoo()
            rows.append(coo.row.astype("int16"))
            cols.append(coo.col.astype("int16"))

            # Compute energies
            dist_map = distances.distance_array(
                ligand_atoms.positions, protein_atoms.positions, box=ts.dimensions
            )
            v_lj = get_force_LJ_atomgroup(protein_top, ligand_top, dist_map)
            v_coul = get_force_Coul_atomgroup(protein_top, ligand_top, dist_map)

            energies.append(
                {
                    "frame": ts.frame,
                    "V_LJ": v_lj,
                    "V_coul": v_coul,
                    "V_total": v_lj + v_coul,
                }
            )

        # Save simulation analysis results
        energy_df = pd.DataFrame(energies)
        contact_maps = np.array(
            [np.concatenate(row_col) for row_col in zip(rows, cols)], dtype=object
        )

        return contact_maps, energy_df
