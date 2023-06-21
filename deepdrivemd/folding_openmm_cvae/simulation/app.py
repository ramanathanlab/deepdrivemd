from pathlib import Path
from typing import Tuple

import MDAnalysis
import numpy as np
import numpy.typing as npt

try:
    import openmm.app as app
    import openmm.unit as u
except ImportError:
    pass  # For testing purposes

from MDAnalysis.analysis import align, distances, rms

from deepdrivemd.apps.openmm_simulation import (
    MDSimulationInput,
    MDSimulationOutput,
    MDSimulationSettings,
)
from deepdrivemd.utils.openmm import OpenMMSimulationApplication

# TODO: A more efficient (but complex) implementation could background the
# contact map and RMSD computation using openmm reporters using a process pool.
# This would overlap the simulations and analysis so they finish at roughly
# the same time.


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

        # Analyze simulation and collect contact maps and RMSD to native state
        contact_maps, rmsds = self.analyze_simulation(pdb_file, traj_file)

        # Save simulation analysis
        np.save(self.workdir / "contact_map.npy", contact_maps)
        np.save(self.workdir / "rmsd.npy", rmsds)

        # Return simulation analysis outputs
        output_data = MDSimulationOutput(
            contact_map_path=self.persistent_dir / "contact_map.npy",
            rmsd_path=self.persistent_dir / "rmsd.npy",
        )

        # Log the output data
        output_data.dump_yaml(self.workdir / "output.yaml")
        self.backup_node_local()

        return output_data

    def analyze_simulation(
        self, pdb_file: Path, traj_file: Path
    ) -> Tuple["npt.ArrayLike", "npt.ArrayLike"]:
        """Analyze trajectory and return a contact map and
        the RMSD to native state for each frame."""

        # Compute contact maps, rmsd, etc in bulk
        mda_u = MDAnalysis.Universe(str(pdb_file), str(traj_file))
        ref_u = MDAnalysis.Universe(str(self.config.rmsd_reference_pdb))
        # Align trajectory to compute accurate RMSD
        align.AlignTraj(
            mda_u, ref_u, select=self.config.mda_selection, in_memory=True
        ).run()
        # Get atomic coordinates of reference atoms
        ref_positions = ref_u.select_atoms(self.config.mda_selection).positions.copy()
        atoms = mda_u.select_atoms(self.config.mda_selection)
        box = mda_u.atoms.dimensions
        rows, cols, rmsds = [], [], []
        for _ in mda_u.trajectory:
            positions = atoms.positions
            # Compute contact map of current frame (scipy lil_matrix form)
            cm = distances.contact_matrix(
                positions, self.config.cutoff_angstrom, box=box, returntype="sparse"
            )
            coo = cm.tocoo()
            rows.append(coo.row.astype("int16"))
            cols.append(coo.col.astype("int16"))

            # Compute RMSD
            rmsd = rms.rmsd(positions, ref_positions, center=True, superposition=True)
            rmsds.append(rmsd)

        # Save simulation analysis results
        contact_maps = np.array(
            [np.concatenate(row_col) for row_col in zip(rows, cols)], dtype=object
        )

        return contact_maps, rmsds
