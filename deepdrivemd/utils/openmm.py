import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import MDAnalysis

try:
    import openmm
    import openmm.app as app
    import openmm.unit as u
except ImportError:
    pass  # For testing purposes

from deepdrivemd.api import Application, PathLike


class OpenMMSimulationApplication(Application):
    """Application logic for instantiating and running OpenMM simulations."""

    def copy_topology(self, directory: Path) -> Optional[Path]:
        """Scan directory for optional topology file (assumes topology
        file is in the same directory as the PDB file and that only
        one PDB/topology file exists in each directory.)"""
        top_file = next(directory.glob("*.top"), None)
        if top_file is None:
            top_file = next(directory.glob("*.prmtop"), None)
        if top_file is not None:
            top_file = self.copy_to_workdir(top_file)
        return top_file

    def generate_restart_pdb(self, sim_dir: Path, frame: int) -> Path:
        """Generate a new PDB from a given `frame` of a previous simulation."""
        old_pdb_file = next(sim_dir.glob("*.pdb"))
        dcd_file = next(sim_dir.glob("*.dcd"))
        # New pdb file to write, example: workdir/run-<uuid>_frame000000.pdb
        pdb_file = self.workdir / f"{old_pdb_file.parent.name}_frame{frame:06}.pdb"
        mda_u = MDAnalysis.Universe(str(old_pdb_file), str(dcd_file))
        mda_u.trajectory[frame]
        mda_u.atoms.write(str(pdb_file))
        return pdb_file

    def _configure_amber_implicit(
        self,
        pdb_file: PathLike,
        top_file: Optional[PathLike],
        dt_ps: float,
        temperature_kelvin: float,
        heat_bath_friction_coef: float,
        platform: "openmm.Platform",
        platform_properties: Dict[str, str],
    ) -> Tuple["app.Simulation", Optional["app.PDBFile"]]:
        """Helper function to configure implicit amber simulations with openmm."""
        # Configure system
        if top_file is not None:
            pdb = None
            top = app.AmberPrmtopFile(str(top_file))
            system = top.createSystem(
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=1.0 * u.nanometer,
                constraints=app.HBonds,
                implicitSolvent=app.OBC1,
            )
        else:
            pdb = app.PDBFile(str(pdb_file))
            top = pdb.topology
            forcefield = app.ForceField("amber14-all.xml", "implicit/gbn2.xml")
            system = forcefield.createSystem(
                top,
                nonbondedMethod=app.CutoffNonPeriodic,
                nonbondedCutoff=1.0 * u.nanometer,
                constraints=app.HBonds,
            )

        # Configure integrator
        integrator = openmm.LangevinIntegrator(
            temperature_kelvin * u.kelvin,
            heat_bath_friction_coef / u.picosecond,
            dt_ps * u.picosecond,
        )
        integrator.setConstraintTolerance(0.00001)

        sim = app.Simulation(top, system, integrator, platform, platform_properties)

        # Returning the pdb file object for later use to reduce I/O.
        # If a topology file is passed, the pdb variable is None.
        return sim, pdb

    def _configure_amber_explicit(
        self,
        top_file: PathLike,
        dt_ps: float,
        temperature_kelvin: float,
        heat_bath_friction_coef: float,
        platform: "openmm.Platform",
        platform_properties: Dict[str, str],
        explicit_barostat: str,
    ) -> "app.Simulation":
        """Helper function to configure explicit amber simulations with openmm."""
        top = app.AmberPrmtopFile(str(top_file))
        system = top.createSystem(
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
        )

        # Congfigure integrator
        integrator = openmm.LangevinIntegrator(
            temperature_kelvin * u.kelvin,
            heat_bath_friction_coef / u.picosecond,
            dt_ps * u.picosecond,
        )

        if explicit_barostat == "MonteCarloBarostat":
            system.addForce(
                openmm.MonteCarloBarostat(1 * u.bar, temperature_kelvin * u.kelvin)
            )
        elif explicit_barostat == "MonteCarloAnisotropicBarostat":
            system.addForce(
                openmm.MonteCarloAnisotropicBarostat(
                    (1, 1, 1) * u.bar, temperature_kelvin * u.kelvin, False, False, True
                )
            )
        else:
            raise ValueError(f"Invalid explicit_barostat option: {explicit_barostat}")

        sim = app.Simulation(
            top.topology, system, integrator, platform, platform_properties
        )

        return sim

    def configure_simulation(
        self,
        pdb_file: PathLike,
        top_file: Optional[PathLike],
        solvent_type: str,
        gpu_index: int,
        dt_ps: float,
        temperature_kelvin: float,
        heat_bath_friction_coef: float,
        explicit_barostat: str = "MonteCarloBarostat",
        run_minimization: bool = True,
        set_positions: bool = True,
        set_velocities: bool = False,
    ) -> "app.Simulation":
        """Configure an OpenMM amber simulation.
        Parameters
        ----------
        pdb_file : PathLike
            The PDB file to initialize the positions (and topology if
            `top_file` is not present and the `solvent_type` is `implicit`).
        top_file : Optional[PathLike]
            The topology file to initialize the systems topology.
        solvent_type : str
            Solvent type can be either `implicit` or `explicit`, if `explicit`
            then `top_file` must be present.
        gpu_index : int
            The GPU index to use for the simulation.
        dt_ps : float
            The timestep to use for the simulation.
        temperature_kelvin : float
            The temperature to use for the simulation.
        heat_bath_friction_coef : float
            The heat bath friction coefficient to use for the simulation.
        explicit_barostat : str, optional
            The barostat used for an `explicit` solvent simulation can be either
            "MonteCarloBarostat" by deafult, or "MonteCarloAnisotropicBarostat".
        run_minimization : bool, optional
            Whether or not to run energy minimization, by default True.
        set_positions : bool, optional
            Whether or not to set positions (Loads the PDB file), by default True.
        set_velocities : bool, optional
            Whether or not to set velocities to temperature, by default True.
        Returns
        -------
        app.Simulation
            Configured OpenMM Simulation object.
        """
        # Configure hardware
        try:
            platform = openmm.Platform.getPlatformByName("CUDA")
            platform_properties = {
                "DeviceIndex": str(gpu_index),
                "CudaPrecision": "mixed",
            }
        except Exception:
            try:
                platform = openmm.Platform.getPlatformByName("OpenCL")
                platform_properties = {"DeviceIndex": str(gpu_index)}
            except Exception:
                platform = openmm.Platform.getPlatformByName("CPU")
                platform_properties = {}

        # Select implicit or explicit solvent configuration
        if solvent_type == "implicit":
            sim, pdb = self._configure_amber_implicit(
                pdb_file,
                top_file,
                dt_ps,
                temperature_kelvin,
                heat_bath_friction_coef,
                platform,
                platform_properties,
            )
        else:
            assert solvent_type == "explicit"
            assert top_file is not None
            pdb = None
            sim = self._configure_amber_explicit(
                top_file,
                dt_ps,
                temperature_kelvin,
                heat_bath_friction_coef,
                platform,
                platform_properties,
                explicit_barostat,
            )

        # Set the positions
        if set_positions:
            if pdb is None:
                pdb = app.PDBFile(str(pdb_file))
            sim.context.setPositions(pdb.getPositions())

        # Minimize energy and equilibrate
        if run_minimization:
            sim.minimizeEnergy()

        # Set velocities to temperature
        if set_velocities:
            sim.context.setVelocitiesToTemperature(
                temperature_kelvin * u.kelvin, random.randint(1, 10000)
            )

        return sim
