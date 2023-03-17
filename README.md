# DeepDriveMD: Coupling streaming AI and HPC ensembles to achieve 100-1000× faster biomolecular simulations
[DeepDriveMD](https://github.com/DeepDriveMD/DeepDriveMD-pipeline) implemented using [Colmena](https://colmena.readthedocs.io/en/latest/).

The computational motif implemented by DeepDriveMD to support ML/AI-coupled simulations comprises four stages. _Simulation_: Simulations are used to explore possible trajectories of a protein or other biomolecular system; _Aggregation_: Simulation results are preprocessed for training. _Training_: Aggregated trajectories are used to train one or more ML models. _Inference_: Trained ML models are used to identify conformations for subsequent iterations of simulations. 

<img src="https://user-images.githubusercontent.com/38300604/205099612-e856d68b-a51b-4f92-acdc-240b229f015c.png" width="530" height="400"/>


## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Contributing](#contributing)
4. [License](#license)
5. [Citations](#citations)

## Installation

Create a conda environment
```console
conda create -n deepdrivemd python=3.9 -y
conda activate deepdrivemd
```

To install OpenMM for simulations:
```console
conda install -c conda-forge gcc=12.1.0 -y
conda install -c conda-forge openmm -y
```

To install `deepdrivemd`:
```console
git clone https://github.com/ramanathanlab/deepdrivemd.git
cd deepdrivemd
make install
```

## Usage

The workflow can be tested on a workstation (a system with a few GPUs) via:
```console
python -m deepdrivemd.workflows.openmm_cvae -c tests/apps-enabled-workstation/test.yaml
```
This will generate an output directory for the run with logs, results, and task specific output folders.

Each test will write a timestamped experiment output directory to the `runs/` directory.

Inside the output directory, you will find:
```console
$ ls runs/experiment-170323-091525/
inference  params.yaml  result  run-info  runtime.log  simulation  train
```
- `params.yaml`: the full configuration file (default parameters included)
- `runtime.log`: the workflow log
- `result`: a directory containing JSON files `simulation.json`, `train.json`, `inference.json` which log task results including success or failure, potential error messages, runtime statistics. This can be helpful for debugging application-level failures.
- `simulation`, `train`, `inference`: output directories each containing subdirectories `run-<uuid>` for each submitted task. This is where the output files of your simulations, preprocessed data, model weights, etc will be written by your applications (it corresponds to the application workdir).
- `run-info`: Parsl logs

An example, the simulation run directories may look like:
```console
ls runs/experiment-170323-091525/simulation/run-08843adb-65e1-47f0-b0f8-34821aa45923:
1FME-unfolded.pdb  contact_map.npy  input.yaml  output.yaml  rmsd.npy  sim.dcd  sim.log
```
- `1FME-unfolded.pdb` the PDB file used to start the simulation
- `contact_map.npy`, `rmsd.npy`: the preprocessed data files which will be input into the train and inference tasks
- `input.yaml`, `output.yaml`: These simply log the task function input and return values, they are helpful for debugging but are not strtictly necessary
- `sim.dcd`: the simulation trajectory file containing all the coordinate frames
- `sim.log`: a simulation log detailing the energy, steps taken, ns/day, etc

By default the `runs/` directory is ignored by git.

Production runs can be configured and run analogously. See `examples/bba-folding-workstation/` for a detailed example of folding the [1FME](https://www.rcsb.org/structure/1FME) protein. **The YAML files document the configuration settings and explain the use case**.


## Contributing

Please report **bugs**, **enhancement requests**, or **questions** through the [Issue Tracker](https://github.com/ramanathanlab/deepdrivemd/issues).

If you are looking to contribute, please see [`CONTRIBUTING.md`](https://github.com/ramanathanlab/deepdrivemd/blob/main/CONTRIBUTING.md).

## License

DeepDriveMD has a MIT license, as seen in the [`LICENSE.md`](https://github.com/ramanathanlab/deepdrivemd/blob/main/LICENSE.md) file.

## Citations

If you use DeepDriveMD in your research, please cite this paper:

```bibtex
@inproceedings{brace2022coupling,
  title={Coupling streaming ai and hpc ensembles to achieve 100--1000$\times$ faster biomolecular simulations},
  author={Brace, Alexander and Yakushin, Igor and Ma, Heng and Trifan, Anda and Munson, Todd and Foster, Ian and Ramanathan, Arvind and Lee, Hyungro and Turilli, Matteo and Jha, Shantenu},
  booktitle={2022 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  pages={806--816},
  year={2022},
  organization={IEEE}
}
```

