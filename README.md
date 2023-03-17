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

The workflow can be tested on a workstation via:
```console
python -m deepdrivemd.workflows.openmm_cvae -c tests/apps-enabled-workstation/test.yaml
```
This will generate an output directory for the run with logs, results, and task specific output folders.

Each test will write a timestamped run directory to the `runs/` directory specified in `tests/basic-local/test.yaml`.

To clean up the runs (by default these are ignored by git):
```console
rm -r runs/
```

Production runs can be configured and run analagously. See `examples/bba-folding-workstation/` for an example of folding the [1FME](https://www.rcsb.org/structure/1FME) protein.

**Note**: Mock testing is specified in each of the application scripts `deepdrivemd/applications/*/app.py`.


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

