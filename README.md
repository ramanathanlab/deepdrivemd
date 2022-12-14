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

To install `deepdrivemd`:
```bash
conda create -n deepdrivemd python=3.9 -y
pip install git+https://github.com/ramanathanlab/deepdrivemd.git
```

To install OpenMM for simulations:
```bash
conda install -c conda-forge openmm -y
```

## Usage

The workflow can be tested locally using mock API's for the tasks by running:
```bash
python -m deepdrivemd.workflows.openmm_cvae --test -c tests/basic-local/test.yaml
```
This will generate an output directory for the run with logs, results, and task specific output folders.

Each test will write a timestamped run directory to the `runs/` directory specified in `tests/basic-local/test.yaml`.

To clean up the runs (by default these are ignored by git):
```bash
rm -r runs/
```

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

