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
python -m deepdrivemd.folding_openmm_cvae.run -c tests/apps-enabled-workstation/test.yaml
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
$ ls runs/experiment-170323-091525/simulation/run-08843adb-65e1-47f0-b0f8-34821aa45923:
1FME-unfolded.pdb  contact_map.npy  input.yaml  output.yaml  rmsd.npy  sim.dcd  sim.log
```
- `1FME-unfolded.pdb` the PDB file used to start the simulation
- `contact_map.npy`, `rmsd.npy`: the preprocessed data files which will be input into the train and inference tasks
- `input.yaml`, `output.yaml`: These simply log the task function input and return values, they are helpful for debugging but are not strtictly necessary
- `sim.dcd`: the simulation trajectory file containing all the coordinate frames
- `sim.log`: a simulation log detailing the energy, steps taken, ns/day, etc

By default the `runs/` directory is ignored by git.

Production runs can be configured and run analogously. See `examples/bba-folding-workstation/` for a detailed example of folding the [1FME](https://www.rcsb.org/structure/1FME) protein. **The YAML files document the configuration settings and explain the use case**.

### Software Interface

Implement a DeepDriveMD workflow with custom MD simulation engines, and AI training/inference methods by inherting from the `DeepDriveMDWorkflow` interface. This workflow implments the `examples/bba-folding-workstation/` example:
```python
from deepdrivemd.api import DeepDriveMDWorkflow

class DeepDriveMD_OpenMM_CVAE(DeepDriveMDWorkflow):
    def __init__(
        self, simulations_per_train: int, simulations_per_inference: int, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.simulations_per_train = simulations_per_train
        self.simulations_per_inference = simulations_per_inference

        # Make sure there has been at least one training task 
        # complete before running inference
        self.model_weights_available: bool = False

        # For batching training/inference inputs
        self.train_input = CVAETrainInput(contact_map_paths=[], rmsd_paths=[])
        self.inference_input = CVAEInferenceInput(
            contact_map_paths=[], rmsd_paths=[], model_weight_path=Path()
        )

        # Communicate results between agents
        self.simulation_input_queue: Queue[MDSimulationInput] = Queue()

    def simulate(self) -> None:
        """Submit either a new outlier to simulate, or a starting conformer."""
        with self.simulation_govenor:
            if not self.simulation_input_queue.empty():
                inputs = self.simulation_input_queue.get()
            else:
                inputs = MDSimulationInput(sim_dir=next(self.simulation_input_dirs))

        self.submit_task("simulation", inputs)

    def train(self) -> None:
        """Submit a new training task."""
        self.submit_task("train", self.train_input)

    def inference(self) -> None:
        """Submit a new inference task once model weights are available."""
        while not self.model_weights_available:
            time.sleep(1)

        self.submit_task("inference", self.inference_input)

    def handle_simulation_output(self, output: MDSimulationOutput) -> None:
        """When a simulation finishes, decide to train a new model or infer outliers."""
        # Collect simulation results
        self.train_input.append(output.contact_map_path, output.rmsd_path)
        self.inference_input.append(output.contact_map_path, output.rmsd_path)

        # Signal train/inference tasks
        num_sims = len(self.train_input)
        if num_sims % self.simulations_per_train == 0:
            self.run_training.set()

        if num_sims % self.simulations_per_inference == 0:
            self.run_inference.set()

    def handle_train_output(self, output: CVAETrainOutput) -> None:
        """When training finishes, update the model weights to use for inference."""
        self.inference_input.model_weight_path = output.model_weight_path
        self.model_weights_available = True

    def handle_inference_output(self, output: CVAEInferenceOutput) -> None:
        """When inference finishes, update the simulation queue with the latest outliers."""
        with self.simulation_govenor:
            self.simulation_input_queue.queue.clear() # Remove old outliers
            for sim_dir, sim_frame in zip(output.sim_dirs, output.sim_frames):
                self.simulation_input_queue.put(
                    MDSimulationInput(sim_dir=sim_dir, sim_frame=sim_frame)
                )
```

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

