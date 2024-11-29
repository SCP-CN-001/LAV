# LAV

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-from-all-vehicles/autonomous-driving-on-carla-leaderboard)](https://paperswithcode.com/sota/autonomous-driving-on-carla-leaderboard?p=learning-from-all-vehicles)

LAV achieved 5th place on the CARLA Leaderboard 1.0, demonstrating the effectiveness of a state-of-the-art, end-to-end imitation learning-based approach. This repository offers a migrated version of LAV for CARLA Leaderboard 2.0, which is part of the benchmark suite for [Ramble](https://github.com/SCP-CN-001/ramble). The original repository can be found [here](git@github.com:dotchen/LAV.git).

The code from the original repository has been refactored for better readability. The previous directory structure, which included `team_code/` and `team_code_v2/`, has now been consolidated into a single `team_code/` folder to reduce confusion. Additionally, we have cleaned up the training scripts in `lav/`, removing the `_v2` suffix from certain files and updating the dependencies. Since the model is trained offline using expert-generated data, the training scripts are compatible with both Leaderboard 1.0 and Leaderboard 2.0 data.

## Reference

If you find our repo, dataset or paper useful, please cite us and the original LAV paper:

```bibtex
@article{li2024end,
  title={End-to-end Driving in High-Interaction Traffic Scenarios with Reinforcement Learning},
  author={Li, Yueyuan and Jiang, Mingyang and Zhang, Songan and Yuan, Wei and Wang, Chunxiang and Yang, Ming},
  journal={arXiv preprint arXiv:2410.02253},
  year={2024}
}

@inproceedings{chen2022learning,
  title={Learning from all vehicles},
  author={Chen, Dian and Kr{\"a}henb{\"u}hl, Philipp},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={17222--17231},
  year={2022}
}
```

## Demo Video
[![Demo](https://img.youtube.com/vi/-TlxbmSQ7rQ/0.jpg)](https://www.youtube.com/watch?v=-TlxbmSQ7rQ)

Also checkout [website](https://dotchen.github.io/LAV/)!

## Reprodce results on CARLA Leaderboard 1.0

> To run CARLA and train the models, make sure you are using a machine with **at least** a mid-end GPU.

### Installation

1. Download and unzip [CARLA 0.9.10.1](https://github.com/carla-simulator/carla/releases/tag/0.9.10.1) to somewhere out of the repository.
2. Install [git lfs](https://git-lfs.github.com/).
    ```shell
    sudo apt-get install git-lfs
    ```
3. Clone the repository with submodules. Check if the weight files under `weights/` are downloaded correctly.
    ```shell
    git clone --recurse-submodules git@github.com:SCP-CN-001/LAV.git
    ```
4. Create a soft link to the CARLA folder in the repository root:
    ```shell
    ln -s /path/to/carla ./CARLA_Leaderboard_10
    ```

### Install dependencies

1. Create a dedicated conda environment. Refer [here](https://www.anaconda.com/products/individual#Downloads) if you do not have conda.
    ```shell
    conda env create -f environment.yaml
    ```
2. Install [PyTorch](https://pytorch.org/get-started/locally/)
3. Install [torch-scatter](https://github.com/rusty1s/pytorch_scatter) based on your `CUDA` and `PyTorch` versions.
4. Setup [wandb](https://docs.wandb.ai/quickstart)

---

> TODO: update the old documentation to the new one

## Data Collection
The data collection scripts reside in the `data-collect` branch of the repo.
It will log dataset to the path specified in `config.yaml`.
Specify the number of runners and towns you would like to collect in `data_collect.py`.
The script supports parallel data collection with `ray`.
You can view wandb for the routes that are being collected. Example: https://wandb.ai/trilobita/lav_data?workspace=user-trilobita

```bash
python data_collect.py --num-runners=8
```

## Training
We adopt a LBC-style staged privileged distillation framework.
Please refer to [TRAINING.md](docs/TRAINING.md) for more details.

## Evaluation
We additionally provide examplery trained weights in the `weights` folder if you would like to directly evaluate.
They are trained on Town01, 03, 04, 06.
Make sure you are launching CARLA with the `-vulkan` flag.

The agent file for the leaderboard submission is contained in `team_code_v2`.

We additionally provide a faster version of our agent that uses `torch.jit` and moves several CPU-heavy computation (point painting etc.) to GPU.
This code resides in `team_code_v2/lav_agent_fast.py`. It will also logs visualization to the `wandb` cloud which you can optionally view and debug.

**Known issues for the fast agent**:
* Since the torchscript trace file is generated using `pytorch==1.7.1`, it might be incompatible with later pytorch versions. Please refer to #23 for more details and how to regenerate the trace files locally. The amount of acceleration is also dependent on hardware platform.


![image](https://user-images.githubusercontent.com/10444308/189553598-bc688742-02fe-4e6a-8e92-b64760eadfa9.png)


Inside the root LAV repo, run
```bash
ROUTES=[PATH TO ROUTES] ./leaderboard/scripts/run_evaluation.sh
```
Use `ROUTES=assets/routes_lav_valid.xml` to run our ablation routes, or `ROUTES=leaderboard/data/routes_valid.xml` for the validation routes provided by leaderboard.
You can also try `ROUTES=assets/routes_lav_train.xml` to test on some harder training routes.

## Dataset
We also release our LAV dataset. Download the dataset [HERE](https://utexas.box.com/s/evo96v5md4r8nooma3z17kcnfjzp2wed).

See [TRAINING.md](docs/TRAINING.md) for more details.

## Acknowledgements
We thank Tianwei Yin for the pillar generation code.
The ERFNet codes are taken from the official ERFNet repo.

## License
This repo is released under the Apache 2.0 License (please refer to the LICENSE file for details).
