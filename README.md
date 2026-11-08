# FOCAL - Safe Offline RL [NeurIPS 2026]

**Paper: Safe Offline Reinforcement Learning using Behavior Regularisation and Latent Feasibility Guidance**

**Authors:** Mahesh Keswani, Parth Bhardwaj, Ankur Kumar, Raunak P. Bhattacharyya

**Abstract:** *Safe offline reinforcement learning (RL) seeks to learn policies that maximize return while satisfying safety constraints, using a static dataset. To characterize state-wise safety, reachability analysis identifies the feasible region, defined as the set of states from which constraints can be satisfied at every future step. Policies are commonly extracted from learned value functions with advantage-weighted regression (AWR). However, AWR relies on exponentiated weights that amplify value estimation errors, whereas behavior-regularized policy gradient methods directly utilize the action gradient of the value function and often make more effective use of learned critics. We propose Feasibility-Optimized Conditional Actor Learning (FOCAL), a method that employs behavior-regularized policy gradients into a conditional variational autoencoder (CVAE) actor. FOCAL optimizes this policy using a first-order objective partitioned by feasibility, updating the actor to increase return on transitions estimated to be feasible and to reduce constraint violations on transitions estimated to be infeasible. During inference, FOCAL dynamically adjusts the scale of the latent prior according to the estimated feasibility of the state to mitigate distribution shift. Theoretically, we show that for AWR, the policy update variance depends exponentially on the advantage error, whereas for FOCAL it scales linearly with the variance of the value estimation error. Across the DSRL benchmark, FOCAL satisfies the constraints on all evaluated tasks while achieving competitive or higher returns compared to baselines.*

## Installation

Install the [OSRL](https://github.com/liuzuxin/OSRL) package, which provides all the datasets, environments, and utilities.
```bash
git clone https://github.com/liuzuxin/OSRL.git
cd OSRL
pip install -e .
```

Next, clone this repository:
```bash
git clone https://github.com/mahesh-keswani/FOCAL-Offline-SafeRL.git
cd FOCAL-Offline-SafeRL
```

## Usage
Run the training script using:
```bash
python train_focal.py --task OfflineHalfCheetahVelocityGymnasium-v1 --project [PROJECT_NAME] --logdir [LOG_DIR] --seed [SEED]
```

The `--task` argument accepts any DSRL task name, for example `OfflineAntVelocityGymnasium-v1`, `OfflineDroneRunGymnasium-v0`, or `OfflineMetadrive-easysparse-v0`. The results in the paper are averaged over three seeds per task.

## Citation
If you find this code useful, please consider citing:
```bibtex
@inproceedings{keswani2026focal,
  title={Safe Offline Reinforcement Learning using Behavior Regularisation and Latent Feasibility Guidance},
  author={Keswani, Mahesh and Bhardwaj, Parth and Kumar, Ankur and Bhattacharyya, Raunak P.},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## Acknowledgements
Datasets: [DSRL](https://github.com/liuzuxin/DSRL)

Implementation builds on: [LSPC](https://github.com/PrajwalKoirala/LSPC-Safe-Offline-RL), [OSRL](https://github.com/liuzuxin/OSRL)
