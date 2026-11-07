import os
import types
from dataclasses import asdict

import bullet_safety_gym  # noqa: F401  (registers Bullet-Safety-Gym tasks)
import dsrl  # noqa: F401  (registers DSRL tasks)
import numpy as np
import pyrallis
from dsrl.infos import DENSITY_CFG
from dsrl.offline_env import OfflineEnvWrapper, wrap_env
from fsrl.utils import WandbLogger
from osrl.common.exp_util import auto_name, seed_all
from tqdm.auto import trange

from focal_configs import BC_DEFAULT_CONFIG, BCTrainConfig
from focal_iql import FOCAL_IQL_kwargs, FOCAL_IQL_Trainer, ReplayBuffer


@pyrallis.wrap()
def train(args: BCTrainConfig):
    # Task-specific defaults, overridden by any values passed on the command line
    cfg, old_cfg = asdict(args), asdict(BCTrainConfig())
    differing_values = {key: cfg[key] for key in cfg.keys() if cfg[key] != old_cfg[key]}
    default_cfg = asdict(BC_DEFAULT_CONFIG[args.task]())
    cfg = {**default_cfg, **differing_values}
    args = types.SimpleNamespace(**cfg)

    # Logger
    if args.name is None:
        args.name = auto_name(default_cfg, cfg, args.prefix, args.suffix) + f"_focal_{args.task}"
    if args.group is None:
        args.group = args.task + "-cost-" + str(int(args.cost_limit))
    if args.logdir is not None:
        args.logdir = os.path.join(args.logdir, args.group, args.name)
    logger = WandbLogger(cfg, args.project, args.group, args.name, args.logdir)
    logger.save_config(cfg, verbose=args.verbose)

    seed_all(args.seed)

    # Environment
    if "Metadrive" in args.task:
        import gym
    else:
        import gymnasium as gym
    env = gym.make(args.task, disable_env_checker=True)

    # Offline dataset
    data = env.get_dataset()
    env.set_target_cost(args.cost_limit)

    cbins, rbins, max_npb, min_npb = None, None, None, None
    if args.density != 1.0:
        density_cfg = DENSITY_CFG[args.task + "_density" + str(args.density)]
        cbins = density_cfg["cbins"]
        rbins = density_cfg["rbins"]
        max_npb = density_cfg["max_npb"]
        min_npb = density_cfg["min_npb"]
    data = env.pre_process_data(
        data,
        args.outliers_percent,
        args.noise_scale,
        args.inpaint_ranges,
        args.epsilon,
        args.density,
        cbins=cbins,
        rbins=rbins,
        max_npb=max_npb,
        min_npb=min_npb,
    )

    env = wrap_env(env=env, reward_scale=args.reward_scale)
    env = OfflineEnvWrapper(env)

    # Model and optimizers
    args_dict = vars(args)
    args_dict["state_dim"] = env.observation_space.shape[0]
    args_dict["action_dim"] = env.action_space.shape[0]

    trainer = FOCAL_IQL_Trainer(
        **FOCAL_IQL_kwargs(args_dict),
        iql_tau=args.iql_tau,
        cost_iql_tau=args.cost_iql_tau,
        discount=args.discount,
        tau=args.tau,
        hazard_unsafe=args.hazard_unsafe,
        warmup_steps=args.warmup_steps,
        alpha_r=args.alpha_r,
        alpha_h=args.alpha_h,
        n_samples=args.n_samples,
        keep_case3=args.keep_case3,
        use_awr=args.use_awr,
        beta_reward=args.beta_reward,
        beta_cost=args.beta_cost,
        awr_max_weight_reward=args.awr_max_weight_reward,
        awr_max_weight_cost=args.awr_max_weight_cost,
        logger=logger,
        episode_len=args.episode_len,
    )

    print(f"Total actor parameters: {sum(p.numel() for p in trainer.actor.parameters())}")
    print(f"Total qf parameters: {sum(p.numel() for p in trainer.qf.parameters())}")
    print(f"Total vf parameters: {sum(p.numel() for p in trainer.vf.parameters())}")

    logger.setup_checkpoint_fn(lambda: {"model_state": trainer.state_dict()})

    # Replay buffer
    dataset_size = len(data["observations"])
    buffer = ReplayBuffer(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        buffer_size=dataset_size,
        device=args.device,
    )
    buffer.load_d4rl_dataset(data)
    logger.store(tab="train", dataset_size=dataset_size)

    # Training
    best_reward, best_cost, best_idx = -np.inf, np.inf, 0
    for step in trange(args.max_timesteps, desc="Training"):
        observations, actions, rewards, costs, next_observations, dones = buffer.sample(args.batch_size)
        trainer.train_one_step(observations, next_observations, actions, rewards, costs, dones)

        # Evaluation
        if (step + 1) % args.eval_every == 0 or step == args.max_timesteps - 1 or step == 0:
            ret, cost, length, ret_std, cost_std, length_std = trainer.evaluate(env, args.eval_episodes)
            logger.store(tab="eval", Cost=cost, Reward=ret, Length=length)
            logger.store(tab="eval", Cost_std=cost_std, Reward_std=ret_std, Length_std=length_std)
            normalized_ret, normalized_cost = env.get_normalized_score(ret, cost)
            logger.store(tab="normalized_eval", Normalized_Cost=normalized_cost, Normalized_Reward=normalized_ret)

            logger.save_checkpoint()
            if (cost < best_cost) or (cost == best_cost and ret > best_reward):
                logger.save_checkpoint(suffix=f"best_{int(cost)}_{int(ret)}")
                best_cost = cost
            if (cost < args.cost_limit) and (ret > best_reward):
                best_reward = ret
                best_idx = step
                logger.save_checkpoint(suffix="best")

            logger.store(tab="train", best_idx=best_idx)

        logger.write(step, display=False)


if __name__ == "__main__":
    train()
