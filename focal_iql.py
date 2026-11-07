import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fsrl.utils import DummyLogger, WandbLogger
from tqdm.auto import trange

TensorBatch = List[torch.Tensor]


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    """Expectile loss used for the reward value function V_r."""
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, buffer_size: int, device: str = "cpu"):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0
        self._device = device

        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._costs = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        """Loads a dataset in D4RL/DSRL format, i.e. Dict[str, np.ndarray]."""
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._costs[:n_transitions] = self._to_tensor(data["costs"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        # Episode ends (termination or time limit) are treated as terminal transitions.
        self._dones[:n_transitions] = (
            self._to_tensor(data["terminals"][..., None]) + self._to_tensor(data["timeouts"][..., None])
        )
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)
        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = torch.randint(0, self._size, (batch_size,))
        return [
            self._states[indices],
            self._actions[indices],
            self._rewards[indices],
            self._costs[indices],
            self._next_states[indices],
            self._dones[indices],
        ]


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        if len(dims) < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
            if dropout is not None:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VAE_policy(nn.Module):
    """CVAE actor: encoder q(z | s, a) and decoder D(s, z)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        beta: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__()
        # Encoder
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)
        self.log_std = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.d1 = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.d2 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Linear(hidden_dim, action_dim)

        self.act_lim = max_action
        self.latent_dim = latent_dim
        self.beta = beta
        self.device = device

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        z = F.relu(self.e1(torch.cat([obs, act], 1)))
        z = F.relu(self.e2(z))
        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        u = self.decode(obs, z)
        return u, mean, std

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        a = F.relu(self.d1(torch.cat([obs, z], 1)))
        a = F.relu(self.d2(a))
        return self.act_lim * torch.tanh(self.d3(a))

    def vae_kl_loss(self, state: torch.Tensor, action: torch.Tensor):
        """Returns the per-sample (beta-weighted) KL term and the reconstructed actions."""
        recon, mean, std = self(state, action)
        kl_loss = -self.beta * 0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean(dim=1)
        return kl_loss, recon


class SafeActor(nn.Module):
    """FOCAL actor: CVAE policy with feasibility-guided candidate selection at inference."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        max_action: float = 1.0,
        latent_dim: int = 32,
        vae_beta: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__()
        self.actor = VAE_policy(state_dim, action_dim, max_action, hidden_dim, latent_dim, vae_beta, device=device)
        self.device = device

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        q_network_reward: nn.Module,
        q_network_cost: nn.Module,
        cost_v_network: nn.Module,
        device: str = "cpu",
        n_samples: int = 16,
        sigma_min: float = 0.1,
        latent_clip: float = 0.5,
    ) -> np.ndarray:
        self.eval()
        state_tensor = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)

        # Adaptive latent scale: sigma_z(s) = clip(-V_h(s), sigma_min, 1)
        v_h = cost_v_network(state_tensor).item()
        sigma_z = np.clip(-v_h, sigma_min, 1.0)

        # Sample N latent candidates from the scaled prior (truncated to [-latent_clip, latent_clip])
        z = torch.randn((n_samples, self.actor.latent_dim), device=device) * sigma_z
        z = torch.clamp(z, -latent_clip, latent_clip)

        expanded_states = state_tensor.expand(n_samples, -1)
        decoded_actions = self.actor.decode(expanded_states, z)

        q_h_vals = torch.max(*q_network_cost.both(expanded_states, decoded_actions))
        q_r_vals = torch.min(*q_network_reward.both(expanded_states, decoded_actions))

        # Candidate filtering: highest Q_r among candidates with Q_h <= 0,
        # otherwise the candidate with the lowest Q_h.
        safe_mask = q_h_vals <= 0.0
        if safe_mask.any():
            safe_q_r = q_r_vals.clone()
            safe_q_r[~safe_mask] = -float("inf")
            best_idx = torch.argmax(safe_q_r)
        else:
            best_idx = torch.argmin(q_h_vals)

        best_action = decoded_actions[best_idx].cpu().data.numpy().reshape(1, -1)
        self.train()
        return best_action


class TwinQ(nn.Module):
    """Twin critics: min for the reward critic, max for the hazard (cost) critic."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2, type: str = "reward"):
        super().__init__()
        if type not in ["reward", "cost"]:
            raise ValueError("Invalid q function type")
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)
        self.type = type

    def both(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.type == "reward":
            return torch.min(*self.both(state, action))
        return torch.max(*self.both(state, action))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def FOCAL_IQL_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    max_action = config["max_action"]
    state_dim = config["state_dim"]
    action_dim = config["action_dim"]
    device = config["device"]

    actor = SafeActor(
        state_dim, action_dim, max_action=max_action,
        latent_dim=config["latent_dim"], vae_beta=config["vae_beta"], device=device,
    ).to(device)
    q_network = TwinQ(state_dim, action_dim).to(device)
    v_network = ValueFunction(state_dim).to(device)
    cost_q_network = TwinQ(state_dim, action_dim, type="cost").to(device)
    cost_v_network = ValueFunction(state_dim).to(device)

    return {
        "max_action": max_action,
        "actor": actor,
        "vae_actor_optimizer": torch.optim.Adam(actor.actor.parameters(), lr=config["actor_lr"]),
        "q_network": q_network,
        "q_optimizer": torch.optim.Adam(q_network.parameters(), lr=config["qf_lr"]),
        "v_network": v_network,
        "v_optimizer": torch.optim.Adam(v_network.parameters(), lr=config["vf_lr"]),
        "cost_q_network": cost_q_network,
        "cost_q_optimizer": torch.optim.Adam(cost_q_network.parameters(), lr=config["qf_lr"]),
        "cost_v_network": cost_v_network,
        "cost_v_optimizer": torch.optim.Adam(cost_v_network.parameters(), lr=config["vf_lr"]),
        "device": device,
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class FOCAL_IQL_Trainer:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        vae_actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        cost_q_network: nn.Module,
        cost_q_optimizer: torch.optim.Optimizer,
        cost_v_network: nn.Module,
        cost_v_optimizer: torch.optim.Optimizer,
        iql_tau: float = 0.7,           # expectile for V_r
        cost_iql_tau: float = 0.9,      # reversed expectile for V_h
        discount: float = 0.99,
        tau: float = 0.005,             # target update rate
        hazard_unsafe: float = 25.0,    # h(s) = M if cost > 0, else -1
        warmup_steps: int = 100000,     # critic-only warm-up steps
        alpha_r: float = 2.5,
        alpha_h: float = 2.5,
        n_samples: int = 16,            # candidates at inference
        device: str = "cpu",
        logger: WandbLogger = DummyLogger(),
        episode_len: int = 300,
        keep_case3: bool = False,       # include contradiction transitions in the infeasible mask
        # AWR ablation only
        use_awr: bool = False,
        beta_reward: float = 3.0,
        beta_cost: float = 5.0,
        awr_max_weight_reward: float = 100.0,
        awr_max_weight_cost: float = 150.0,
    ):
        self.max_action = max_action
        self.actor = actor
        self.vae_actor_optimizer = vae_actor_optimizer

        # Reward critics
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.q_optimizer = q_optimizer
        self.vf = v_network
        self.v_optimizer = v_optimizer

        # Hazard (cost) critics
        self.cost_qf = cost_q_network
        self.cost_q_target = copy.deepcopy(self.cost_qf).requires_grad_(False).to(device)
        self.cost_q_optimizer = cost_q_optimizer
        self.cost_vf = cost_v_network
        self.cost_v_optimizer = cost_v_optimizer

        self.iql_tau = iql_tau
        self.cost_iql_tau = cost_iql_tau
        self.discount = discount
        self.tau = tau
        self.hazard_unsafe = hazard_unsafe
        self.warmup_steps = warmup_steps
        self.alpha_r = alpha_r
        self.alpha_h = alpha_h
        self.n_samples = n_samples
        self.keep_case3 = keep_case3

        self.use_awr = use_awr
        self.rew_beta = beta_reward
        self.cost_beta = beta_cost
        self.awr_max_weight_reward = awr_max_weight_reward
        self.awr_max_weight_cost = awr_max_weight_cost

        self.logger = logger
        self.episode_len = episode_len
        self.total_it = 0
        self.device = device

    # ----- reward critics (IQL) -----

    def _update_v(self, observations, actions, log_dict):
        with torch.no_grad():
            target_q = self.q_target(observations, actions)
        v = self.vf(observations)
        v_loss = asymmetric_l2_loss(target_q - v, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

    def _update_q(self, next_v, observations, actions, rewards, terminals, log_dict):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.tau)

    # ----- hazard critics (feasibility) -----

    def _update_cost_v(self, states, actions, log_dict):
        with torch.no_grad():
            target_cost_q = self.cost_q_target(states, actions)
        cost_v = self.cost_vf(states)
        adv_h = target_cost_q - cost_v
        # Reversed expectile: larger weight on negative residuals
        weight = torch.abs(self.cost_iql_tau - (adv_h > 0).float())
        cost_v_loss = torch.mean(weight * (adv_h**2))

        log_dict["cost_v_mean"] = cost_v.mean().item()
        log_dict["cost_v_max"] = torch.max(cost_v).item()
        log_dict["cost_v_min"] = torch.min(cost_v).item()
        log_dict["cost_q_mean"] = target_cost_q.mean().item()
        log_dict["cost_v_loss"] = cost_v_loss.item()

        self.cost_v_optimizer.zero_grad()
        cost_v_loss.backward()
        self.cost_v_optimizer.step()

    def _update_cost_q(self, next_cost_v, states, actions, hazards, dones, log_dict):
        # Feasible Bellman target: (1 - gamma) h(s) + gamma * max(h(s), V_h(s'))
        future_v_h = torch.max(hazards, next_cost_v.detach())
        targets = (1.0 - self.discount) * hazards + self.discount * (
            (1.0 - dones.float()) * future_v_h + dones.float() * hazards
        )
        cost_qs = self.cost_qf.both(states, actions)
        cost_q_loss = sum(F.mse_loss(q, targets) for q in cost_qs) / len(cost_qs)
        log_dict["cost_q_loss"] = cost_q_loss.item()
        self.cost_q_optimizer.zero_grad()
        cost_q_loss.backward()
        self.cost_q_optimizer.step()
        soft_update(self.cost_q_target, self.cost_qf, self.tau)

    # ----- actor -----

    def _update_policy(self, observations, actions, log_dict):
        with torch.no_grad():
            v_h = self.cost_vf(observations).squeeze()
            q_h_target = self.cost_q_target(observations, actions).squeeze()

        # Feasibility masks
        feas_mask = ((v_h <= 0.0) & (q_h_target <= 0.0)).float()
        contradiction_mask = ((v_h > 0.0) & (q_h_target <= 0.0)).float()
        if self.keep_case3:
            infeas_mask = (v_h > 0.0).float()
        else:
            infeas_mask = ((v_h > 0.0) & (q_h_target > 0.0)).float()
        # Transitions with V_h <= 0 and Q_h > 0 are excluded from the actor update.
        valid_mask = feas_mask + infeas_mask

        kl_loss, recon_actions = self.actor.actor.vae_kl_loss(observations, actions)
        bc_loss = torch.sum((recon_actions - actions) ** 2, dim=1)

        if self.use_awr:
            # AWR ablation (weights clipped as in FISOR)
            with torch.no_grad():
                v_r = self.vf(observations).squeeze()
                q_r_target = self.q_target(observations, actions).squeeze()
                adv_r = q_r_target - v_r
                adv_h = q_h_target - v_h
            weight_feas = torch.exp(self.rew_beta * adv_r).clamp(max=self.awr_max_weight_reward)
            weight_infeas = torch.exp(-self.cost_beta * adv_h).clamp(max=self.awr_max_weight_cost)
            weighted_bc = feas_mask * weight_feas * bc_loss + infeas_mask * weight_infeas * bc_loss
            total_loss = (weighted_bc + valid_mask * kl_loss).mean()
        else:
            # Feasibility-partitioned first-order objective
            for p in self.qf.parameters():
                p.requires_grad = False
            for p in self.cost_qf.parameters():
                p.requires_grad = False

            q_r_recon = self.qf(observations, recon_actions).squeeze()
            q_h_recon = self.cost_qf(observations, recon_actions).squeeze()

            for p in self.qf.parameters():
                p.requires_grad = True
            for p in self.cost_qf.parameters():
                p.requires_grad = True

            # Normalized step sizes (stop-gradient)
            lmbda_r = self.alpha_r / (q_r_recon.abs().mean().detach() + 1e-5)
            lmbda_h = self.alpha_h / (q_h_recon.abs().mean().detach() + 1e-5)

            loss_feas = bc_loss - lmbda_r * q_r_recon
            loss_infeas = bc_loss + lmbda_h * q_h_recon
            total_loss = (feas_mask * loss_feas + infeas_mask * loss_infeas + valid_mask * kl_loss).mean()

        self.vae_actor_optimizer.zero_grad()
        total_loss.backward()
        self.vae_actor_optimizer.step()

        log_dict["vae/case1_ratio"] = feas_mask.mean().item()
        log_dict["vae/case4_ratio"] = infeas_mask.mean().item()
        log_dict["vae/case3_contradiction_ratio"] = contradiction_mask.mean().item()
        log_dict["vae/bc_loss"] = bc_loss.mean().item()
        log_dict["vae/total_loss"] = total_loss.item()

    # ----- training step -----

    def train_one_step(self, observations, next_observations, actions, rewards, costs, dones):
        self.total_it += 1
        log_dict = {}

        rewards = rewards.squeeze(dim=-1)
        costs = costs.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)

        hazards = torch.where(
            costs > 0,
            torch.tensor(self.hazard_unsafe, device=self.device),
            torch.tensor(-1.0, device=self.device),
        )

        with torch.no_grad():
            next_v = self.vf(next_observations)
            next_cost_v = self.cost_vf(next_observations)

        self._update_v(observations, actions, log_dict)
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        self._update_cost_v(observations, actions, log_dict)
        self._update_cost_q(next_cost_v, observations, actions, hazards, dones, log_dict)

        if self.total_it > self.warmup_steps:
            self._update_policy(observations, actions, log_dict)
        else:
            log_dict["vae/total_loss"] = 0.0

        self.logger.store(**log_dict)

    # ----- checkpointing -----

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "vae_actor_optimizer": self.vae_actor_optimizer.state_dict(),
            "cost_qf": self.cost_qf.state_dict(),
            "cost_vf": self.cost_vf.state_dict(),
            "cost_q_optimizer": self.cost_q_optimizer.state_dict(),
            "cost_v_optimizer": self.cost_v_optimizer.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        # Older checkpoints contain an unused `safety_encoder`; its entries are skipped.
        actor_state = {k: v for k, v in state_dict["actor"].items() if not k.startswith("safety_encoder.")}
        self.actor.load_state_dict(actor_state)
        self.vae_actor_optimizer.load_state_dict(state_dict["vae_actor_optimizer"])

        self.cost_qf.load_state_dict(state_dict["cost_qf"])
        self.cost_q_optimizer.load_state_dict(state_dict["cost_q_optimizer"])
        self.cost_q_target = copy.deepcopy(self.cost_qf)

        self.cost_vf.load_state_dict(state_dict["cost_vf"])
        self.cost_v_optimizer.load_state_dict(state_dict["cost_v_optimizer"])

        self.total_it = state_dict["total_it"]

    # ----- evaluation -----

    @torch.no_grad()
    def evaluate(self, env, eval_episodes: int) -> Tuple:
        self.actor.eval()
        rets, costs, lengths = [], [], []
        for _ in trange(eval_episodes, desc="Evaluating FOCAL...", leave=False):
            ret, cost, length = self.rollout(env)
            rets.append(ret)
            costs.append(cost)
            lengths.append(length)
        self.actor.train()
        return np.mean(rets), np.mean(costs), np.mean(lengths), np.std(rets), np.std(costs), np.std(lengths)

    @torch.no_grad()
    def rollout(self, env) -> Tuple[float, float, int]:
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        for _ in range(self.episode_len):
            act = self.actor.act(
                obs, self.qf, self.cost_qf, self.cost_vf, device=self.device, n_samples=self.n_samples
            ).flatten()
            obs_next, reward, terminated, truncated, info = env.step(act)
            episode_ret += reward
            episode_len += 1
            episode_cost += info["cost"]
            if terminated or truncated:
                break
            obs = obs_next
        return episode_ret, episode_cost, episode_len
