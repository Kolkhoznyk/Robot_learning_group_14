
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch import nn
from torch.optim import Adam

from stable_baselines3 import SAC
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.utils import polyak_update


class SACDisentangledAlpha(SAC):
    """
    SAC with disentangled entropy coefficients alpha_k per task k.
    Task id is inferred from the last `num_tasks` entries of the observation (one-hot).

    Variant A:
      - One parameter vector log_ent_coef_vec (num_tasks,)
      - One optimizer Adam over that vector
      - Per-sample alpha is selected via gather/indexing by task_id
    """

    def __init__(
        self,
        policy: Union[str, Type],
        env: Union[GymEnv, str],
        num_tasks: int = 10,
        **kwargs,
    ):
        self.num_tasks = int(num_tasks)
        super().__init__(policy=policy, env=env, **kwargs)

    def _setup_model(self) -> None:
        # Let SB3 create actor/critic/etc.
        super()._setup_model()

        # We only support automatic entropy tuning in this class
        if self.ent_coef != "auto":
            raise ValueError("SACDisentangledAlpha requires ent_coef='auto' (automatic entropy tuning).")

        # Replace scalar log_ent_coef with vector parameter (num_tasks,)
        # Initialize with SB3's scalar if available, else 0.
        init_value = 0.0
        if hasattr(self, "log_ent_coef") and self.log_ent_coef is not None:
            try:
                init_value = float(self.log_ent_coef.detach().cpu().item())
            except Exception:
                init_value = 0.0

        self.log_ent_coef_vec = nn.Parameter(th.ones(self.num_tasks, device=self.device) * init_value)

        # New optimizer for the vector parameter
        self.ent_coef_optimizer = Adam([self.log_ent_coef_vec], lr=self.lr_schedule(1.0))

        # Keep target_entropy exactly as SB3 sets it (often "auto" → -|A|)
        # self.target_entropy already exists from SAC

    def _task_id_from_obs(self, obs: th.Tensor) -> th.Tensor:
        """
        obs: (batch, obs_dim)
        returns task_id: (batch,) long
        """
        onehot = obs[:, -self.num_tasks:]
        task_id = th.argmax(onehot, dim=1).long()
        return task_id

    def _alpha_from_task_id(self, task_id: th.Tensor) -> th.Tensor:
        """
        task_id: (batch,) long
        returns alpha: (batch, 1) float
        """
        alpha = th.exp(self.log_ent_coef_vec[task_id])  # (batch,)
        return alpha.unsqueeze(1)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Mostly SB3 SAC.train(), with alpha replaced by per-task alpha

        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer, self.ent_coef_optimizer])

        # Loggers (SB3 expects these keys sometimes)
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            obs = replay_data.observations
            next_obs = replay_data.next_observations

            # --- Per-sample alpha from task ids ---
            task_id = self._task_id_from_obs(obs)
            next_task_id = self._task_id_from_obs(next_obs)

            alpha = self._alpha_from_task_id(task_id)                 # (batch,1)
            alpha_next = self._alpha_from_task_id(next_task_id)       # (batch,1)

            # --- Actor: sample actions + log prob ---
            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.reshape(-1, 1)

            # --- Entropy coefficient loss (per task) ---
            # SB3 scalar version: ent_coef_loss = -(log_ent_coef * (log_prob + target_entropy).detach()).mean()
            # Here: select log_ent_coef per sample by task_id.
            log_ent_coef_task = self.log_ent_coef_vec[task_id].reshape(-1, 1)
            ent_coef_loss = -(log_ent_coef_task * (log_prob + self.target_entropy).detach()).mean()

            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            # --- Critic target ---
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                next_log_prob = next_log_prob.reshape(-1, 1)

                next_q_values = th.cat(self.critic_target(next_obs, next_actions), dim=1)
                next_q, _ = th.min(next_q_values, dim=1, keepdim=True)

                # target: r + gamma * (minQ - alpha(task(next_obs))*logpi)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * (
                    next_q - alpha_next * next_log_prob
                )

            # --- Critic loss ---
            current_q_values = self.critic(obs, replay_data.actions)
            critic_loss = 0.5 * sum((q - target_q).pow(2).mean() for q in current_q_values)

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # --- Actor loss ---
            q_values_pi = th.cat(self.critic(obs, actions_pi), dim=1)
            min_q_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

            actor_loss = (alpha * log_prob - min_q_pi).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # --- Target network update ---
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            # Collect logs
            ent_coef_losses.append(ent_coef_loss.item())
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

            # for logging: mean alpha over batch
            ent_coefs.append(float(alpha.mean().detach().cpu().item()))

        # Standard SB3 logs
        self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
        self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
        self.logger.record("train/ent_coef_loss", float(np.mean(ent_coef_losses)))
        self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))

        # Extra: log per-task alpha (useful!)
        with th.no_grad():
            alpha_vec = th.exp(self.log_ent_coef_vec).detach().cpu().numpy()
        for k in range(self.num_tasks):
            self.logger.record(f"alpha/alpha_task_{k}", float(alpha_vec[k]))
