import os
import numpy as np
import optuna
import torch
import metaworld

# SB3
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# Gym / MetaWorld env creation (dein gym.make wie im Train-Skript)
import gymnasium as gym


# -----------------------------
# 1) Pruning-Callback (optional aber sehr sinnvoll)
# -----------------------------
class TrialEvalCallback(BaseCallback):
    """
    Evaluiert alle eval_freq steps, reported an Optuna und pruned ggf.
    """
    def __init__(self, eval_env, trial: optuna.Trial, n_eval_episodes: int, eval_freq: int, deterministic: bool = True):
        super().__init__()
        self.eval_env = eval_env
        self.trial = trial
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=False,
            )
            # report an Optuna
            self.trial.report(mean_reward, step=self.n_calls)

            # prune?
            # if self.trial.should_prune():
            #     return False

        return True

class OptunaStudy:
    def __init__(
        self,
        # ---- Env-Konstanten (alles was du in gym.make hattest)
        benchmark_id: str,                 # "Meta-World/MT1"
        env_name: str,                     # "reach-v3"
        algorithm: str,
        reward_function_version: str = "v3",
        max_episode_steps: int = 200,
        terminate_on_success: bool = False,
        n_envs: int = 1,
        use_parallel: bool = False,
        start_method: str = "spawn",
        n_envs_choices = (1,8,32,128),

        # ---- Training/Eval Konstanten
        total_timesteps: int = 50_000,
        eval_freq: int = 25_000,
        n_eval_episodes: int = 10,
        deterministic_eval: bool = True,
        device: str = "auto",

        # ---- SAC “fixe” Konstanten (nicht von Optuna gesampled)
        policy: str = "MlpPolicy",
        verbose: int = 0,
        ent_coef: str = "auto",

        # ---- Optuna Search-Ranges (die du „konstant“ übergeben willst)
        buffersize: int = (300000,),
        learning_starts: int = 10000,
        lr_min: float = 1e-5,
        lr_max: float = 3e-4,
        batch_sizes=(256, 512, 1024),
        gamma_min: float = 0.95,
        gamma_max: float = 0.999,
        tau_min: float = 0.001,
        tau_max: float = 0.02,
        train_freq_choices=(1, 4, 8),
        gradient_steps_choices=(1, 4, 8),
        target_entropy: str = "auto",
        use_sde: bool = False,
        net_arch = [256, 256, 256],
        activation_fn = torch.nn.ReLU,
        log_std_init = -3,

        # ---- Seed offsets für Train/Eval (damit reproduzierbar)
        seed: int = 42,
        eval_seed_offset: int = -2,
        final_eval_seed_offset: int = -3,
        normalize_rewards: bool = False,
    ):
        # Env
        self.benchmark_id = benchmark_id
        self.env_name = env_name
        self.seed = seed
        self.reward_function_version = reward_function_version
        self.max_episode_steps = max_episode_steps
        self.terminate_on_success = terminate_on_success
        self.n_envs = n_envs
        self.use_parallel = use_parallel
        self.start_method = start_method
        self.n_envs_choices = n_envs_choices

        # Training / Eval
        self.algorithm = algorithm
        self.total_timesteps = total_timesteps
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic_eval = deterministic_eval
        self.device = device
        self.buffersize = buffersize
        self.learning_starts = learning_starts
        self.target_entropy = target_entropy
        self.use_sde = use_sde
        self.net_arch = net_arch  # Deeper network
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init  # Initial exploration level

        # SAC fixed
        self.policy = policy
        self.verbose = verbose
        self.ent_coef = ent_coef

        # Optuna Search ranges
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.batch_sizes = list(batch_sizes)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.train_freq_choices = list(train_freq_choices)
        self.gradient_steps_choices = list(gradient_steps_choices)

        # Seed offsets
        self.eval_seed_offset = eval_seed_offset
        self.final_eval_seed_offset = final_eval_seed_offset
        self.normalize_reward = normalize_rewards

    def set_params(self, params: dict):
        for k, v in params.items():
            setattr(self, k, v)     

    def make_env(self, max_episode_steps, rank: int):
        def _init():
            env = gym.make(
                self.benchmark_id,              # "Meta-World/MT1"
                env_name=self.env_name,         # "reach-v3"
                seed=self.seed + rank,
                reward_function_version=self.reward_function_version,
                max_episode_steps=max_episode_steps,
                terminate_on_success=self.terminate_on_success,
            )

            if self.normalize_reward:
                env = gym.wrappers.NormalizeReward(env)

            env = Monitor(env)
            return env

        return _init

    def make_training_env(self, max_episode_steps):
        if self.use_parallel and self.n_envs > 1:
            print(f"Creating {self.n_envs} parallel Meta-World envs")
            return SubprocVecEnv(
                [self.make_env(max_episode_steps, i) for i in range(self.n_envs)],
                start_method=self.start_method,
            )
        else:
            print("Creating single Meta-World env")
            return self.make_env(max_episode_steps, 0)()
        
    def objective(self, trial: optuna.Trial) -> float:
        self.n_envs = trial.suggest_categorical("n_envs", self.n_envs_choices)
        max_episode_steps = trial.suggest_categorical("max_epsiode_steps", self.max_episode_steps)
        buffersize = trial.suggest_categorical("buffersize", self.buffersize)
        learning_rate = trial.suggest_float("learning_rate", self.lr_min, self.lr_max, log=True)
        batch_size = trial.suggest_categorical("batch_size", self.batch_sizes)
        gamma = trial.suggest_float("gamma", self.gamma_min, self.gamma_max)
        tau = trial.suggest_float("tau", self.tau_min, self.tau_max, log=True)

        # train_freq = trial.suggest_categorical("train_freq", self.train_freq_choices)
        # gradient_steps = trial.suggest_categorical("gradient_steps", self.gradient_steps_choices)

        net_key = trial.suggest_categorical("net_key", list(self.net_arch.keys()))
        net_arch = self.net_arch[net_key]

        # Print trial stats
        print("=" * 80)
        print(f"Trial {trial.number} started")
        for k, v in trial.params.items():
            print(f"  {k}: {v}")
        print("=" * 80)

        # --- Envs (Train + Eval)
        train_env = self.make_training_env(max_episode_steps)
        eval_env = self.make_env(max_episode_steps, self.seed+self.eval_seed_offset)()

        if self.algorithm == "SAC":
            model = SAC(
                policy=self.policy,
                env=train_env,
                learning_rate=learning_rate,
                buffer_size=buffersize,
                learning_starts=self.learning_starts,  # Start training sooner
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,  # Higher gamma for multi-step tasks
                train_freq=self.train_freq_choices,
                gradient_steps=self.gradient_steps_choices,  # Train on all available data
                ent_coef=self.ent_coef,  # Automatic entropy tuning - crucial for SAC
                target_entropy=self.target_entropy,  # Automatically set target entropy
                use_sde=self.use_sde,  # State-dependent exploration (can be enabled for more exploration)
                policy_kwargs=dict(
                    net_arch=net_arch,  # Deeper network
                    activation_fn=self.activation_fn,
                    log_std_init=self.log_std_init,  # Initial exploration level
                ),
                verbose=self.verbose,
                device=self.device,
                seed=self.seed,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        callback = TrialEvalCallback(
            eval_env=eval_env,
            trial=trial,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=self.eval_freq,
            deterministic=self.deterministic_eval,
        )

        try:
            model.learn(total_timesteps=self.total_timesteps, callback=callback, progress_bar=True)
        finally:
            # VecEnvs sauber schließen, sonst hängen Prozesse
            train_env.close()
            eval_env.close()

        # --- Final Eval ---
        final_eval_env = self.make_env(max_episode_steps, self.seed-self.final_eval_seed_offset)()
        try:
            mean_reward, _ = evaluate_policy(
                model,
                final_eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic_eval,
            )
        finally:
            final_eval_env.close()

        return float(mean_reward)



