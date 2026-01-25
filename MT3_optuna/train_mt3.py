import os
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from metaworld_envs.env_wrappers import MWEnvWrapperOneHotRewardScale
from metaworld_envs.mt3_env import MetaWorldMT3Env
from algos.sac_disentangled_alpha import SACDisentangledAlpha

def mk_mt3_env():
    def _init(rank, seed,task_env_names, max_epsiode_steps, reward_scales, first_task):
        env = MetaWorldMT3Env(
            task_env_names=task_env_names,
            seed= seed+rank,
            terminate_on_success=False,
            max_episode_steps=max_epsiode_steps,
            task_set="train",
            first_task=first_task,
        )
        env = MWEnvWrapperOneHotRewardScale(env=env, reward_scales=reward_scales)
        env = Monitor(env)
        return env
    return _init

if __name__ == "__main__":
    ALGORITHM = "SAC"
    TASK_NAMES = sorted(["reach-v3", "push-v3", "pick-place-v3"])

    N_ENVS = 3
    SEED = 42

    TOTAL_TIMESTEPS = 3_500_000
    MAX_EPISODE_STEPS = 150
    REWARD_SCALES = [4e-4, 2e-4, 2e-4]

    EVAL_FREQ = 15_000  # Evaluate every N steps
    N_EVAL_EPISODES = 20  # Number of episodes for evaluation
    CHECKPOINT_FREQ = 20_000  # Save checkpoint every N steps

    # Gemeinsame Parameter von Baseline-SAC und Disentangled-Alpha-SAC
    COMMON_SAC_ARGS = dict(
        learning_rate=0.0002524548054563359,
        gamma=0.9493925038998,
        tau=0.004187861038711214,
        buffer_size=300000,
        learning_starts=10000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        use_sde=False,
        policy_kwargs=dict(
            net_arch=[256, 256, 256],
            activation_fn=torch.nn.ReLU,
            log_std_init=0.0,
        ),
        verbose=1,
        device="auto",
        seed=SEED,
    )

    # Create output directories
    os.makedirs("./metaworld_models", exist_ok=True)
    os.makedirs("./metaworld_logs", exist_ok=True)

    print(f"=" * 60)
    print(f"Meta-World MT1 Training: {TASK_NAMES}")
    print(f"Algorithm: {ALGORITHM}")
    print(f"=" * 60)

    # Create Training Environment
    if N_ENVS > 1:
        print(f"Creating {N_ENVS} parallel training environments...")
        env = SubprocVecEnv(
            [mk_mt3_env(rank=i, seed=SEED, task_env_names=TASK_NAMES, max_epsiode_steps=MAX_EPISODE_STEPS, first_task=TASK_NAMES[i%3]) for i in range(N_ENVS)],
            start_method='spawn'
        )
    else:
        print("Creating single training environment...")
        env = DummyVecEnv([mk_mt3_env(rank=0, seed=SEED, task_env_names=TASK_NAMES, max_epsiode_steps=MAX_EPISODE_STEPS, first_task=TASK_NAMES[0])])

    # Create evaluation environment (without reward normalization for accurate eval)
    print("Creating evaluation environment...")
    # eval_env = make_env_mt10(0, SEED + 1000, MAX_EPISODE_STEPS, normalize_reward=False)() old
    eval_env =  DummyVecEnv([mk_mt3_env(rank=0, seed=SEED-10, task_env_names=TASK_NAMES, max_epsiode_steps=MAX_EPISODE_STEPS, first_task=TASK_NAMES[0])])

     # Create Model
    if ALGORITHM == "SAC_dA":
        model = SACDisentangledAlpha("MlpPolicy", env, num_tasks=3,
                                    tensorboard_log=f"./metaworld_logs/MT3_{ALGORITHM}/",
                                    **COMMON_SAC_ARGS)
    elif ALGORITHM == "SAC":
        model = SAC("MlpPolicy", env,
                    tensorboard_log=f"./metaworld_logs/MT3_{ALGORITHM}/",
                    **COMMON_SAC_ARGS)
    else:
        raise ValueError(f"Unknown algorithm: {ALGORITHM}")

    # Callbacks
    # Save checkpoint every CHECKPOINT_FREQ steps
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=f"./metaworld_models/checkpoints_MT3/",
        name_prefix=f"{ALGORITHM.lower()}_MT3",
        verbose=1
    )

    # Evaluate every EVAL_FREQ steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./metaworld_models/best_MT3_{ALGORITHM}/",
        log_path=f"./metaworld_logs/eval_MT3/",
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,  # More episodes for robust evaluation
        deterministic=True,
        render=False,
        verbose=1,
        warn=False
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True
    )
    
    # Save the final model
    print("\nSaving final model...")
    model.save(f"./metaworld_models/{ALGORITHM.lower()}_MT3_final")
    print("\n" + "=" * 60)
    print("Training complete!")

    # Cleanup
    env.close()
    eval_env.close()

