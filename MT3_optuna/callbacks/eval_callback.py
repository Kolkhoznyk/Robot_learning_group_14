import optuna
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

class OptunaEvalCallback(BaseCallback):
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

            # # prune?
            # if self.trial.should_prune():
            #     return False

        return True