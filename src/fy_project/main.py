# module imports
from fy_project.env import P2PEnergyTrading, P2PEnergyTradingAuction
from fy_project.agent import P2PTradingPolicy, P2PTradingPolicyAuction
from fy_project.callbacks import EnergyLoggerCallbacks, EnergyLoggerCallbacksWandb
from fy_project.paths import CONFIG_DIR, CHECKPOINT_DIR

# internal imports
import os
import json
import logging
import warnings
from pathlib import Path

# RLlib imports
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModuleSpec, MultiRLModuleSpec, RLModule

# Testing imports
from ray import tune
from ray.tune import Tuner
from ray.air import RunConfig, CheckpointConfig
from ray.air.integrations.wandb import WandbLoggerCallback

# initialize ray logger
logger = logging.getLogger("ray.rllib")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def train(
    env_class=P2PEnergyTrading,
    policy_class: RLModule = P2PTradingPolicy,
    debug: bool = False,
    use_tuner: bool = False,
):
    def env_creator(cfg):
        return env_class(**cfg)

    # Load config files
    with open(Path.joinpath(CONFIG_DIR, "env_config.json"), "r") as f:
        env_config = json.load(f)
        agent_cfg = env_config["agent_cfg"]

    # Register the env with RLlib
    register_env("P2P_env", env_creator)

    # Get the observation and action spaces from the environment
    temp_env = env_class(**env_config)
    observation_space = temp_env.get_observation_space(0)
    action_space = temp_env.get_action_space(0)
    del temp_env

    # Create the RLModuleSpec for the policy
    module_spec = MultiRLModuleSpec(
        rl_module_specs={
            "shared_policy": RLModuleSpec(
                module_class=policy_class,
                observation_space=observation_space,
                action_space=action_space,
                model_config=agent_cfg,
            )
        }
    )

    # Create the PPOConfig
    config = (
        PPOConfig()
        .environment(
            env="P2P_env",
            env_config=env_config,
        )
        .training(train_batch_size=512)
        .callbacks(EnergyLoggerCallbacksWandb)
        .framework("torch")
        .rl_module(rl_module_spec=module_spec)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .env_runners(num_env_runners=6, num_envs_per_env_runner=1)
    )

    if debug:
        config.env_runners(num_env_runners=0, num_envs_per_env_runner=1)
        config.debugging(log_level="DEBUG")

    if use_tuner:
        tuner = Tuner(
            "PPO",
            param_space=config,
            run_config=RunConfig(
                name="p2p_energy_experiment",
                stop={"training_iteration": 1},
                callbacks=[
                    WandbLoggerCallback(
                        project="p2p-energy-market",
                        name="ppo-run-testing",
                        config=config.to_dict(),
                        sync_tensorboard=True,
                    )
                ],
                checkpoint_config=CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_frequency=10,
                    checkpoint_at_end=True,
                    checkpoint_score_attribute="episode_reward_mean",
                ),
            ),
        )

        results = tuner.fit()
        best_result = results.get_best_result()
        best_checkpoint = best_result.checkpoint
        print(best_checkpoint)

    else:
        ray.init()
        algo: Algorithm = config.build_algo()
        for _ in range(200):
            algo.train()


if __name__ == "__main__":
    train(
        env_class=P2PEnergyTradingAuction,
        policy_class=P2PTradingPolicyAuction,
        debug=False,
        use_tuner=True,
    )
