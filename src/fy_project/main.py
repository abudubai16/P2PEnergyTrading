# module imports
from fy_project.env import P2PEnergyTrading, P2PEnergyTradingAuction
from fy_project.agent import P2PTradingPolicy, P2PTradingPolicyAuction
from fy_project.callbacks import EnergyLoggerCallbacksWandb
from fy_project.paths import ENV_CFG, AGENT_CFG

# internal imports
import os
import logging
import warnings

# RLlib imports
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module import RLModuleSpec, MultiRLModuleSpec, RLModule

# Ray Tune imports
from ray.tune import Tuner
from ray.air import RunConfig, CheckpointConfig
from ray.air.integrations.wandb import WandbLoggerCallback

# initialize ray logger
logger = logging.getLogger("ray.rllib")
warnings.filterwarnings("ignore", category=DeprecationWarning)


CPU_COUNT = os.cpu_count()


def register_custom_env(env_class=P2PEnergyTradingAuction):
    def env_creator(cfg):
        return env_class(**cfg)

    # Register the env with RLlib
    register_env("P2P_env_auction", env_creator)


def train(
    env_class=P2PEnergyTrading,
    policy_class: RLModule = P2PTradingPolicy,
    debug: bool = False,
    use_tuner: bool = False,
):
    env_name = "P2P_env_auction" if env_class == P2PEnergyTradingAuction else "P2P_env"

    # Get the observation and action spaces from the environment
    temp_env = env_class(**ENV_CFG)
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
                model_config=AGENT_CFG,
            )
        }
    )

    # Create the PPOConfig
    config = (
        PPOConfig()
        .environment(
            env=env_name,
            env_config=ENV_CFG,
        )
        .training(train_batch_size=512)
        .callbacks(EnergyLoggerCallbacksWandb)
        .framework("torch")
        .rl_module(rl_module_spec=module_spec)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
        )
        .env_runners(num_env_runners=CPU_COUNT - 1, num_envs_per_env_runner=1)
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
                stop={"training_iteration": 200},
                callbacks=[
                    WandbLoggerCallback(
                        project="p2p-energy-market",
                        name="ppo-run-training-1",
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
        debug=True,
        use_tuner=False,
    )
