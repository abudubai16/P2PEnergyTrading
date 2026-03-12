import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks


class EnergyLoggerCallbacks(DefaultCallbacks):

    def on_episode_step(self, *, episode, metrics_logger, **kwargs):

        infos = episode.get_infos()

        for agent_id, agent_infos in infos.items():

            if not agent_infos:
                continue

            info = agent_infos[-1]

            for key, value in info.items():

                metric_name = f"{agent_id}/{key}"

                metrics_logger.log_value(metric_name, value, reduce="mean")
