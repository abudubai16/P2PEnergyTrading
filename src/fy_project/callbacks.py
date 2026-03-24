import numpy as np

import wandb
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.air.integrations.wandb import WandbLoggerCallback

from ray.rllib.env.multi_agent_episode import MultiAgentEpisode


class EnergyLoggerCallbacks(DefaultCallbacks):

    def on_episode_step(self, *, episode, metrics_logger, **kwargs):

        infos = episode.get_infos()

        for agent_id, agent_infos in infos.items():
            if not agent_infos:
                continue

            info = agent_infos[-1]

            for key, value in info.items():
                if "household" not in key:
                    continue

                metric_name = f"{agent_id}/{key}"
                metrics_logger.log_value(metric_name, value, reduce="mean")


class EnergyLoggerCallbacksWandb(DefaultCallbacks):

    def on_episode_start(self, *, episode, **kwargs):
        # Storage for accumulating metrics
        self._agent_metric_buffer = {}

    # -------- STEP COLLECTION -------- #
    def on_episode_step(self, *, episode: MultiAgentEpisode, **kwargs):

        infos = episode.get_infos()
        if not infos:
            return

        for agent_id, agent_infos in infos.items():
            if not agent_infos:
                continue

            info = agent_infos[-1] if isinstance(agent_infos, list) else agent_infos

            # Initialize agent buffer if needed
            if agent_id not in self._agent_metric_buffer:
                self._agent_metric_buffer[agent_id] = {}

            for key, value in info.items():

                # Optional: filter only meaningful metrics
                if not isinstance(value, (int, float, np.number)):
                    continue

                if key not in self._agent_metric_buffer[agent_id]:
                    self._agent_metric_buffer[agent_id][key] = []

                self._agent_metric_buffer[agent_id][key].append(value)

    def on_episode_end(self, *, episode, metrics_logger, **kwargs):

        if not hasattr(self, "_agent_metric_buffer"):
            return

        for agent_id, metrics in self._agent_metric_buffer.items():
            for key, values in metrics.items():
                if len(values) == 0 or "household" not in key:
                    continue

                metric_name = f"{agent_id}/{key}"

                metrics_logger.log_value(
                    metric_name,
                    float(np.mean(values)),
                    reduce="mean",
                )
