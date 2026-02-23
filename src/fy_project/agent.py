# RLlib modules
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI

# pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F


class P2PTradingPolicy(TorchRLModule, ValueFunctionAPI):
    """
    Custom RLModule for the P2P energy trading environment.

    Config options:
    - demand_hidden_size : int
    - generation_hidden_size : int
    - time_blocks : int (the number of time blocks per day)
    """

    @override(TorchRLModule)
    def setup(self):
        time_blocks = self.model_config["time_blocks"]
        self.demand_layer = nn.LSTM(
            input_size=time_blocks,
            hidden_size=self.model_config["demand_hidden_size"],
            num_layers=2,
            batch_first=True,
        )
        self.generation_layer = nn.LSTM(
            input_size=time_blocks,
            hidden_size=self.model_config["generation_hidden_size"],
            num_layers=2,
            batch_first=True,
        )
        self.encoder_layer = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64)
        )

        self.decoder_layer = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 6)
        )
        self.value_layer = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def _compute_embeddings(self, batch):
        obs = batch.get(Columns.OBS, batch)

        input1 = torch.cat(
            (
                obs["time"],
                obs["market_price"],
                obs["battery_soc"],
                obs["battery_capacity"],
            )
        )
        embedding1 = self.encoder_layer(input1)
        demand_embedding = self.demand_layer(obs["forecast_demand"].unsqueeze(-1))[1][
            0
        ][-1]

        generation_embedding = self.generation_layer(
            obs["forecast_generation"].unsqueeze(-1)
        )[1][0][-1]

        embedding = torch.cat(
            (embedding1, demand_embedding, generation_embedding), dim=-1
        )
        return embedding

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # TODO : check if the values returned will be a numpy array or tensor
        embedding = self._compute_embeddings(batch)
        logits = self.decoder_layer(embedding)

        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        embeddings = self._compute_embeddings(batch)
        logits = self.decoder_layer(embeddings)
        return {Columns.ACTION_DIST_INPUTS: logits, Columns.EMBEDDINGS: embeddings}

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        if embeddings is None:
            embeddings = self._compute_embeddings(batch)
        value = self.value_layer(embeddings)
        return value
