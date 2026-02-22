# RLlib modules
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core import Columns

# pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F


class P2PTradingPolicy(TorchRLModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.generation_hidden_size = config["generation_hidden_size"]
        self.demand_hidden_size = config["demand_hidden_size"]

    def setup(self):
        time_blocks = self.observation_space["forecast_demand"].shape[0]
        self.demand_layer = nn.LSTM(
            input_size=time_blocks,
            hidden_size=self.demand_hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.generation_layer = nn.LSTM(
            input_size=time_blocks,
            hidden_size=self.generation_hidden_size,
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

        self.optimizer = self.configure_optimizers()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config["lr"])

    def _forward(self, batch, **kwargs):
        # TODO : check if the values returned will be a numpy array or tensor
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

        logits = self.decoder_layer(embedding)
        value = self.value_layer(embedding)
        return {Columns.ACTIONS: logits, Columns.VF_PREDS: value.squeeze(-1)}
