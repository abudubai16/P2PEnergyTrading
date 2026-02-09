from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.models.modelv2 import ModelV2

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyTradingModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Scalars: time, price, soc, generation, demand
        self.scalar_fc = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
        )

        # Cloud forecast timeseries
        self.cloud_fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs)
        )

        self.value_head = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        scalars = torch.cat([
            obs["time"],
            obs["market_price"],
            obs["battery_soc"],
            obs["generation"],
            obs["demand"]
        ], dim=1)

        scalar_features = self.scalar_fc(scalars)
        cloud_features = self.cloud_fc(obs["cloud_forecast"])

        features = torch.cat([scalar_features, cloud_features], dim=1)

        logits = self.policy_head(features)
        self._value_out = self.value_head(features).squeeze(1)

        return logits, state

    def value_function(self):
        return self._value_out
    
ModelCatalog.register_custom_model(
    "energy_trading_model", EnergyTradingModel
)