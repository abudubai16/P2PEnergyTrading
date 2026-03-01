import numpy as np

import json
from pathlib import Path

from fy_project.paths import CONFIG_DIR


with open(Path.joinpath(CONFIG_DIR, "regulations.json"), "r") as f:
    REGULATIONS = json.load(f)
    DEMAND_CFG = REGULATIONS["demand_config"]
    DAILY_PROFILE = np.array(DEMAND_CFG["daily_profile"])


# Tested
class HouseholdDemand:
    def __init__(self):
        self.PROFILE = DEMAND_CFG["daily_profile"]
        hp = DEMAND_CFG["household_parameters"]
        self.house_scale = np.random.uniform(*hp["scale_range"])
        self.peak_kw = np.random.uniform(*hp["peak_kw_range"])

    def get_load(self, hour, day_of_week, day_of_year, temp) -> int:
        base = self.PROFILE[hour] * self.peak_kw * self.house_scale

        # stochastic noise
        noise = np.random.normal(0, DEMAND_CFG["noise_std"])
        load = base * (1 + noise)

        # temperature effect
        temperature = 30 if temp is None else temp
        temp_cfg = DEMAND_CFG["temperature_effect"]
        if temperature > temp_cfg["threshold"]:
            load += temp_cfg["sensitivity"] * (temperature - temp_cfg["threshold"])

        # appliances
        appliances = DEMAND_CFG["appliances"]

        # cooking
        cook = appliances["cooking"]
        if hour in cook["hours"] and np.random.rand() < cook["probability"]:
            load += cook["load_kw"]

        # AC
        ac = appliances["ac"]
        if hour in ac["hours"] and np.random.rand() < ac["probability"]:
            load += np.random.uniform(*ac["load_kw_range"])

        # water pump (any hour)
        pump = appliances["pump"]
        if np.random.rand() < pump["probability"]:
            load += pump["load_kw"]

        # weekend effect
        if day_of_week in [5, 6]:
            load *= DEMAND_CFG["weekend_multiplier"]

        return max(load, DEMAND_CFG["minimum_load"])
