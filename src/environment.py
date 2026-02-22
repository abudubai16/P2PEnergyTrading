# RL libraries
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
from gymnasium.spaces import Box, Dict

# Math + ML libraries
import numpy as np

# Frist party libraries
import json
import requests
from typing import Dict, Tuple
from pathlib import Path
from datetime import datetime, timedelta

# https://docs.ray.io/en/latest/rllib/multi-agent-envs.html#rllib-multi-agent-environments-doc

"""
    Env-> 
    - time (0-23)
    - market_price 
    - cloud_forecast -> generation scaled from cloud cover  - Done
    - battery_soc -> capacity + state of charge
    - double auction logic 
    - Indian tariffs and other regulatory constraints and logic
    - Demand prediction - Done
"""

# Indian regulations and tariffs
with open(Path("config/regulations.json"), "r") as f:
    REGULATIONS = json.load(f)
    DAILY_PROFILE = np.array(REGULATIONS["daily_profile"])

with open(Path("config/agents.json"), "r") as f:
    AGENTS_CONFIG = json.load(f)


class IEXMarket:
    def __init__(self):
        pass


# Tested
class Weather:
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

    def __init__(self, city, lat, lon, cache_dir="weather_cache"):
        self.city = city.replace(" ", "_")
        self.lat = lat
        self.lon = lon
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_year = None
        self._year_data = None

    def _cache_path(self, year):
        return self.cache_dir / f"{self.city}_{year}.json"

    def _cache_valid(self, year):
        """
        Validates if cache exists AND contains enough hourly entries.
        A full year should have ~8760 hours.
        """
        path = self._cache_path(year)
        if not path.exists():
            return False

        try:
            with open(path, "r") as f:
                data = json.load(f)
            # minimal sanity check
            return len(data) > 8000
        except Exception:
            return False

    def _download_year(self, year):
        print(f"[Weather] Downloading NASA POWER data for {self.city} {year}...")

        params = {
            "parameters": "ALLSKY_SFC_SW_DWN,T2M",
            "community": "RE",
            "longitude": self.lon,
            "latitude": self.lat,
            "start": f"{year}0101",
            "end": f"{year}1231",
            "format": "JSON",
        }

        r = requests.get(self.BASE_URL, params=params, timeout=120)
        r.raise_for_status()

        raw = r.json()

        ghi_data = raw["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
        temp_data = raw["properties"]["parameter"]["T2M"]

        # Flatten structure for fast lookup
        processed = {}
        for key in ghi_data.keys():
            processed[key] = {"ghi": ghi_data[key], "temp": temp_data[key]}

        # Save to cache
        with open(self._cache_path(year), "w") as f:
            json.dump(processed, f)

        print(f"[Weather] Cached {len(processed)} hourly entries")

    def _ensure_year_loaded(self, year):
        if self._loaded_year == year:
            return

        if not self._cache_valid(year):
            self._download_year(year)

        with open(self._cache_path(year), "r") as f:
            self._year_data = json.load(f)

        self._loaded_year = year

    def get_weather(self, dt: datetime) -> Tuple[float, float]:
        """
        Returns: (temperature °C, GHI W/m²)
        """

        year = dt.year
        self._ensure_year_loaded(year)

        key = dt.strftime("%Y%m%d%H")

        if key not in self._year_data:
            # rare missing hour — fallback to nearest
            hour_back = dt.replace(hour=max(dt.hour - 1, 0))
            key = hour_back.strftime("%Y%m%d%H")

        entry = self._year_data[key]
        return entry["temp"], entry["ghi"]


# Tested
class HouseholdDemand:
    def __init__(self):
        self.load_config()

        hp = self.DEMAND_CONFIG["household_parameters"]
        self.house_scale = np.random.uniform(*hp["scale_range"])
        self.peak_kw = np.random.uniform(*hp["peak_kw_range"])

    def load_config(self):
        with open("config/demand_config.json", "r") as f:
            self.DEMAND_CONFIG = json.load(f)
            self.PROFILE = self.DEMAND_CONFIG["daily_profile"]

    def get_load(self, hour, day_of_week, day_of_year, temp) -> int:
        base = self.PROFILE[hour] * self.peak_kw * self.house_scale

        # stochastic noise
        noise = np.random.normal(0, self.DEMAND_CONFIG["noise_std"])
        load = base * (1 + noise)

        # temperature effect
        temperature = 30 if temp is None else temp
        temp_cfg = self.DEMAND_CONFIG["temperature_effect"]
        if temperature > temp_cfg["threshold"]:
            load += temp_cfg["sensitivity"] * (temperature - temp_cfg["threshold"])

        # appliances
        appliances = self.DEMAND_CONFIG["appliances"]

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
            load *= self.DEMAND_CONFIG["weekend_multiplier"]

        return max(load, self.DEMAND_CONFIG["minimum_load"])


# Tested
class Battery:
    def __init__(self, config_path="battery_config.json", timestep_hours=1.0):

        with open(f"config/{config_path}", "r") as f:
            cfg = json.load(f)

        self.dt = timestep_hours
        self.efficiency = cfg["efficiency"]

        # -------- Randomized physical parameters --------
        self.capacity_kwh = self._sample(cfg["capacity_kwh"])
        self.max_charge_kw = self._sample(cfg["max_charge_power_kw"])
        self.max_discharge_kw = self._sample(cfg["max_discharge_power_kw"])

        # -------- Initial State of Charge --------
        soc_low, soc_high = cfg["initial_soc_range"]
        self.soc = np.random.uniform(soc_low, soc_high)

    def _sample(self, dist):
        val = np.random.normal(dist["mean"], dist["std"])
        return float(np.clip(val, dist["min"], dist["max"]))

    def charge(self, requested_energy_kwh):
        """
        Attempt to store energy into battery.
        Returns actual energy accepted (kWh).
        """

        # convert power limit to energy limit
        max_possible = self.max_charge_kw * self.dt

        energy = min(requested_energy_kwh, max_possible)

        # available space
        available_space = (1 - self.soc) * self.capacity_kwh

        energy = min(energy, available_space)

        # apply efficiency (loss occurs during charging)
        stored_energy = energy * self.efficiency

        self.soc += stored_energy / self.capacity_kwh
        self.soc = min(self.soc, 1.0)

        return energy  # energy drawn from grid/PV

    # --------------------------------------------------
    # Discharging
    # --------------------------------------------------

    def discharge(self, requested_energy_kwh):
        """
        Attempt to supply energy from battery.
        Returns actual energy delivered (kWh).
        """

        max_possible = self.max_discharge_kw * self.dt
        energy = min(requested_energy_kwh, max_possible)

        available_energy = self.soc * self.capacity_kwh
        energy = min(energy, available_energy)

        # inverter loss during discharge
        delivered_energy = energy * self.efficiency

        self.soc -= energy / self.capacity_kwh
        self.soc = max(self.soc, 0.0)

        return delivered_energy

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def energy_stored(self):
        return self.soc * self.capacity_kwh

    def reset(self):
        # randomize initial SOC at new episode
        self.soc = np.random.uniform(0.3, 0.8)


class P2PEnergyTrading(MultiAgentEnv):
    def __init__(self, num_households):
        super().__init__()

        self.time_delta = (
            timedelta(hours=1) if AGENTS_CONFIG["use_hr"] else timedelta(minutes=15)
        )

        self.agents = [f"household_{i}" for i in range(num_households)]
        self._create_agents()

        """
            Action space : 
                a ~ [q_buy, q_sell, q_charge, q_discharge, p_buy, p_sell]
            Observation space:
                o ~ [time, market_price, battery_soc, battery_capacity, forecast_demand, generation_forecast]
        """
        self.action_spaces = {
            agent: self.get_action_space(agent_id)
            for agent_id, agent in enumerate(self.agents)
        }

        self.observation_spaces = {
            agent: self.get_observation_space(agent_id)
            for agent_id, agent in enumerate(self.agents)
        }

    def _create_agents(self):
        self.weather = Weather(city="Bangalore", lat=12.9719, lon=77.5937)
        self.market = IEXMarket()

        self.demand_model = [HouseholdDemand() for _ in self.agents]
        self.batteries = [Battery() for _ in self.agents]
        self.agent_solar = [
            np.random.uniform(0, AGENTS_CONFIG["max_solar_generation_kwh"])  # TODO
            for _ in self.agents
        ]

    def get_action_space(self, agent_id):
        return Dict(
            {
                # Market transactions
                "q_buy": Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array(
                        [REGULATIONS["max_transaction_kwh"]], dtype=np.float32
                    ),
                    dtype=np.float32,
                ),
                "q_sell": Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array(
                        [REGULATIONS["max_transaction_kwh"]], dtype=np.float32
                    ),
                    dtype=np.float32,
                ),
                # Battery control
                "q_charge": Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array(
                        [REGULATIONS["battery_power_limit_kwh"]], dtype=np.float32
                    ),
                    dtype=np.float32,
                ),
                "q_discharge": Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array(
                        [REGULATIONS["battery_power_limit_kwh"]], dtype=np.float32
                    ),
                    dtype=np.float32,
                ),
                # Market prices
                "p_buy": Box(
                    low=np.array([REGULATIONS["price_floor"]], dtype=np.float32),
                    high=np.array([REGULATIONS["price_cap"]], dtype=np.float32),
                    dtype=np.float32,
                ),
                "p_sell": Box(
                    low=np.array([REGULATIONS["price_floor"]], dtype=np.float32),
                    high=np.array([REGULATIONS["price_cap"]], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )

    def get_observation_space(self, agent_id):
        T = REGULATIONS["time_blocks_per_day"]
        return Dict(
            {
                "time": Box(low=0, high=23, shape=(1,), dtype=np.int32),
                "market_price": Box(
                    low=REGULATIONS["price_floor"],
                    high=REGULATIONS["price_cap"],
                    shape=(1,),
                    dtype=np.float32,
                ),
                "battery_soc": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "battery_capacity": Box(
                    low=0.0,
                    high=self.batteries[agent_id].capacity_kwh,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "forecast_demand": Box(
                    low=np.zeros(T, dtype=np.float32),
                    high=DAILY_PROFILE,
                    shape=(T,),
                    dtype=np.float32,
                ),
                "generation": Box(
                    low=np.zeros(T, dtype=np.float32),
                    high=np.ones(T, dtype=np.float32) * 2000,
                    shape=(T,),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self, *, seed=None, options=None):
        observations = {}

        t = 0
        self.date = datetime(
            2024, 1, 1, t
        )  # fixed date for consistent weather patterns
        market_price = self.market.get_price(t)
        demand_forecast = DAILY_PROFILE
        _, ghi = self.weather.get_weather(self.date)
        generation_forecast = ghi * self.agent_solar[i]

        for i, agent in enumerate(self.agents):
            observations[agent] = {
                "time": np.array([t], dtype=np.int32),
                "market_price": np.array([market_price], dtype=np.float32),
                "battery_soc": np.array([self.batteries[i].soc], dtype=np.float32),
                "battery_capacity": np.array(
                    [self.batteries[i].capacity_kwh], dtype=np.float32
                ),
                "forecast_demand": demand_forecast * self.demand_model[i].house_scale,
                "generation_forecast": generation_forecast * self.agent_solar[i],
            }

        return observations, {}

    def step(self):  # TODO
        self.date += self.time_delta
        t = self.date.hour
        market_price = self.market.get_price(self.date)
        demand_forecast = DAILY_PROFILE
        _, ghi = self.weather.get_weather(self.date)
