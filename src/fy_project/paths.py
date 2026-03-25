import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # src directory
CONFIG_DIR = BASE_DIR / "config"
IEX_DATA_DIR = BASE_DIR / "fy_project" / "IEX_data" / "iex_hourly_market.csv"
IEX_DATA_DIR2 = BASE_DIR / "fy_project" / "IEX_data" / "iex_daily_market.csv"
CHECKPOINT_DIR = BASE_DIR.parent / "checkpoints"
WEATHER_DATA_DIR = BASE_DIR.parent / "weather_cache"

with open(Path.joinpath(CONFIG_DIR, "regulations.json"), "r") as f:
    REGULATIONS = json.load(f)
    DEMAND_CFG = REGULATIONS["demand_config"]
    BATTERY_CFG = REGULATIONS["battery_config"]
    DAILY_PROFILE = np.array(DEMAND_CFG["daily_profile"], dtype=np.float32)
    RL_CONST = REGULATIONS["rl_const"]
    GENERATION_CFG = REGULATIONS["solar_generation_config"]

with open(Path.joinpath(CONFIG_DIR, "env_config.json"), "r") as f:
    ENV_CFG = json.load(f)
    AGENT_CFG = ENV_CFG["agent_cfg"]
