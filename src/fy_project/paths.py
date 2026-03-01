from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # src directory
CONFIG_DIR = BASE_DIR / "config"

IEX_DATA_DIR = BASE_DIR / "fy_project" / "IEX_data" / "iex_hourly_market.csv"
