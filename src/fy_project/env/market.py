import csv
from datetime import datetime
from fy_project.paths import IEX_DATA_DIR


class IEXMarket:
    def __init__(self):
        print(IEX_DATA_DIR)
        self.reset()

    def get_price(self, dt: datetime) -> float:
        try:
            val = next(self.reader)
        except StopIteration:
            self.reset()
            print("\n\n Hit end of file, looping back to start \n\n")
            val = next(self.reader)

        date_str = val["datetime"]
        if date_str != dt.strftime("%Y-%m-%d %H:%M:%S"):
            raise IndexError(
                f"Expected datetime {dt} not found in IEX data. Got {date_str} instead."
            )

        return float(val["price"]) / 1000 if "price" in val else 10.0

    def reset(self):
        self.reader = csv.DictReader(open(IEX_DATA_DIR, "r"))
