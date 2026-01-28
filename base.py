
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import xarray as xr

class WeatherModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def get_latest_init_time(self) -> datetime:
        pass

    @abstractmethod
    def fetch_data(self, variable, init_time: datetime, forecast_hour: int, region) -> xr.DataArray:
        pass

    def valid_time(self, init_time: datetime, forecast_hour: int) -> datetime:
        return init_time + timedelta(hours=forecast_hour)
