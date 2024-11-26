import os
import timeit
from dataclasses import dataclass
from typing import Any, Union, Tuple, Optional, Dict, TypeVar, Generic

import numpy as np
import pandas as pd

T = TypeVar('T')

@dataclass(kw_only=True)
class IndicatorBase:
    measure_time: bool = False
    save_to_csv: bool = False
    NUM_COLUMNS: int = 5

    def __post_init__(self) -> None:
        self.data: Optional[Union[pd.DataFrame, np.ndarray]] = None
        self._initialize_arrays()

    def _initialize_arrays(self) -> None:
        self.open = np.array([], dtype=np.float64)
        self.high = np.array([], dtype=np.float64)
        self.low = np.array([], dtype=np.float64)
        self.close = np.array([], dtype=np.float64)
        self.volume = np.array([], dtype=np.float64)

    def get_data(self, new_data: Union[pd.DataFrame, np.ndarray]) -> None:
        if isinstance(new_data, pd.DataFrame):
            self._handle_dataframe(new_data)
        elif isinstance(new_data, np.ndarray):
            self._handle_numpy_array(new_data)
        else:
            raise TypeError("Data must be a Pandas DataFrame or a NumPy array")

    def calculate_indicator(self, func: Any, *args: Any, required_length: int = 0, **kwargs: Any) -> Any:
        if self.data is None:
            raise ValueError("Data not initialized. Call get_data() first.")

        if len(self.data) < required_length:
            raise ValueError(f"Insufficient data. Need at least {required_length} data points, but only have {len(self.data)}.")

        if self.measure_time:
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            print(f"{func.__name__} took {timeit.default_timer() - start_time:.4f} seconds")
        else:
            result = func(*args, **kwargs)

        if self.save_to_csv:
            self._save_indicator_result_to_csv(func.__name__, result)

        return result

    def _save_indicator_result_to_csv(self, indicator_name: str, indicator_result: Union[np.ndarray, Tuple]) -> None:
        data: Dict[str, np.ndarray] = {'close': self.close}
        close_length = len(self.close)

        def add_to_data(key: str, array: np.ndarray) -> None:
            if len(array) == close_length:
                data[key] = array
            elif array.ndim == 2 and array.shape[1] == close_length:
                data.update({f"{key}_{i}": array[i, :] for i in range(array.shape[0])})

        if isinstance(indicator_result, tuple):
            for i, array in enumerate(indicator_result):
                add_to_data(f"{indicator_name}_{i}", array)
        else:
            add_to_data(indicator_name, indicator_result)

        if len(data) > 1:
            df = pd.DataFrame(data)
            csv_filename = f"{indicator_name}_results.csv"
            csv_dir = os.path.dirname(csv_filename) or '.'
            os.makedirs(csv_dir, exist_ok=True)
            df.to_csv(os.path.join(csv_dir, csv_filename), index=False)

    def _handle_dataframe(self, dataframe: pd.DataFrame) -> None:
        expected_cols = {'open', 'high', 'low', 'close', 'volume'}
        df_cols = {col.lower(): col for col in dataframe.columns}

        timestamp_variants = {'timestamp', 'date', 'datetime', 'time'}
        timestamp_col = next((df_cols[col] for col in timestamp_variants
                              if col in df_cols), None)

        missing = expected_cols - set(df_cols.keys())
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Create a mapping for required columns
        col_mapping = {
            'open': df_cols['open'],
            'high': df_cols['high'],
            'low': df_cols['low'],
            'close': df_cols['close'],
            'volume': df_cols['volume']
        }

        numeric_cols = {}
        for target, source in col_mapping.items():
            series = pd.to_numeric(dataframe[source], errors='coerce')
            numeric_cols[target] = series.astype(np.float64, copy=False)

        if timestamp_col:
            try:
                if not pd.api.types.is_datetime64_any_dtype(dataframe[timestamp_col]):
                    self.timestamp = pd.to_datetime(dataframe[timestamp_col]).to_numpy()
                else:
                    self.timestamp = dataframe[timestamp_col].to_numpy()
            except Exception as e:
                raise ValueError(f"Failed to process timestamp column: {str(e)}")

        self.open = numeric_cols['open'].to_numpy(copy=False)
        self.high = numeric_cols['high'].to_numpy(copy=False)
        self.low = numeric_cols['low'].to_numpy(copy=False)
        self.close = numeric_cols['close'].to_numpy(copy=False)
        self.volume = numeric_cols['volume'].to_numpy(copy=False)

        self.data = pd.DataFrame(numeric_cols, copy=False)

    def _handle_numpy_array(self, array: np.ndarray) -> None:
        has_timestamp = array.shape[1] == self.NUM_COLUMNS + 1
        expected_columns = self.NUM_COLUMNS + 1 if has_timestamp else self.NUM_COLUMNS

        if array.shape[1] != expected_columns:
            raise ValueError(f"Array must have {expected_columns} columns")

        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("All columns must be numerical")

        # Use copy=False for NumPy arrays
        self.data = array.astype(np.float64, copy=False)
        if has_timestamp:
            _, self.open, self.high, self.low, self.close, self.volume = self.data.T.copy()
        else:
            self.open, self.high, self.low, self.close, self.volume = self.data.T.copy()


class IndicatorCategory(Generic[T]):
    def __init__(self, base: IndicatorBase) -> None:
        self._base = base

    @property
    def open(self) -> np.ndarray:
        return self._base.open

    @property
    def high(self) -> np.ndarray:
        return self._base.high

    @property
    def low(self) -> np.ndarray:
        return self._base.low

    @property
    def close(self) -> np.ndarray:
        return self._base.close

    @property
    def volume(self) -> np.ndarray:
        return self._base.volume
