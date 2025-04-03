import json
import os
import pickle
from pprint import pformat
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt


class PerformanceResult:
    """
    Represents a single kernel performance result.
    """

    def __init__(self, configs: Dict[str, Any], latency: float, **kwargs):
        """
        Initialize a performance result.

        Args:
            configs (Dict): Configuration parameters used for this benchmark
            latency (float): Measured latency in microseconds
            **kwargs: Additional metrics or metadata to store with the result
        """
        self.configs = configs
        self.latency = latency

        # Store all additional keyword arguments as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Keep track of the additional keys for later use
        self._additional_keys = list(kwargs.keys())

    def __repr__(self) -> str:
        return f"PerformanceResult(latency={self.latency:.2f}ms, configs={self.configs})"

    def to_dict(self) -> Dict:
        """Convert to dictionary representation including all attributes."""
        result = {"configs": self.configs, "latency": self.latency}
        for key in getattr(self, "_additional_keys", []):
            result[key] = getattr(self, key)
        return result


class PerformanceMetrics:
    """
    Class to manage kernel performance metrics results.
    """

    def __init__(self):
        """Initialize an empty collection of performance results."""
        self.results: List[PerformanceResult] = []

    def add_result(self, configs: Dict[str, Any], latency: float, **kwargs) -> None:
        """
        Add a new performance result.

        Args:
            configs (Dict): Configuration parameters used for this benchmark
            latency (float): Measured latency in microseconds
            **kwargs: Additional metrics or metadata to store with the result
        """
        result = PerformanceResult(configs, latency, **kwargs)
        self.results.append(result)

    def add_result_from_dict(self, data: Dict) -> None:
        """
        Add a new performance result from a dictionary.

        Args:
            data (Dict): Dictionary with keys 'configs', 'latency', and potentially other keys
        """
        configs = data.pop("configs")
        latency = data.pop("latency")
        self.add_result(configs, latency, **data)

    def get_best_result(self) -> PerformanceResult:
        """
        Get the result with the lowest latency.

        Returns:
            PerformanceResult or None: The best performing configuration or None if no results.
        """
        assert self.results, "Performance results are empty."
        return min(self.results, key=lambda result: result.latency)

    def to_dict_list(self) -> List[Dict]:
        """
        Convert all results to a list of dictionaries.

        Returns:
            List[Dict]: List of dictionary representations of all results.
        """
        return [result.to_dict() for result in self.results]

    def save(self, cache_dir: str) -> None:
        """
        Save the metrics to log and pickle files.

        Args:
            log_path (str): Path to save the formatted text log
            pickle_path (str, optional): Path to save the pickle file
        """
        best_result = self.get_best_result()
        with open(f"{cache_dir}/perf_metrics.log", "w") as f:
            f.write(pformat(self.to_dict_list()))
            f.write(f"\nThe best latency is {best_result.latency} ms for the config {best_result.configs}")

        with open(f"{cache_dir}/perf_metrics.pkl", "wb") as f:
            pickle.dump(self.to_dict_list(), f)

    def __len__(self) -> int:
        """Return the number of results."""
        return len(self.results)
