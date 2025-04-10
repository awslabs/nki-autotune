import json
import os
import pickle
from pprint import pformat
from typing import Any, Dict, List


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
        """Enhanced representation including all kwargs attributes"""
        base = f"PerformanceResult(latency={self.latency:.2f}ms, configs={self.configs}"
        for key in getattr(self, "_additional_keys", []):
            base += f", {key}={getattr(self, key)}"
        base += ")"
        return base

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
        Convert all results to a list of dictionaries, sorted by performance
        (best/lowest latency first).

        Returns:
            List[Dict]: List of dictionary representations of all results.
        """
        # Sort results by latency (ascending order - lower is better)
        sorted_results = sorted(self.results, key=lambda result: result.latency)
        return [result.to_dict() for result in sorted_results]

    def append(self, result: PerformanceResult):
        self.results.append(result)

    def save(self, cache_dir: str) -> None:
        """
        Save the metrics to log and JSON files.

        Args:
            cache_dir (str): Directory to save the result files
        """
        os.makedirs(cache_dir, exist_ok=True)
        json_data = {"results": self.to_dict_list()}
        with open(f"{cache_dir}/perf_metrics.json", "w") as f:
            json.dump(json_data, f, indent=2, sort_keys=True)

    def __len__(self) -> int:
        """Return the number of results."""
        return len(self.results)
