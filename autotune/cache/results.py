import json
import os
from typing import Any, Dict, List


class PerformanceResult:
    """
    Represents a single kernel performance result.
    """

    def __init__(self, main_metric: str, lower_is_better: bool):
        """
        Initialize a performance result.
        """
        self.main_metric = main_metric
        self.lower_is_better = lower_is_better
        self.attributes = set()

    def __repr__(self) -> str:
        """Enhanced representation showing only attributes in self.attributes"""
        attributes = [f"{k}={getattr(self, k)}" for k in self.attributes]
        return f"PerformanceResult({', '.join(attributes)})"

    def __getattr__(self, name: str) -> Any:
        """
        ADDED: Handle attribute access gracefully when attribute doesn't exist.

        Args:
            name: Name of the attribute to access

        Returns:
            None if the attribute doesn't exist

        Raises:
            AttributeError: If the attribute doesn't exist and strict mode is enabled
        """
        return None

    def to_dict(self) -> Dict:
        """Convert to dictionary representation including only attributes in self.attributes."""
        result = {}
        for attr in self.attributes:
            result[attr] = getattr(self, attr)
        return result

    def add_fields(self, **kwargs):
        """
        Add additional fields to this ProfileJob instance.

        Args:
            **kwargs: Arbitrary keyword arguments to add as attributes
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.attributes.add(key)

    def add_error(self, error_msg: str):
        self.error = error_msg
        if self.lower_is_better:
            setattr(self, self.main_metric, float("inf"))
        else:
            setattr(self, self.main_metric, -float("inf"))
        self.attributes.add("error")
        self.attributes.add(self.main_metric)


class PerformanceMetrics:
    """
    Class to manage kernel performance metrics results.
    """

    def __init__(self, sort_key: str, lower_is_better: bool):
        """
        Initialize an empty collection of performance results.

        Args:
            sort_key: The metric name to use for sorting results
            lower_is_better: Whether lower values of the sort key are better (default: True)
        """
        self.results: List[PerformanceResult] = []
        self.sort_key = sort_key
        self.lower_is_better = lower_is_better

    def add_result(self, **kwargs) -> PerformanceResult:
        """
        Add a new performance result.

        Args:
            **kwargs: Metrics or metadata to store with the result

        Returns:
            The created PerformanceResult instance

        Raises:
            ValueError: If sort_key is not provided in kwargs (only when sort_key is not empty)
        """
        result = PerformanceResult(self.sort_key, self.lower_is_better)
        result.add_fields(**kwargs)
        self.results.append(result)
        return result

    def get_best_result(self) -> PerformanceResult:
        """
        Get the best performing result based on the sort_key.
        If sort_key is empty, returns the first result.

        Returns:
            PerformanceResult: The best performing configuration

        Raises:
            ValueError: If performance results are empty
        """
        if not self.results:
            raise ValueError("Performance results are empty")

        # If sort_key is empty, return the first result
        if not self.sort_key:
            return self.results[0]

        return (
            min(self.results, key=lambda result: getattr(result, self.sort_key))
            if self.lower_is_better
            else max(self.results, key=lambda result: getattr(result, self.sort_key))
        )

    def to_dict_list(self) -> List[Dict]:
        """
        Convert all results to a list of dictionaries.
        Results are sorted by the sort_key if it's not empty.

        Returns:
            List[Dict]: List of dictionary representations of all results.
        """
        # If sort_key is empty, return results without sorting
        if not self.sort_key:
            return [result.to_dict() for result in self.results]

        # Otherwise, sort by the sort_key
        sorted_results = sorted(
            self.results,
            key=lambda result: getattr(result, self.sort_key),
            reverse=not self.lower_is_better,  # Reverse sort if higher values are better
        )
        return [result.to_dict() for result in sorted_results]

    def save(self, cache_dir: str, filename: str = "perf_metrics.json") -> str:
        """
        Save the metrics to a JSON file.

        Args:
            cache_dir: Directory to save the result file
            filename: Name of the JSON file (default: "perf_metrics.json")

        Returns:
            Full path to the saved file

        Raises:
            OSError: If the directory cannot be created or the file cannot be written
        """
        os.makedirs(cache_dir, exist_ok=True)
        filepath = os.path.join(cache_dir, filename)
        json_data = {
            "metadata": {
                "sort_key": self.sort_key,
                "lower_is_better": self.lower_is_better,
                "num_results": len(self.results),
            },
            "results": self.to_dict_list(),
        }

        try:
            with open(filepath, "w") as f:
                json.dump(json_data, f, indent=2, sort_keys=True)
            return filepath
        except Exception as e:
            raise OSError(f"Failed to save metrics to {filepath}: {str(e)}")

    @classmethod
    def load(cls, filepath: str) -> "PerformanceMetrics":
        """
        Load metrics from a JSON file.

        Args:
            filepath: Path to the JSON file to load

        Returns:
            A new PerformanceMetrics instance with the loaded results

        Raises:
            FileNotFoundError: If the file doesn't exist
            json.JSONDecodeError: If the file contains invalid JSON
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Extract metadata if available
        metadata = data.get("metadata", {})
        sort_key = metadata.get("sort_key", "")
        lower_is_better = metadata.get("lower_is_better", True)

        # Create a new instance with appropriate size
        results_data = data.get("results", [])
        metrics = cls(size=len(results_data), sort_key=sort_key, lower_is_better=lower_is_better)

        # Load results
        for result_dict in results_data:
            metrics.add_result(**result_dict)

        return metrics

    def __len__(self) -> int:
        """Return the number of results."""
        return len(self.results)

    def __getitem__(self, index: int) -> PerformanceResult:
        """Access results by index."""
        return self.results[index]

    def __repr__(self) -> str:
        """Return a string representation of the PerformanceMetrics instance."""
        result_str = f"PerformanceMetrics(sort_key='{self.sort_key}', lower_is_better={self.lower_is_better})"

        # Include up to 10 results as a preview
        preview_limit = min(10, len(self.results))
        if preview_limit > 0:
            for i in range(preview_limit):
                result_str += f"\n    {i}: {self.results[i]}"

            # Indicate if there are more results not shown
            if len(self.results) > preview_limit:
                result_str += f"\n    ... and {len(self.results) - preview_limit} more"

        return result_str
