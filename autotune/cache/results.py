import json
import os
import pickle
import sys
import traceback
from typing import Dict, List, Tuple

from autotune.typing.infra_types import KERNEL_DTYPE, KERNEL_KWARGS_DTYPE


class ProfileResult:
    """
    Represents a single kernel performance result.
    """

    def __init__(
        self,
        index: int,
        main_metric: str,
        lower_is_better: bool,
        kernel: KERNEL_DTYPE,
        kernel_kwargs: KERNEL_KWARGS_DTYPE,
        compiler_flags: str,
        cache_dir: str,
    ):
        """
        Initialize a performance result.
        """
        self.index = index
        self.main_metric = main_metric
        self.lower_is_better = lower_is_better
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
        self.compiler_flags = compiler_flags
        self.cache_dir = cache_dir

    def __repr__(self) -> str:
        """Enhanced representation showing only attributes in self.attributes"""
        attributes = [f"{k}={getattr(self, k)}" for k in self.attributes]
        return f"ProfileResult({', '.join(attributes)})"

    @property
    def has_error(self) -> bool:
        return hasattr(self, "error")

    @property
    def main_metric_val(self) -> float:
        if "error" in self.attributes or self.main_metric not in self.attributes:
            if self.lower_is_better:
                val = float("inf")
            else:
                val = float("-inf")
        else:
            val = getattr(self, self.main_metric)
        return val

    @property
    def sort_val(self) -> Tuple[float, float]:
        if self.lower_is_better:
            score = self.main_metric_val
        else:
            score = -self.main_metric_val
        if "error" in self.attributes:
            priority = 2
        elif self.main_metric not in self.attributes:
            priority = 1
        else:
            priority = 0
        return (priority, score)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation including only attributes in self.attributes."""
        result = {}
        for attr in self.attributes:
            result[attr] = getattr(self, attr)
        return result

    def add_fields(self, **kwargs):
        """
        Add additional fields to this ProfileResult instance.

        Args:
            **kwargs: Arbitrary keyword arguments to add as attributes.
                If an attribute already exists, will attempt to add to it (e.g., update a dict).
                Will not overwrite existing attributes and will raise a warning if addition is not possible.
        """

        for key, value in kwargs.items():
            assert not hasattr(self, key), f"Attribute {key} already exists in ProfileResult."
            setattr(self, key, value)

    def remove_fields(self, *keys):
        """
        Remove fields from this ProfileResult instance.

        Args:
            **kwargs: Arbitrary keyword arguments to remove as attributes
        """
        for key in keys:
            if key in self.attributes:
                delattr(self, key)
                self.attributes.remove(key)

    def add_error(self, error_msg: str):
        """
        Add error information, but only if no error has been recorded yet.
        This ensures we keep the earliest error encountered.

        Args:
            error_msg: The error message to record
        """
        if "error" not in self.attributes:
            self.error = error_msg
            self.attributes.add("error")

    def save(self) -> str:
        """
        Save the ProfileResult instance to disk in its cache directory.

        Returns:
            str: The filepath where the object was saved

        Raises:
            AttributeError: If cache_dir attribute is not set
        """
        if not hasattr(self, "cache_dir") or "cache_dir" not in self.attributes:
            raise AttributeError(
                "Cannot save ProfileResult: 'cache_dir' attribute not found. "
                "Add a cache_dir attribute using add_fields() before calling save()."
            )
        filepath = os.path.join(self.cache_dir, "performance_result.pkl")

        # Collect all essential data and attributes
        state = {
            "main_metric": self.main_metric,
            "lower_is_better": self.lower_is_better,
            "attributes": self.attributes,
        }

        # Add all tracked attributes
        for attr in self.attributes:
            state[attr] = getattr(self, attr)

        # Save using pickle
        with open(filepath, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        return filepath


class ProfileResults:
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
        self.results: Dict[int, ProfileResult] = {}
        self.sort_key = sort_key
        self.lower_is_better = lower_is_better

    def add_result(self, result: ProfileResult) -> None:
        """
        Add a new performance result.

        Args:
            **kwargs: Metrics or metadata to store with the result

        Returns:
            The created ProfileResult instance

        Raises:
            ValueError: If sort_key is not provided in kwargs (only when sort_key is not empty)
        """
        self.results[result.index] = result

    def get_best_result(self) -> ProfileResult:
        """
        Get the best performing result based on the sort_key.
        If sort_key is empty, returns the first result.

        Returns:
            ProfileResult: The best performing configuration

        Raises:
            ValueError: If performance results are empty
        """
        if not self.results:
            raise ValueError("Performance results are empty")

        # If sort_key is empty, return the first result
        if not self.sort_key:
            return self.results[0]

        return (
            min(self.results, key=lambda result: getattr(result, self.sort_key, float("inf")))
            if self.lower_is_better
            else max(self.results, key=lambda result: getattr(result, self.sort_key, float("-inf")))
        )

    def to_dict_list(self) -> List[Dict]:
        """
        Convert all results to a list of dictionaries.
        Results are sorted by the sort_key if it's not empty.

        Returns:
            List[Dict]: List of dictionary representations of all results.
        """
        try:
            sorted_results = sorted(
                self.results,
                key=lambda result: getattr(result, self.sort_key),
                reverse=not self.lower_is_better,  # Reverse sort if higher values are better
            )
        except:
            sorted_results = self.results
        return [result.to_dict() for result in sorted_results]

    def dump_summary(self):
        """
        Dump the metrics summary to a JSON file.
        Results within each file are sorted by the sort_key.

        Raises:
            OSError: If the directory cannot be created or the file cannot be written
        """
        # Group results by filepath first
        results_by_filepath = {}
        filename = "perf_metrics.json"

        for result_index in self.results:
            result = self.results[result_index]
            workload_cache_dir = os.path.dirname(result.cache_dir)
            filepath = os.path.join(workload_cache_dir, filename)
            if filepath not in results_by_filepath:
                results_by_filepath[filepath] = []
            results_by_filepath[filepath].append(result)

        # Sort results within each filepath group and prepare JSON data
        json_data = {}
        for filepath, results in results_by_filepath.items():
            # Sort results using the same logic as to_dict_list
            sorted_results = sorted(results, key=lambda result: result.sort_val)

            # Count results with postprocessing_result: true
            correct_count = sum(
                1
                for result in sorted_results
                if "postprocessing_result" in result.attributes and getattr(result, "postprocessing_result") is True
            )

            # Count and categorize errors
            error_count = 0
            error_types = {}

            for result in sorted_results:
                if "error" in result.attributes:
                    error_count += 1
                    error_msg = result.error
                    error_type = error_msg.split("\n")[0]

                    # Increment count for this error type
                    if error_type not in error_types:
                        error_types[error_type] = 0
                    error_types[error_type] += 1

            json_data[filepath] = {
                "metadata": {
                    "sort_key": self.sort_key,
                    "lower_is_better": self.lower_is_better,
                    "num_results": len(sorted_results),
                    "num_correct_results": correct_count,
                    "num_error_results": error_count,
                    "error_types": error_types,
                },
                "results": [result.to_dict() for result in sorted_results],
            }

        # Write JSON files
        for filepath in json_data:
            try:
                directory = os.path.dirname(filepath)
                os.makedirs(directory, exist_ok=True)
                with open(filepath, "w") as f:
                    json.dump(json_data[filepath], f, indent=2, sort_keys=True)
            except Exception as e:
                raise OSError(f"Failed to save metrics to {filepath}: {str(e)}")

    def __repr__(self) -> str:
        """Return a string representation of the ProfileResults instance."""
        result_str = f"ProfileResults(sort_key='{self.sort_key}', lower_is_better={self.lower_is_better})"

        # Include up to 10 results as a preview
        preview_limit = min(10, len(self.results))
        if preview_limit > 0:
            for i in range(preview_limit):
                result_str += f"\n    {i}: {self.results[i]}"

            # Indicate if there are more results not shown
            if len(self.results) > preview_limit:
                result_str += f"\n    ... and {len(self.results) - preview_limit} more"

        return result_str


def get_best_result(data: Dict):
    # Filter out results with an "error" attribute
    valid_results = [result for result in data["results"] if not "error" in result]

    if not data["metadata"]["sort_key"]:
        return valid_results[0]

    sort_key = data["metadata"]["sort_key"]
    lower_is_better = data["metadata"]["lower_is_better"]
    return (
        min(valid_results, key=lambda result: getattr(result, sort_key, float("inf")))
        if lower_is_better
        else max(valid_results, key=lambda result: getattr(result, sort_key, float("-inf")))
    )


def capture_error_message(e) -> str:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    error_string = f"{exc_type.__name__}: {str(e)}\n"
    error_string += "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    return error_string
