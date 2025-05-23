import random
from itertools import product
from typing import Dict, List, Tuple


class GenTests:
    def __init__(self, **kwargs) -> None:
        """
        Store parameters for later test generation.
        """
        self._original_kwargs = kwargs
        self.keys = list(kwargs.keys())
        self.values = [kwargs[key] for key in self.keys]
        self.valid_tests = None  # Will be populated when needed

    def get_all_tests(self) -> List[Tuple]:
        """
        Generate and return all valid tests.
        """
        if self.valid_tests is None:
            combinations = list(product(*self.values))
            tests = []
            for combo in combinations:
                combo_dict = {self.keys[i]: combo[i] for i in range(len(self.keys))}
                tests.append(combo_dict)

            valid_tests = []
            for test in tests:
                if self.process_test_config(test):
                    valid_tests.append(self._config_to_tuple(test))

            if not valid_tests:
                raise ValueError("No valid tests found")

            random.shuffle(valid_tests)
            self.valid_tests = valid_tests

        return self.valid_tests

    def process_test_config(self, config: Dict) -> bool:
        """
        Determine if a config is valid.

        Args:
            config (Dict): A test config

        Returns:
            bool: True if test config is valid
        """
        raise Exception(
            "process_test_config is not implemented in the base class. Must specify a method to determine if a config is valid (True/False)."
        )

    def _config_to_tuple(self, config: Dict) -> Tuple:
        """
        Convert a valid config dictionary to a tuple.

        Args:
            config (Dict): A test config that has been validated

        Returns:
            Tuple: The config values as a tuple, maintaining the original key order
        """
        config_tuple = tuple(config.get(key) for key in self.keys)
        return config_tuple

    def _index_to_combination(self, index: int) -> Dict:
        """
        Convert a flat index to a specific combination.

        Args:
            index (int): The index of the combination to retrieve

        Returns:
            Dict: A dictionary with parameter combinations
        """
        combo = []
        remaining_index = index

        # Go through each value list in reverse order
        for value_list in reversed(self.values):
            value_count = len(value_list)
            value_index = remaining_index % value_count
            combo.insert(0, value_list[value_index])
            remaining_index //= value_count

        # Create and return the combination dictionary
        return {self.keys[i]: combo[i] for i in range(len(self.keys))}

    def sample_tests(self, num_tests: int) -> List[Tuple]:
        """
        Sample num_tests valid tests without generating all combinations first.

        Args:
            num_tests (int): Number of valid tests to sample

        Returns:
            List[Tuple]: A list of num_tests valid test configurations
        """
        # Calculate total possible combinations
        total_combinations = 1
        for value_list in self.values:
            total_combinations *= len(value_list)

        # Create a list of all available indices
        available_indices = list(range(total_combinations))
        random.shuffle(available_indices)  # Shuffle for randomness

        valid_samples = []

        # Continue until we have num_tests valid tests or run out of indices
        while available_indices and len(valid_samples) < num_tests:
            # Take the next index from our shuffled list
            index = available_indices.pop()

            # Convert index to combination
            test_config = self._index_to_combination(index)

            # Process test config
            if self.process_test_config(test_config):
                valid_samples.append(self._config_to_tuple(test_config))

        # Check if we found enough valid tests
        if len(valid_samples) < num_tests:
            print(f"Warning: Could only find {len(valid_samples)} valid tests out of requested {num_tests}")

        return valid_samples
