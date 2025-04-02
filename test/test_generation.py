import random
from itertools import permutations, product
from typing import Dict, List, Tuple


class GenTests:
    def __init__(self, **kwargs) -> None:
        """
        Create combinations of the input parameters.
        Filter the invalid ones with a custom subclass method.
        """
        keys = list(kwargs.keys())
        values = [kwargs[key] for key in keys]
        combinations = list(product(*values))
        tests = []
        for combo in combinations:
            combo_dict = {keys[i]: combo[i] for i in range(len(keys))}
            tests.append(combo_dict)
        self.valid_tests = self._filter(tests)
        random.shuffle(self.valid_tests)

    def _filter(self, tests: List[Dict]) -> List[Tuple]:
        """
        Get a list of test configs

        Returns:
            List[Dict]: A list of test configs
        """
        valid_tests = []
        for test in tests:
            processed_test = self.process_test_config(test)
            if processed_test:
                valid_tests.append(processed_test)
        assert valid_tests, "No valid tests found"
        return valid_tests

    def process_test_config(self, config: Dict) -> Tuple | None:
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
