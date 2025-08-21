from itertools import product
from typing import Dict, List


def generate_configs(**kwargs) -> List[Dict]:
    """
    Generate a list of configurations by combining all possible component options.

    Parameters:
    - **kwargs: Component names and their option lists
        e.g., NUM_BLOCK_M=[1,2,4], template=["MN","MKN"]

    Returns:
    - List of configuration dictionaries
    """
    # Get all component names and their option lists
    component_names = list(kwargs.keys())
    option_lists = list(kwargs.values())

    # Generate all possible combinations
    all_combinations = list(product(*option_lists))

    # Create a configuration dictionary for each combination
    configs = []
    for combo in all_combinations:
        # Start with base config if provided, otherwise empty dict
        config = {}

        # Add the specific component values to this config
        for i, name in enumerate(component_names):
            config[name] = combo[i]

        configs.append(config)

    return configs
