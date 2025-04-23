import glob
import json
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from autotune.cache.directories import extract_mnk_from_dirname, get_cache_dir, get_save_path


def safe_subtract(a, b):
    """
    Safely subtract b from a, handling infinity edge cases.

    Args:
        a: First operand
        b: Second operand to subtract from a

    Returns:
        Result of a-b with proper infinity handling
    """
    if math.isinf(a) and math.isinf(b):
        # If both are infinity with the same sign, return 0
        if (a > 0 and b > 0) or (a < 0 and b < 0):
            return 0
        # If they have different signs, return a (since inf - (-inf) = inf)
        return a
    # Normal subtraction for all other cases
    return a - b


def safe_div(a, b):
    """
    Safely divide a by b, handling zero division and infinity edge cases.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Result of a/b with proper handling of edge cases
    """
    # Both infinity case
    if math.isinf(a) and math.isinf(b):
        # Same sign infinities
        if (a > 0 and b > 0) or (a < 0 and b < 0):
            return 1.0
        # Different sign infinities
        return -1.0

    # Zero division cases
    if b == 0 or b < 0 and a > 0:
        return float("inf")
    if b == 0 or b < 0 and a < 0:
        return float("-inf")
    if a == 0 and b == 0:
        return 1.0  # Equal values

    # Infinity in numerator or denominator
    if math.isinf(a) and b != 0:
        return a if b > 0 else -a
    if math.isinf(b):
        return 0.0

    # Standard division for normal cases
    return a / b


def analyze_parameter_importance(json_file):
    """Calculate importance of each parameter based on performance impact."""
    # Load performance metrics
    with open(json_file, "r") as f:
        data = json.load(f)

    # Convert results to DataFrame for easier analysis
    results = data["results"]
    df = pd.DataFrame([{**r["config"], "min_ms": r["min_ms"]} for r in results])

    # Get parameter names dynamically from the config
    if not results:
        print(f"Warning: No results found in {json_file}")
        return {}, pd.DataFrame(), []

    # Extract parameter names from the first configuration
    all_parameters = list(results[0]["config"].keys())

    # Dictionary to store impact metrics for each parameter
    param_impact = {}

    # Calculate impact for each parameter
    for param in all_parameters:
        # For parameters with only one unique value, add with zero impact
        if df[param].nunique() <= 1:
            single_value = df[param].iloc[0]
            param_impact[param] = {
                "impact_range_ms": 0.0,
                "impact_ratio": 1.0,  # Ratio of 1.0 means no impact
                "best_value": single_value,
                "worst_value": single_value,
                "best_perf": df["min_ms"].mean(),
                "worst_perf": df["min_ms"].mean(),
                "single_value": True,  # Mark this as a single-value parameter
            }
            continue

        # Group by parameter value and calculate mean performance
        performance_by_value = df.groupby(param)["min_ms"].agg(["mean", "count"])

        # Find best and worst values (lowest and highest mean latency)
        best_value = performance_by_value["mean"].idxmin()
        worst_value = performance_by_value["mean"].idxmax()
        best_perf = performance_by_value.loc[best_value, "mean"]
        worst_perf = performance_by_value.loc[worst_value, "mean"]

        # Calculate impact metrics
        impact_range_ms = safe_subtract(worst_perf, best_perf)  # Absolute difference (ms)
        impact_ratio = safe_div(worst_perf, best_perf)  # How many times worse

        # Store results
        param_impact[param] = {
            "impact_range_ms": impact_range_ms,
            "impact_ratio": impact_ratio,
            "best_value": best_value,
            "worst_value": worst_value,
            "best_perf": best_perf,
            "worst_perf": worst_perf,
            "single_value": False,  # Mark this as a multi-value parameter
        }

    return param_impact, df, all_parameters


def plot_parameter_importance(
    param_impact, metric="impact_ratio", m=None, n=None, k=None, run_type=None, plots_dir="plots", include_all=True
):
    """
    Create a plot showing ranked parameter importance.

    Args:
        param_impact: Dictionary of parameter impact data
        metric: Which metric to use for ranking ('impact_ratio' or 'impact_range_ms')
        m, n, k: Matrix dimensions
        run_type: 'tuned' or 'baseline'
        plots_dir: Base directory for plots
        include_all: Whether to include parameters with single values (no impact)
    """
    # Extract parameters that have meaningful impact (multiple values)
    if include_all:
        params = list(param_impact.keys())
    else:
        params = [p for p, data in param_impact.items() if not data.get("single_value", False)]

    if not params:
        print("Warning: No parameters to plot")
        return pd.DataFrame()

    impact_values = [param_impact[p][metric] for p in params]

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({"Parameter": params, "Impact": impact_values})

    # Sort by impact
    plot_df = plot_df.sort_values("Impact", ascending=False)

    # Set up the plot
    plt.figure(figsize=(12, 8))

    # Create bar chart
    bars = plt.bar(plot_df["Parameter"], plot_df["Impact"], color="steelblue")

    # Add parameter values for best/worst configurations
    for i, param in enumerate(plot_df["Parameter"]):
        best_val = str(param_impact[param]["best_value"])
        worst_val = str(param_impact[param]["worst_value"])

        # For single-value parameters, just add the value
        if param_impact[param].get("single_value", False):
            plt.text(i, 0.5, f"Value: {best_val}", ha="center", color="black", fontweight="bold")
        # For multi-value parameters with enough impact, show best/worst
        elif plot_df["Impact"].iloc[i] > 1.0:  # Only add if bar is tall enough
            plt.text(
                i, plot_df["Impact"].iloc[i] * 0.4, f"Best: {best_val}", ha="center", color="white", fontweight="bold"
            )
            plt.text(
                i, plot_df["Impact"].iloc[i] * 0.7, f"Worst: {worst_val}", ha="center", color="white", fontweight="bold"
            )

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{height:.2f}x", ha="center", va="bottom")

    # Set labels and title
    title_text = "Parameter Performance Impact (Ratio of Worst/Best Performance)"
    if metric == "impact_range_ms":
        title_text = "Parameter Performance Impact (Absolute Difference in ms)"

    plt.title(title_text, fontsize=16)
    plt.ylabel("Impact Factor", fontsize=14)
    plt.xlabel("Parameter", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save to param_details subdirectory
    param_details_dir = os.path.join(plots_dir, "param_details")
    save_dir, filename = get_save_path(param_details_dir, "parameter_importance", m, n, k, run_type)
    save_path = os.path.join(save_dir, filename)

    # Save the plot
    plt.savefig(save_path, dpi=300)
    plt.close()

    return plot_df


def create_parameter_interaction_heatmap(df, param1, param2, plots_dir, m=None, n=None, k=None, run_type=None):
    """Create a heatmap showing how two parameters interact to affect performance."""
    # Check if both parameters have multiple values
    if df[param1].nunique() <= 1 or df[param2].nunique() <= 1:
        plt.figure(figsize=(8, 6))
        plt.text(
            0.5,
            0.5,
            f"Cannot create interaction heatmap:\nParameter '{param1}' has {df[param1].nunique()} values\n"
            + f"Parameter '{param2}' has {df[param2].nunique()} values\n"
            + "Both parameters need multiple values for interactions.",
            ha="center",
            va="center",
        )
        plt.axis("off")
        plt.tight_layout()

        # Save to param_details subdirectory
        param_details_dir = os.path.join(plots_dir, "param_details")
        save_dir, filename = get_save_path(param_details_dir, f"interaction_{param1}_{param2}", m, n, k, run_type)
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()
        return

    # Create pivot table showing average performance for each combination
    pivot = df.pivot_table(index=param1, columns=param2, values="min_ms", aggfunc="mean")

    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis_r")
    plt.title(f"Average Performance: {param1} vs {param2} (ms, lower is better)")
    plt.tight_layout()

    # Save to param_details subdirectory
    param_details_dir = os.path.join(plots_dir, "param_details")
    save_dir, filename = get_save_path(param_details_dir, f"interaction_{param1}_{param2}", m, n, k, run_type)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()


def analyze_single_file(json_file, plots_dir):
    """
    Analyze a single perf_metrics.json file.

    Args:
        json_file: Path to the perf_metrics.json file
        plots_dir: Directory to save plot outputs
    """
    print(f"Analyzing: {json_file}")

    # Extract run type and shape info from file path
    path_parts = json_file.split(os.sep)

    # Find run type
    run_type = None
    for part in path_parts:
        if part in ["baseline", "tuned"]:
            run_type = part
            break

    # Extract the MNK directory name and values
    shape_dir = os.path.basename(os.path.dirname(json_file))
    m, n, k = extract_mnk_from_dirname(shape_dir)

    # Calculate parameter importance
    param_impact, df, all_parameters = analyze_parameter_importance(json_file)

    if not param_impact:
        print(f"No parameter impact data available for {json_file}")
        return None, df, None

    # Create main importance ranking plot - include all parameters
    ranking_df = plot_parameter_importance(
        param_impact, metric="impact_ratio", m=m, n=n, k=k, run_type=run_type, plots_dir=plots_dir, include_all=True
    )

    # Create detailed plots for each parameter
    create_parameter_detail_plots(
        param_impact, df, all_parameters, m=m, n=n, k=k, run_type=run_type, plots_dir=plots_dir
    )

    # Find top 2 parameters with variation for interaction heatmap
    varied_params = [p for p, data in param_impact.items() if not data.get("single_value", False)]
    if len(varied_params) >= 2:
        varied_ranking = ranking_df[ranking_df["Parameter"].isin(varied_params)]
        if not varied_ranking.empty:
            top_params = varied_ranking.iloc[:2]["Parameter"].tolist()
            create_parameter_interaction_heatmap(
                df, top_params[0], top_params[1], plots_dir, m=m, n=n, k=k, run_type=run_type
            )

    # Save analysis results as text file in param_details subdirectory
    param_details_dir = os.path.join(plots_dir, "param_details")
    save_dir, _ = get_save_path(param_details_dir, "parameter_analysis", m, n, k, run_type)
    with open(os.path.join(save_dir, "parameter_analysis.txt"), "w") as f:
        f.write(f"GEMM Parameter Analysis for M{m}N{n}K{k}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Parameter ranking by performance impact ratio:\n")
        for i, (_, impact) in enumerate(ranking_df.iterrows()):
            param_name = impact["Parameter"]
            impact_value = impact["Impact"]

            # Check if this is a single-value parameter
            if param_impact[param_name].get("single_value", False):
                single_value = param_impact[param_name]["best_value"]
                f.write(f"{i+1}. {param_name}: No impact (single value: {single_value})\n")
            else:
                best = param_impact[param_name]["best_value"]
                worst = param_impact[param_name]["worst_value"]
                f.write(f"{i+1}. {param_name}: {impact_value:.2f}x difference " f"(Best: {best}, Worst: {worst})\n")

        f.write("\nAll Parameters in Dataset:\n")
        for param in all_parameters:
            if param in param_impact:
                if param_impact[param].get("single_value", False):
                    f.write(f"  {param}: Single value = {param_impact[param]['best_value']}\n")
                else:
                    f.write(f"  {param}: Multiple values ({df[param].nunique()} unique values)\n")
            else:
                f.write(f"  {param}: Unknown\n")

        f.write("\nTop 5 configurations:\n")
        top_configs = df.nsmallest(5, "min_ms")
        for i, (_, row) in enumerate(top_configs.iterrows()):
            # Write all parameters from the configuration
            f.write(f"\nConfig {i+1}: {row['min_ms']:.4f} ms\n")
            for param in all_parameters:
                if param in row:
                    f.write(f"  {param}: {row[param]}\n")

    return param_impact, df, ranking_df


def create_parameter_detail_plots(
    param_impact, df, all_parameters, m=None, n=None, k=None, run_type=None, plots_dir="plots"
):
    """
    Create detailed plots showing performance for different values of each parameter.

    Args:
        param_impact: Dictionary of parameter impact data
        df: DataFrame with all results
        all_parameters: List of all parameters to ensure we plot all of them
        m, n, k: Matrix dimensions
        run_type: 'tuned' or 'baseline'
        plots_dir: Base directory to save plots
    """
    # Base directory for parameter details
    param_details_dir = os.path.join(plots_dir, "param_details")
    os.makedirs(param_details_dir, exist_ok=True)

    if not param_impact:
        print(f"No parameter impact data available")
        return

    for param in all_parameters:
        if param not in param_impact:
            continue

        # Get the directory path for saving this parameter's plot
        save_dir, _ = get_save_path(param_details_dir, param, m, n, k, run_type)

        # Skip plotting for single-value parameters with no variation
        if param_impact[param].get("single_value", False):
            plt.figure(figsize=(6, 3))
            plt.text(
                0.5,
                0.5,
                f"Parameter '{param}' has only one value: {param_impact[param]['best_value']}",
                ha="center",
                va="center",
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{param}_performance.png"), dpi=300)
            plt.close()
            continue

        # Group by parameter value and get mean performance
        perf_by_value = df.groupby(param)["min_ms"].mean().reset_index()

        # Sort by performance (ascending)
        perf_by_value = perf_by_value.sort_values("min_ms")

        # Create the plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(perf_by_value[param].astype(str), perf_by_value["min_ms"])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.05, f"{height:.2f}", ha="center", va="bottom")

        plt.title(f"Performance by {param} Value", fontsize=16)
        plt.ylabel("Latency (ms, lower is better)", fontsize=14)
        plt.xlabel(param, fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(save_dir, f"{param}_performance.png"), dpi=300)
        plt.close()


def find_all_perf_metrics_files(workload_name):
    """
    Find all perf_metrics.json files under a workload directory.

    Args:
        workload_name: Name of the workload

    Returns:
        List of found perf_metrics.json files
    """
    baseline_dir = get_cache_dir(workload_name, "baseline")
    tuned_dir = get_cache_dir(workload_name, "tuned")
    baseline_files = glob.glob(f"{baseline_dir}/M*-N*-K*/perf_metrics.json")
    tuned_files = glob.glob(f"{tuned_dir}/M*-N*-K*/perf_metrics.json")

    return baseline_files + tuned_files


def analyze_and_visualize(workload_name):
    """
    Analyze all perf_metrics.json files under a workload directory.

    Args:
        workload_name: Name of the workload
    """
    # Find all json files
    json_files = find_all_perf_metrics_files(workload_name)

    if not json_files:
        print(f"No perf_metrics.json files found under {workload_name}")
        return

    print(f"Found {len(json_files)} perf_metrics.json files to analyze")

    # Create plots directory
    plots_dir = get_cache_dir(workload_name, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Process each file
    for json_file in json_files:
        try:
            analyze_single_file(json_file, plots_dir)
            print(f"Completed analysis for {json_file}")
        except Exception as e:
            print(f"Error analyzing {json_file}: {str(e)}")

    print(f"Analysis complete for all {len(json_files)} files")
