import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def analyze_parameter_importance(json_file):
    """Calculate importance of each parameter based on performance impact."""
    # Load performance metrics
    with open(json_file, "r") as f:
        data = json.load(f)

    # Convert results to DataFrame for easier analysis
    results = data["results"]
    df = pd.DataFrame([{**r["config"], "min_ms": r["min_ms"]} for r in results])

    # Dictionary to store impact metrics for each parameter
    param_impact = {}
    parameter_names = ["NUM_BLOCK_M", "NUM_BLOCK_N", "NUM_BLOCK_K", "BUFFER_M", "BUFFER_N", "BUFFER_K", "loop_order"]

    # Calculate impact for each parameter
    for param in parameter_names:
        # Group by parameter value and calculate mean performance
        performance_by_value = df.groupby(param)["min_ms"].agg(["mean", "count"])

        # Find best and worst values (lowest and highest mean latency)
        best_value = performance_by_value["mean"].idxmin()
        worst_value = performance_by_value["mean"].idxmax()
        best_perf = performance_by_value.loc[best_value, "mean"]
        worst_perf = performance_by_value.loc[worst_value, "mean"]

        # Calculate impact metrics
        impact_range_ms = worst_perf - best_perf  # Absolute difference (ms)
        impact_ratio = worst_perf / best_perf if best_perf > 0 else float("inf")  # How many times worse

        # Store results
        param_impact[param] = {
            "impact_range_ms": impact_range_ms,
            "impact_ratio": impact_ratio,
            "best_value": best_value,
            "worst_value": worst_value,
            "best_perf": best_perf,
            "worst_perf": worst_perf,
        }

    return param_impact, df


def plot_parameter_importance(param_impact, metric="impact_ratio", output_file="parameter_importance.png"):
    """Create a plot showing ranked parameter importance."""
    # Extract the chosen metric for each parameter
    params = list(param_impact.keys())
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
        # Only add text if there's enough space in the bar
        if plot_df["Impact"].iloc[i] > 1.2:  # Only add if bar is tall enough
            plt.text(
                i, plot_df["Impact"].iloc[i] * 0.4, f"Best: {best_val}", ha="center", color="white", fontweight="bold"
            )
            plt.text(
                i, plot_df["Impact"].iloc[i] * 0.7, f"Worst: {worst_val}", ha="center", color="white", fontweight="bold"
            )

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.1, f"{height:.2f}x", ha="center", va="bottom")

    # Set labels and title
    title_text = "Parameter Performance Impact (Ratio of Worst/Best Performance)"
    if metric == "impact_range_ms":
        title_text = "Parameter Performance Impact (Absolute Difference in ms)"

    plt.title(title_text, fontsize=16)
    plt.ylabel("Impact Factor", fontsize=14)
    plt.xlabel("Parameter", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300)
    plt.close()

    return plot_df


def create_parameter_detail_plots(param_impact, df, output_dir):
    """Create detailed plots showing performance for different values of each parameter."""
    os.makedirs(output_dir, exist_ok=True)

    for param, impact in param_impact.items():
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
        plt.savefig(f"{output_dir}/{param}_performance.png", dpi=300)
        plt.close()


def create_parameter_interaction_heatmap(df, param1, param2, output_file):
    """Create a heatmap showing how two parameters interact to affect performance."""
    # Create pivot table showing average performance for each combination
    pivot = df.pivot_table(index=param1, columns=param2, values="min_ms", aggfunc="mean")

    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis_r")
    plt.title(f"Average Performance: {param1} vs {param2} (ms, lower is better)")
    plt.tight_layout()

    plt.savefig(output_file, dpi=300)
    plt.close()


def analyze_single_file(json_file):
    """Analyze a single perf_metrics.json file."""
    print(f"Analyzing: {json_file}")

    # Get directory containing the json file for saving outputs
    output_dir = os.path.dirname(json_file)

    # Extract M, N, K values from directory name
    dir_name = os.path.basename(output_dir)
    size_info = ""
    if "M" in dir_name and "N" in dir_name and "K" in dir_name:
        size_info = f" ({dir_name})"

    # Calculate parameter importance
    param_impact, df = analyze_parameter_importance(json_file)

    # Create main importance ranking plot
    ranking_df = plot_parameter_importance(
        param_impact, "impact_ratio", os.path.join(output_dir, f"parameter_importance_ratio{size_info}.png")
    )

    # Create detailed plots for each parameter
    param_details_dir = os.path.join(output_dir, "param_details")
    create_parameter_detail_plots(param_impact, df, param_details_dir)

    # Get top 2 parameters and create interaction heatmap
    if len(ranking_df) >= 2:
        top_params = ranking_df["Parameter"].iloc[:2].tolist()
        create_parameter_interaction_heatmap(
            df, top_params[0], top_params[1], os.path.join(output_dir, f"top_param_interaction{size_info}.png")
        )

    # Save analysis results as text file
    with open(os.path.join(output_dir, "parameter_analysis.txt"), "w") as f:
        f.write(f"GEMM Parameter Analysis{size_info}\n")
        f.write("=" * 50 + "\n\n")

        f.write("Parameter ranking by performance impact ratio:\n")
        for i, (_, impact) in enumerate(ranking_df.iterrows()):
            best = param_impact[impact["Parameter"]]["best_value"]
            worst = param_impact[impact["Parameter"]]["worst_value"]
            f.write(
                f"{i+1}. {impact['Parameter']}: {impact['Impact']:.2f}x difference " f"(Best: {best}, Worst: {worst})\n"
            )

        f.write("\nTop 5 configurations:\n")
        top_configs = df.nsmallest(5, "min_ms")
        for i, (_, row) in enumerate(top_configs.iterrows()):
            f.write(f"\nConfig {i+1}: {row['min_ms']:.4f} ms\n")
            for param in [
                "NUM_BLOCK_M",
                "NUM_BLOCK_N",
                "NUM_BLOCK_K",
                "BUFFER_M",
                "BUFFER_N",
                "BUFFER_K",
                "loop_order",
            ]:
                f.write(f"  {param}: {row[param]}\n")

    return param_impact, df, ranking_df


def find_all_perf_metrics_files(base_dir):
    """Find all perf_metrics.json files under a base directory."""
    return glob.glob(f"{base_dir}/**/perf_metrics.json", recursive=True)


def analyze_and_visualize(base_dir):
    """Analyze all perf_metrics.json files under a base directory."""
    # Find all json files
    json_files = find_all_perf_metrics_files(base_dir)

    if not json_files:
        print(f"No perf_metrics.json files found under {base_dir}")
        return

    print(f"Found {len(json_files)} perf_metrics.json files to analyze")

    # Process each file
    for json_file in json_files:
        try:
            analyze_single_file(json_file)
            print(f"Completed analysis for {json_file}")
        except Exception as e:
            print(f"Error analyzing {json_file}: {str(e)}")

    print(f"Analysis complete for all {len(json_files)} files")
