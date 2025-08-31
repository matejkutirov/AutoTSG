import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from pathlib import Path


def load_results():
    """Load results from results.json"""
    with open("results.json", "r") as f:
        return json.load(f)


def create_results_bar_plots():
    """Create bar plots for results with shared y-axis"""
    # Load results from JSON file
    with open("results/results.json", "r") as f:
        results = json.load(f)

    datasets = list(results.keys())
    models = list(results[datasets[0]].keys())
    metrics = list(results[datasets[0]][models[0]].keys())

    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(18, 30), sharex=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(len(datasets))
    width = 0.12  # smaller bars

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # For each model, plot bars for all datasets
        for k, model in enumerate(models):
            values = []
            for dataset in datasets:
                if model in results[dataset]:
                    values.append(results[dataset][model][metric])
                else:
                    values.append(0)

            model_x = x + (k - len(models) / 2) * width + width / 2
            ax.bar(
                model_x,
                values,
                width=width,
                label=model,
                color=colors[k % len(colors)],
                edgecolor="black",
            )

        ax.set_ylabel(metric, fontsize=18, fontweight="bold")
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(True, axis="y", alpha=0.3)

        if i == n_metrics - 1:  # last row, set dataset names
            ax.set_xticks(x)
            ax.set_xticklabels(
                [d.capitalize() for d in datasets], fontsize=20, fontweight="bold"
            )
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([])

        if i == 0:
            ax.legend(
                bbox_to_anchor=(0.5, 1.25), loc="center", ncol=len(models), fontsize=20
            )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.savefig("results_bar_plots.png", dpi=300, bbox_inches="tight")
    plt.show()


def create_automl_bar_plots():
    """Create bar plots comparing AutoML best models with TSG best models for each metric with shared y-axis"""
    # Load results from JSON files
    with open("results/results.json", "r") as f:
        results = json.load(f)

    with open("results/automl_results.json", "r") as file:
        automl_results = json.load(file)

    # Get datasets from both files
    datasets = list(results.keys())
    datasets_automl = list(automl_results.keys())

    # Use intersection of datasets that exist in both files
    common_datasets = [d for d in datasets if d in datasets_automl]

    if not common_datasets:
        print("No common datasets found between results.json and automl_results.json")
        return

    print(f"Common datasets: {common_datasets}")

    # Get metrics from TSG results (excluding Training_Time)
    first_dataset = common_datasets[0]
    first_model = list(results[first_dataset].keys())[0]
    metrics = list(results[first_dataset][first_model].keys())
    if "Training_Time" in metrics:
        metrics.remove("Training_Time")

    print(f"Metrics to plot: {metrics}")

    # Get AutoML best models for each dataset (only time_budget_300)
    automl_best_models = {}
    automl_scores = {}
    for dataset in common_datasets:
        if "time_budget_300" in automl_results[dataset]:
            run_data = automl_results[dataset]["time_budget_300"]
            automl_best_models[dataset] = run_data.get("selected_model", "unknown")
            automl_scores[dataset] = run_data.get("best_score", 0)
        else:
            print(f"Warning: No time_budget_300 found for dataset {dataset}")
            automl_best_models[dataset] = "unknown"
            automl_scores[dataset] = 0

    print(f"AutoML best models: {automl_best_models}")

    # Get TSG best models for each metric and dataset combination
    tsg_best_models = {}
    for metric in metrics:
        tsg_best_models[metric] = {}
        for dataset in common_datasets:
            best_model = None
            best_score = float("inf")
            for model in results[dataset]:
                if metric in results[dataset][model]:
                    score = results[dataset][model][metric]
                    if score < best_score:
                        best_score = score
                        best_model = model
            tsg_best_models[metric][dataset] = best_model

    print(f"TSG best models: {tsg_best_models}")

    # Create the plot - one subplot for each metric with shared y-axis
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 25), sharex=True)

    # Ensure axes is always a list
    if n_metrics == 1:
        axes = [axes]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x = np.arange(len(common_datasets))
    width = 0.35  # wider bars for two models

    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Plot AutoML best model results for this metric
        automl_values = []
        automl_best_models_for_metric = {}  # Store which model was best for each dataset and metric

        for dataset in common_datasets:
            if "time_budget_300" in automl_results[dataset]:
                run_data = automl_results[dataset]["time_budget_300"]
                if "all_models" in run_data:
                    # Find the best model for this specific metric in this dataset
                    best_score = float("inf")
                    best_model = "unknown"

                    for model_name, model_data in run_data["all_models"].items():
                        if (
                            "comprehensive_metrics" in model_data
                            and metric in model_data["comprehensive_metrics"]
                        ):
                            score = model_data["comprehensive_metrics"][metric]
                            if score < best_score:
                                best_score = score
                                best_model = model_name

                    if best_model != "unknown":
                        automl_values.append(best_score)
                        automl_best_models_for_metric[dataset] = best_model
                    else:
                        automl_values.append(0)
                        automl_best_models_for_metric[dataset] = "unknown"
                else:
                    automl_values.append(0)
                    automl_best_models_for_metric[dataset] = "unknown"
            else:
                automl_values.append(0)
                automl_best_models_for_metric[dataset] = "unknown"

        # Plot TSG best model results for this metric
        tsg_values = []
        for dataset in common_datasets:
            best_model = tsg_best_models[metric][dataset]
            if (
                best_model in results[dataset]
                and metric in results[dataset][best_model]
            ):
                tsg_values.append(results[dataset][best_model][metric])
            else:
                tsg_values.append(0)

        # Create bars
        x_pos = x - width / 2
        ax.bar(
            x_pos,
            tsg_values,
            width,
            label=f"TSG",
            color=colors[1],
            edgecolor="black",
            alpha=0.8,
        )

        x_pos = x + width / 2
        ax.bar(
            x_pos,
            automl_values,
            width,
            label=f"AutoML",
            color=colors[0],
            edgecolor="black",
            alpha=0.8,
        )

        # Add model names on top of each bar
        for j, dataset in enumerate(common_datasets):
            # TSG bar
            if j < len(tsg_values) and tsg_values[j] > 0:
                tsg_model = tsg_best_models[metric][dataset]
                ax.text(
                    j - width / 2,
                    tsg_values[j] + max(tsg_values) * 0.01,
                    tsg_model,
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                )

            # AutoML bar
            if j < len(automl_values) and automl_values[j] > 0:
                automl_model = automl_best_models_for_metric.get(dataset, "unknown")
                ax.text(
                    j + width / 2,
                    automl_values[j] + max(automl_values) * 0.01,
                    automl_model,
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    fontweight="bold",
                )

        ax.set_ylabel(metric, fontsize=18, fontweight="bold")
        ax.tick_params(axis="y", labelsize=18)
        ax.grid(True, axis="y", alpha=0.3)

        if i == n_metrics - 1:  # last row, set dataset names
            ax.set_xticks(x)
            ax.set_xticklabels(
                [d.capitalize() for d in common_datasets],
                fontsize=20,
                fontweight="bold",
            )
        else:
            ax.set_xticks(x)
            ax.set_xticklabels([])

        if i == 0:
            ax.legend(bbox_to_anchor=(0.5, 1.25), loc="center", ncol=2, fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.savefig("automl_vs_tsg_bar_plots.png", dpi=300, bbox_inches="tight")
    plt.show()


def calculate_accuracy():
    # Calculate accuracy for different AutoML time budgets vs TSG
    with open("results/results.json", "r") as f:
        results = json.load(f)
    with open("results/automl_results.json", "r") as f:
        automl_results = json.load(f)

    # Only use datasets that exist in both files (exclude 'air')
    common_datasets = [d for d in results.keys() if d in automl_results]
    print(f"Calculating accuracy for datasets: {common_datasets}")

    # Get metrics from TSG results (excluding Training_Time)
    first_dataset = common_datasets[0]
    first_model = list(results[first_dataset].keys())[0]
    metrics = list(results[first_dataset][first_model].keys())
    if "Training_Time" in metrics:
        metrics.remove("Training_Time")

    print(f"Metrics to calculate accuracy for: {metrics}")

    # Initialize accuracy tracking for each dataset and time budget
    accuracy_matrix = {}
    for dataset in common_datasets:
        accuracy_matrix[dataset] = {}
        for time_budget in [
            "time_budget_120",
            "time_budget_300",
            "time_budget_600",
            "time_budget_1200",
        ]:
            accuracy_matrix[dataset][time_budget] = {"correct": 0, "total": 0}

    # Calculate accuracy for each combination
    for dataset in common_datasets:
        for metric in metrics:  # Iterate over metrics, not model names
            for time_budget in automl_results[dataset].keys():
                if time_budget in [
                    "time_budget_120",
                    "time_budget_300",
                    "time_budget_600",
                    "time_budget_1200",
                ]:
                    accuracy_matrix[dataset][time_budget]["total"] += 1

                    # Find the best AutoML model for this specific metric in this dataset and time budget
                    automl_best_model = "unknown"
                    if time_budget in automl_results[dataset]:
                        run_data = automl_results[dataset][time_budget]
                        if "all_models" in run_data:
                            best_score = float("inf")
                            for model_name, model_data in run_data[
                                "all_models"
                            ].items():
                                if (
                                    "comprehensive_metrics" in model_data
                                    and metric in model_data["comprehensive_metrics"]
                                ):
                                    score = model_data["comprehensive_metrics"][metric]
                                    if score < best_score:
                                        best_score = score
                                        automl_best_model = model_name

                    # Get TSG best model for this metric in this dataset
                    tsg_best_model = None
                    tsg_best_score = float("inf")
                    for model in results[dataset]:
                        if metric in results[dataset][model]:
                            score = results[dataset][model][metric]
                            if score < tsg_best_score:
                                tsg_best_score = score
                                tsg_best_model = model

                    # Compare AutoML best vs TSG best for this metric and dataset
                    if (
                        automl_best_model == tsg_best_model
                        and automl_best_model != "unknown"
                    ):
                        accuracy_matrix[dataset][time_budget]["correct"] += 1
                        print(
                            f"✓ {dataset} - {metric} - {time_budget}: AutoML ({automl_best_model}) = TSG ({tsg_best_model})"
                        )
                    else:
                        print(
                            f"✗ {dataset} - {metric} - {time_budget}: AutoML ({automl_best_model}) ≠ TSG ({tsg_best_model})"
                        )

    # Display accuracy for each dataset and time budget combination
    print(f"\n📊 Accuracy Summary by Dataset and Time Budget:")
    print("=" * 60)

    total_correct = 0
    total_comparisons = 0

    for dataset in common_datasets:
        print(f"\n{dataset.upper()}:")
        print("-" * 30)
        for time_budget in [
            "time_budget_120",
            "time_budget_300",
            "time_budget_600",
            "time_budget_1200",
        ]:
            if time_budget in accuracy_matrix[dataset]:
                correct = accuracy_matrix[dataset][time_budget]["correct"]
                total = accuracy_matrix[dataset][time_budget]["total"]
                if total > 0:
                    accuracy_percentage = (correct / total) * 100
                    print(
                        f"  {time_budget}: {correct}/{total} = {accuracy_percentage:.1f}%"
                    )
                    total_correct += correct
                    total_comparisons += total
                else:
                    print(f"  {time_budget}: No data available")
            else:
                print(f"  {time_budget}: Not found in results")

    # Overall accuracy
    if total_comparisons > 0:
        overall_accuracy = (total_correct / total_comparisons) * 100
        print(
            f"\n🎯 OVERALL ACCURACY: {total_correct}/{total_comparisons} = {overall_accuracy:.1f}%"
        )
    else:
        print("\nNo comparisons found!")


def create_combined_visualization_plots():
    """Create combined plots for t-SNE and distribution visualizations side by side"""
    viz_path = Path("visualizations")

    # Define the specific datasets and models as requested
    datasets = ["google", "air", "eeg", "appliances"]  # Columns
    models = ["arima", "exp_smooth", "timevae", "timegan", "timetransformer"]  # Rows

    # Create one figure with both visualizations side by side
    fig, axes = plt.subplots(
        len(models), len(datasets) * 2, figsize=(20, 10), sharey=True, sharex=True
    )

    # Ensure axes is always a 2D array
    if len(models) == 1:
        axes = axes.reshape(1, -1)
    if len(datasets) == 1:
        axes = axes.reshape(-1, 1)

    # Load and display both t-SNE and distribution plots
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            # Skip fourierflow as requested
            if model == "fourierflow":
                continue

            # Skip timetransformer for appliances dataset
            if dataset == "appliances" and model == "timetransformer":
                # Create empty subplot
                axes[i, j * 2].axis("off")
                axes[i, j * 2 + 1].axis("off")
                continue

            # Load t-SNE plot (left subplot)
            tsne_file = viz_path / dataset / f"{dataset}_{model}_tsne.png"
            if tsne_file.exists():
                img = mpimg.imread(tsne_file)
                axes[i, j * 2].imshow(img)
                axes[i, j * 2].axis("off")
            else:
                axes[i, j * 2].axis("off")
                axes[i, j * 2].text(
                    0.5,
                    0.5,
                    f"No t-SNE data\n{dataset}_{model}",
                    ha="center",
                    va="center",
                    transform=axes[i, j * 2].transAxes,
                )

            # Load distribution plot (right subplot)
            dist_file = viz_path / dataset / f"{dataset}_{model}_distribution.png"
            if dist_file.exists():
                img = mpimg.imread(dist_file)
                axes[i, j * 2 + 1].imshow(img)
                axes[i, j * 2 + 1].axis("off")
            else:
                axes[i, j * 2 + 1].axis("off")
                axes[i, j * 2 + 1].text(
                    0.5,
                    0.5,
                    f"No distribution data\n{dataset}_{model}",
                    ha="center",
                    va="center",
                    transform=axes[i, j * 2 + 1].transAxes,
                )

            # Set row labels (model names) on the left side
            if j == 0:  # First column (left)
                axes[i, j * 2].set_ylabel(
                    f"{model}", fontsize=12, fontweight="bold", rotation=0, ha="right"
                )
                axes[i, j * 2 + 1].set_ylabel("", fontsize=12, fontweight="bold")

    # Add overall title
    fig.suptitle("t-SNE and Distribution", fontsize=16, fontweight="bold")

    # Set x-tick labels (dataset names) at the bottom using figure.text
    for j, dataset in enumerate(datasets):
        # Position x-label below the bottom row, centered between t-SNE and Distribution plots
        # Each dataset column spans 2 subplots, so center between them
        x_pos = (j * 2 + 1) / (len(datasets) * 2)  # Center of each dataset column
        y_pos = 0.05  # Small padding from bottom
        fig.text(
            x_pos,
            y_pos,
            dataset.capitalize(),
            fontsize=18,
            fontweight="bold",
            ha="center",
            va="top",
        )

    # Set y-tick labels (model names) on the left side using figure.text
    for i, model in enumerate(models):
        # Calculate equal spacing between labels regardless of subplot size
        # Available height from 0.10 to 0.95 (accounting for title and bottom padding)
        total_height = 0.70
        label_spacing = total_height / (len(models) - 1)  # Space between labels

        # Position each label with equal spacing
        y_pos = 0.85 - (i * label_spacing)

        # Position label to the left with consistent padding
        x_pos = 0.02  # Small padding from left edge

        fig.text(
            x_pos,
            y_pos,
            model,
            fontsize=18,
            fontweight="bold",
            ha="right",
            va="center",
            rotation=90,  # Vertical text
        )

    # Adjust layout to accommodate the labels with tighter spacing
    plt.subplots_adjust(
        top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.0, wspace=0.05
    )

    # Save figure
    fig.savefig("combined_visualization_plots.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # print("Creating results bar plots...")
    # create_results_bar_plots()

    # print("Creating AutoML vs TSG bar plots...")
    # create_automl_bar_plots()

    # print("Creating combined visualization plots...")
    # create_combined_visualization_plots()

    print("Calculating accuracy...")
    calculate_accuracy()

    print("All plots created successfully!")
    print("Files saved:")
    print("- results_bar_plots.png")
    print("- combined_tsne_plots.png")
    print("- combined_distribution_plots.png")
