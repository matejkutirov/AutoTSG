import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, Any
import warnings

warnings.filterwarnings("ignore")


def make_sure_path_exist(file_path):
    """Create directory if it doesn't exist"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


def visualize_tsne(ori_data, gen_data, result_path, save_file_name):
    """Generate t-SNE visualization comparing original vs generated data"""

    sample_num = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:sample_num]

    ori_data = ori_data[idx]
    gen_data = gen_data[idx]

    prep_data = np.mean(ori_data, axis=1)
    prep_data_hat = np.mean(gen_data, axis=1)

    colors = ["C0" for i in range(sample_num)] + ["C1" for i in range(sample_num)]

    prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

    try:
        # Remove n_iter parameter as it's deprecated in newer scikit-learn versions
        tsne = TSNE(n_components=2, verbose=0, perplexity=30, random_state=42)
        tsne_results = tsne.fit_transform(prep_data_final)
    except Exception as e:
        print(f"Error in t-SNE computation for {save_file_name}: {e}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))

    ax.scatter(
        tsne_results[:sample_num, 0],
        tsne_results[:sample_num, 1],
        c=colors[:sample_num],
        alpha=0.5,
        label="Original",
        s=5,
    )
    ax.scatter(
        tsne_results[sample_num:, 0],
        tsne_results[sample_num:, 1],
        c=colors[sample_num:],
        alpha=0.5,
        label="Generated",
        s=5,
    )

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for pos in ["top", "bottom", "left", "right"]:
        ax.spines[pos].set_visible(False)

    save_path = os.path.join(result_path, f"{save_file_name}_tsne.png")
    make_sure_path_exist(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved to: {save_path}")


def visualize_distribution(ori_data, gen_data, result_path, save_file_name):
    """Generate distribution visualization comparing original vs generated data"""

    sample_num = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:sample_num]

    ori_data = ori_data[idx]
    gen_data = gen_data[idx]

    prep_data = np.mean(ori_data, axis=1)
    prep_data_hat = np.mean(gen_data, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    sns.kdeplot(prep_data.flatten(), color="C0", linewidth=2, label="Original", ax=ax)

    # Plotting KDE for generated data on the same axes
    sns.kdeplot(
        prep_data_hat.flatten(),
        color="C1",
        linewidth=2,
        linestyle="--",
        label="Generated",
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)

    # Remove top and right spines for cleaner look
    for pos in ["top", "right"]:
        ax.spines[pos].set_visible(False)

    save_path = os.path.join(result_path, f"{save_file_name}_distribution.png")
    make_sure_path_exist(save_path)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Distribution plot saved to: {save_path}")


def create_visualizations(model_results: Dict[str, Any], dataset: str) -> None:
    """Create t-SNE and distribution visualizations for a trained model"""

    # Create visualizations directory
    viz_dir = "visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    # Create dataset-specific subdirectory
    dataset_dir = os.path.join(viz_dir, dataset)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Get model information
    model_type = model_results.get("model_type", "unknown")
    train_sequences = model_results.get("train_sequences")
    generated_data = model_results.get("generated_data")

    if train_sequences is None or generated_data is None:
        print(f"Warning: Cannot create visualizations for {model_type} - missing data")
        return

    # Validate data shapes
    if len(train_sequences.shape) < 3 or len(generated_data.shape) < 3:
        print(
            f"Warning: Cannot create visualizations for {model_type} - data must be 3D (samples, time, features)"
        )
        return

    if train_sequences.shape[0] < 10 or generated_data.shape[0] < 10:
        print(
            f"Warning: Cannot create visualizations for {model_type} - insufficient samples (need at least 10)"
        )
        return

    # Create unique filename for this model
    save_file_name = f"{dataset}_{model_type}"

    try:
        # Generate t-SNE visualization
        visualize_tsne(train_sequences, generated_data, dataset_dir, save_file_name)

        # Generate distribution visualization
        visualize_distribution(
            train_sequences, generated_data, dataset_dir, save_file_name
        )

        print(f"✓ Visualizations created for {model_type} on {dataset} dataset")

    except Exception as e:
        print(f"Error creating visualizations for {model_type}: {e}")
        import traceback

        traceback.print_exc()
