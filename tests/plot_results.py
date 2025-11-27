"""
Plotting Script for Test Results

Creates comprehensive visualizations from CSV test results:
1. Model comparison heatmap (score by image)
2. Discrimination gap comparison
3. Preprocessing impact analysis
4. Timing performance comparison
5. Accuracy comparison
6. Score distribution by model
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 10

OUTPUT_DIR = Path(__file__).parent / "output"  # Relative to test file
CSV_DIR = OUTPUT_DIR / "csv"
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load all CSV files."""
    print("📂 Loading CSV data...")

    # Check if CSV files exist
    required_files = [
        CSV_DIR / "results_summary.csv",
        CSV_DIR / "results_by_image.csv",
        CSV_DIR / "results_by_model.csv",
    ]

    missing_files = [f for f in required_files if not f.exists()]

    if missing_files:
        print("\n" + "=" * 100)
        print("❌ ERROR: CSV files not found!")
        print("=" * 100)
        print("\nMissing files:")
        for f in missing_files:
            print(f"   ❌ {f}")
        print("\nExpected location: output/csv/")
        print("\n" + "=" * 100)
        print("SOLUTION: Run the comprehensive test first to generate CSV files")
        print("=" * 100)
        print("\nRun this command:")
        print("   uv run python tests/test_comprehensive.py")
        print("\nThen run this plotting script:")
        print("   uv run python tests/plot_results.py")
        print("=" * 100 + "\n")
        raise FileNotFoundError(
            f"Missing CSV files in {CSV_DIR}. Please run test_comprehensive.py first!"
        )

    data = {}
    data["summary"] = pd.read_csv(CSV_DIR / "results_summary.csv")
    data["by_image"] = pd.read_csv(CSV_DIR / "results_by_image.csv")
    data["by_model"] = pd.read_csv(CSV_DIR / "results_by_model.csv")

    print(f"   ✅ Loaded {len(data['summary'])} test results from {CSV_DIR}")
    return data


def plot_model_comparison_heatmap(df_summary):
    """Plot heatmap of similarity scores for each model-image combination."""
    print("\n📊 Generating heatmap: Model vs Image scores...")

    # Filter for WITH preprocessing
    df_prep = df_summary[df_summary["preprocessing"] == True].copy()

    # Create pivot table
    pivot = df_prep.pivot_table(
        values="similarity_score", index="model_name", columns="image_name", aggfunc="mean"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0.65,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Similarity Score"},
        ax=ax,
    )

    ax.set_title(
        "Model Performance Heatmap (WITH Preprocessing)\nHigher scores = Better for Wawel, Lower = Better for Unrelated",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Test Image", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    filepath = PLOT_DIR / "1_heatmap_model_vs_image.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved: {filepath}")
    plt.close()


def plot_discrimination_gap(df_by_model):
    """Plot discrimination gap comparison across models."""
    print("\n📊 Generating plot: Discrimination gap comparison...")

    # Sort by discrimination gap
    df_sorted = df_by_model.sort_values("discrimination_gap", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar plot
    x = np.arange(len(df_sorted))
    width = 0.35

    df_with = df_sorted[df_sorted["preprocessing"] == True]
    df_without = df_sorted[df_sorted["preprocessing"] == False]

    # Group by model to align bars
    models = df_sorted["model_name"].unique()
    x_pos = np.arange(len(models))

    with_gaps = []
    without_gaps = []

    for model in models:
        with_val = df_with[df_with["model_name"] == model]["discrimination_gap"].values
        without_val = df_without[df_without["model_name"] == model]["discrimination_gap"].values
        with_gaps.append(with_val[0] if len(with_val) > 0 else 0)
        without_gaps.append(without_val[0] if len(without_val) > 0 else 0)

    bars1 = ax.bar(x_pos - width / 2, with_gaps, width, label="WITH Preprocessing", color="#2ecc71")
    bars2 = ax.bar(
        x_pos + width / 2, without_gaps, width, label="WITHOUT Preprocessing", color="#3498db"
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Add threshold lines
    ax.axhline(y=0.30, color="green", linestyle="--", alpha=0.5, label="Excellent (>30%)")
    ax.axhline(y=0.15, color="orange", linestyle="--", alpha=0.5, label="Good (>15%)")
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.5, label="Minimum (>5%)")

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Discrimination Gap (Wawel - Unrelated)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Discrimination Gap by Model\nHigher = Better ability to distinguish locations",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = PLOT_DIR / "2_discrimination_gap.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved: {filepath}")
    plt.close()


def plot_preprocessing_impact(df_by_model):
    """Plot preprocessing impact analysis."""
    print("\n📊 Generating plot: Preprocessing impact...")

    models = df_by_model["model_name"].unique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Extract data
    wawel_with = []
    wawel_without = []
    unrelated_with = []
    unrelated_without = []

    for model in models:
        df_model = df_by_model[df_by_model["model_name"] == model]

        with_row = (
            df_model[df_model["preprocessing"] == True].iloc[0]
            if len(df_model[df_model["preprocessing"] == True]) > 0
            else None
        )
        without_row = (
            df_model[df_model["preprocessing"] == False].iloc[0]
            if len(df_model[df_model["preprocessing"] == False]) > 0
            else None
        )

        if with_row is not None and without_row is not None:
            wawel_with.append(with_row["avg_wawel_score"])
            wawel_without.append(without_row["avg_wawel_score"])
            unrelated_with.append(with_row["avg_unrelated_score"])
            unrelated_without.append(without_row["avg_unrelated_score"])

    x = np.arange(len(models))
    width = 0.35

    # Plot 1: Wawel scores
    bars1 = ax1.bar(
        x - width / 2, wawel_with, width, label="WITH Preprocessing", color="#2ecc71", alpha=0.8
    )
    bars2 = ax1.bar(
        x + width / 2,
        wawel_without,
        width,
        label="WITHOUT Preprocessing",
        color="#95a5a6",
        alpha=0.8,
    )

    ax1.set_xlabel("Model", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Average Similarity Score", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Wawel Images: Preprocessing Impact\n(Higher = Better)", fontsize=12, fontweight="bold"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(y=0.65, color="red", linestyle="--", alpha=0.5, label="Threshold")

    # Plot 2: Unrelated scores
    bars3 = ax2.bar(
        x - width / 2, unrelated_with, width, label="WITH Preprocessing", color="#e74c3c", alpha=0.8
    )
    bars4 = ax2.bar(
        x + width / 2,
        unrelated_without,
        width,
        label="WITHOUT Preprocessing",
        color="#c0392b",
        alpha=0.8,
    )

    ax2.set_xlabel("Model", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Average Similarity Score", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Unrelated Images: Preprocessing Impact\n(Lower = Better)", fontsize=12, fontweight="bold"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(y=0.65, color="red", linestyle="--", alpha=0.5, label="Threshold")

    plt.tight_layout()
    filepath = PLOT_DIR / "3_preprocessing_impact.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved: {filepath}")
    plt.close()


def plot_timing_comparison(df_by_model):
    """Plot timing performance comparison."""
    print("\n📊 Generating plot: Timing comparison...")

    models = df_by_model["model_name"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.35

    # Extract timing data
    with_times = []
    without_times = []

    for model in models:
        df_model = df_by_model[df_by_model["model_name"] == model]
        with_row = (
            df_model[df_model["preprocessing"] == True].iloc[0]
            if len(df_model[df_model["preprocessing"] == True]) > 0
            else None
        )
        without_row = (
            df_model[df_model["preprocessing"] == False].iloc[0]
            if len(df_model[df_model["preprocessing"] == False]) > 0
            else None
        )

        if with_row is not None and without_row is not None:
            with_times.append(with_row["avg_inference_time_s"])
            without_times.append(without_row["avg_inference_time_s"])

    bars1 = ax.bar(x - width / 2, with_times, width, label="WITH Preprocessing", color="#3498db")
    bars2 = ax.bar(
        x + width / 2, without_times, width, label="WITHOUT Preprocessing", color="#95a5a6"
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height * 1000:.1f}ms",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Inference Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Inference Time Comparison\n(Lower = Faster)", fontsize=14, fontweight="bold", pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filepath = PLOT_DIR / "4_timing_comparison.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved: {filepath}")
    plt.close()


def plot_accuracy_comparison(df_by_model):
    """Plot accuracy comparison."""
    print("\n📊 Generating plot: Accuracy comparison...")

    models = df_by_model["model_name"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.35

    # Extract accuracy data
    with_acc = []
    without_acc = []

    for model in models:
        df_model = df_by_model[df_by_model["model_name"] == model]
        with_row = (
            df_model[df_model["preprocessing"] == True].iloc[0]
            if len(df_model[df_model["preprocessing"] == True]) > 0
            else None
        )
        without_row = (
            df_model[df_model["preprocessing"] == False].iloc[0]
            if len(df_model[df_model["preprocessing"] == False]) > 0
            else None
        )

        if with_row is not None and without_row is not None:
            with_acc.append(with_row["accuracy"] * 100)
            without_acc.append(without_row["accuracy"] * 100)

    bars1 = ax.bar(x - width / 2, with_acc, width, label="WITH Preprocessing", color="#2ecc71")
    bars2 = ax.bar(
        x + width / 2, without_acc, width, label="WITHOUT Preprocessing", color="#3498db"
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Classification Accuracy by Model\n(Correctly identifying Wawel vs Unrelated)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=100, color="green", linestyle="--", alpha=0.5, label="Perfect")
    ax.axhline(y=90, color="orange", linestyle="--", alpha=0.5, label="Excellent")

    plt.tight_layout()
    filepath = PLOT_DIR / "5_accuracy_comparison.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved: {filepath}")
    plt.close()


def plot_score_distribution(df_summary):
    """Plot score distribution by model."""
    print("\n📊 Generating plot: Score distribution...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    models = df_summary["model_name"].unique()

    for idx, model in enumerate(models[:4]):  # Limit to 4 models
        ax = axes[idx]

        df_model = df_summary[
            (df_summary["model_name"] == model) & (df_summary["preprocessing"] == True)
        ]

        # Separate Wawel and Unrelated
        wawel_scores = df_model[df_model["is_wawel"] == True]["similarity_score"]
        unrelated_scores = df_model[df_model["is_wawel"] == False]["similarity_score"]

        # Plot histograms
        ax.hist(
            wawel_scores,
            bins=20,
            alpha=0.6,
            label="Wawel (Should Pass)",
            color="#2ecc71",
            edgecolor="black",
        )
        ax.hist(
            unrelated_scores,
            bins=20,
            alpha=0.6,
            label="Unrelated (Should Fail)",
            color="#e74c3c",
            edgecolor="black",
        )

        # Add threshold line
        ax.axvline(x=0.65, color="black", linestyle="--", linewidth=2, label="Threshold (0.65)")

        ax.set_xlabel("Similarity Score", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(f"{model}\nScore Distribution", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Similarity Score Distributions by Model\n(WITH Preprocessing)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    filepath = PLOT_DIR / "6_score_distribution.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved: {filepath}")
    plt.close()


def plot_per_image_performance(df_by_image):
    """Plot performance per image across all models."""
    print("\n📊 Generating plot: Per-image performance...")

    # Filter for WITH preprocessing
    df_prep = df_by_image[df_by_image["preprocessing"] == True]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Sort by type (wawel first, then unrelated)
    df_sorted = df_prep.sort_values(["is_wawel", "avg_score"], ascending=[False, False])

    x = np.arange(len(df_sorted))

    # Plot 1: Average scores
    colors = ["#2ecc71" if is_wawel else "#e74c3c" for is_wawel in df_sorted["is_wawel"]]
    bars = ax1.bar(x, df_sorted["avg_score"], color=colors, alpha=0.7, edgecolor="black")

    # Add error bars (std)
    ax1.errorbar(
        x,
        df_sorted["avg_score"],
        yerr=df_sorted["std_score"],
        fmt="none",
        ecolor="black",
        capsize=3,
        alpha=0.5,
    )

    ax1.axhline(y=0.65, color="black", linestyle="--", linewidth=2, label="Threshold")
    ax1.set_ylabel("Average Similarity Score", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Average Score per Image (WITH Preprocessing)\nGreen=Wawel, Red=Unrelated",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_sorted["image_name"], rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Plot 2: Accuracy per image
    bars2 = ax2.bar(x, df_sorted["accuracy"] * 100, color=colors, alpha=0.7, edgecolor="black")

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Image", fontsize=11, fontweight="bold")
    ax2.set_title("Classification Accuracy per Image", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_sorted["image_name"], rotation=45, ha="right")
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(y=100, color="green", linestyle="--", alpha=0.5)

    plt.tight_layout()
    filepath = PLOT_DIR / "7_per_image_performance.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"   ✅ Saved: {filepath}")
    plt.close()


def create_summary_report(data):
    """Create a text summary report."""
    print("\n📄 Generating summary report...")

    df_summary = data["summary"]
    df_by_model = data["by_model"]

    report_path = OUTPUT_DIR / "TEST_REPORT.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE TEST RESULTS SUMMARY\n")
        f.write("=" * 100 + "\n\n")

        # Overall stats
        total_tests = len(df_summary)
        total_correct = df_summary["is_correct"].sum()
        overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0

        f.write(f"Total Tests: {total_tests}\n")
        f.write(f"Correct Classifications: {total_correct}\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n\n")

        # Best model
        f.write("=" * 100 + "\n")
        f.write("BEST MODEL (WITH Preprocessing)\n")
        f.write("=" * 100 + "\n\n")

        df_with_prep = df_by_model[df_by_model["preprocessing"] == True]
        best_model = df_with_prep.loc[df_with_prep["discrimination_gap"].idxmax()]

        f.write(f"Model: {best_model['model_name']}\n")
        f.write(
            f"Discrimination Gap: {best_model['discrimination_gap']:.4f} ({best_model['discrimination_gap'] * 100:.1f}%)\n"
        )
        f.write(f"Average Wawel Score: {best_model['avg_wawel_score']:.4f}\n")
        f.write(f"Average Unrelated Score: {best_model['avg_unrelated_score']:.4f}\n")
        f.write(f"Accuracy: {best_model['accuracy'] * 100:.1f}%\n")
        f.write(f"Average Inference Time: {best_model['avg_inference_time_s'] * 1000:.1f}ms\n\n")

        # All models comparison
        f.write("=" * 100 + "\n")
        f.write("ALL MODELS COMPARISON (WITH Preprocessing)\n")
        f.write("=" * 100 + "\n\n")

        f.write(
            f"{'Model':<30} {'Gap':<10} {'Wawel':<10} {'Unrel':<10} {'Acc':<8} {'Time(ms)':<10}\n"
        )
        f.write("-" * 100 + "\n")

        df_sorted = df_with_prep.sort_values("discrimination_gap", ascending=False)
        for _, row in df_sorted.iterrows():
            f.write(
                f"{row['model_name']:<30} "
                f"{row['discrimination_gap']:.4f}    "
                f"{row['avg_wawel_score']:.4f}    "
                f"{row['avg_unrelated_score']:.4f}    "
                f"{row['accuracy'] * 100:.1f}%    "
                f"{row['avg_inference_time_s'] * 1000:.1f}ms\n"
            )

        f.write("\n" + "=" * 100 + "\n")
        f.write("RECOMMENDATION\n")
        f.write("=" * 100 + "\n\n")

        if best_model["discrimination_gap"] > 0.30:
            f.write("🎯 EXCELLENT: Strong discrimination capability (gap > 30%)\n")
            f.write("✅ System is PRODUCTION READY\n\n")
        elif best_model["discrimination_gap"] > 0.15:
            f.write("✅ GOOD: Acceptable discrimination capability (gap > 15%)\n")
            f.write("✅ System is PRODUCTION READY with careful monitoring\n\n")
        else:
            f.write("⚠️  WARNING: Weak discrimination (gap < 15%)\n")
            f.write("Consider improvements before production deployment\n\n")

        f.write("Recommended Production Config:\n")
        f.write(f"  Model: {best_model['model_name']}\n")
        f.write("  Threshold: 0.65\n")
        f.write("  Preprocessing: ENABLE for user queries\n")

    print(f"   ✅ Saved: {report_path}")


def main():
    """Main plotting function."""
    print("\n" + "=" * 100)
    print("📊 GENERATING PLOTS FROM TEST RESULTS".center(100))
    print("=" * 100)

    # Load data
    data = load_data()

    # Generate all plots
    plot_model_comparison_heatmap(data["summary"])
    plot_discrimination_gap(data["by_model"])
    plot_preprocessing_impact(data["by_model"])
    plot_timing_comparison(data["by_model"])
    plot_accuracy_comparison(data["by_model"])
    plot_score_distribution(data["summary"])
    plot_per_image_performance(data["by_image"])

    # Create summary report
    create_summary_report(data)

    print("\n" + "=" * 100)
    print("✅ ALL PLOTS GENERATED SUCCESSFULLY!".center(100))
    print(f"📁 Output directory: {PLOT_DIR.absolute()}".center(100))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
