import json
import matplotlib.pyplot as plt


def plot_behavioral_stability(output_file):
    with open(output_file, "r") as f:
        results = json.load(f)

    average_behaviors = eval(results["average_behaviors"])
    # first_behaviors = eval(results.get("first_behaviors", "[]"))
    # worst_behaviors = eval(results.get("worst_behaviors", "[]"))

    average_behaviors = [
        behavior for behavior in average_behaviors if behavior is not None
    ]
    # first_behaviors = [behavior for behavior in first_behaviors if behavior is not None]
    # worst_behaviors = [behavior for behavior in worst_behaviors if behavior is not None]

    total_average = sum(average_behaviors) / len(average_behaviors)
    # average_first = sum(first_behaviors) / len(first_behaviors)
    # average_worst = sum(worst_behaviors) / len(worst_behaviors)

    # pos_behaviors = [behavior for behavior in average_behaviors if behavior > 0.5]
    # pos_behaviors_frac = len(pos_behaviors) / len(average_behaviors)
    # neg_behaviors_frac = 1 - pos_behaviors_frac

    plt.figure(figsize=(8, 5))
    plt.hist(average_behaviors, bins=10, range=(0, 1), edgecolor="white")

    model_name = results.get("model_name", "model").split("/")[-1]
    dataset_name = results.get("dataset_name", "dataset")
    subset = results.get("subset", None)
    subset_str = f" (n={subset})" if subset else ""

    plt.title(
        f"Behavioral Stability: {model_name} on {dataset_name}{subset_str}", pad=20
    )

    # Add total average text below the title, above the plot
    plt.text(
        0.5,
        1.0,
        f"Total average: {total_average:.2f}",
        ha="center",
        va="bottom",
        transform=plt.gca().transAxes,
        fontsize=14,
        fontweight="bold",
    )
    #   Average first: {average_first:.2f}
    #   Average worst: {average_worst:.2f}

    plt.xlabel("Behavioral Stability")
    plt.xticks([i / 10 for i in range(11)])
    plt.ylabel("Frequency")
    plt.xlim(0, 1)

    output_png = output_file.replace(".json", ".png")
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    print(f"Histogram saved to {output_png}")
    plt.close()

    return output_png
