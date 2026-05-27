import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_track(df, track_name, outfile):
    # track_name in {"standard", "simplified"}
    stages = ["P0_test", f"P1_test_{track_name}", f"P2_test_{track_name}"]
    # For display labels
    stage_labels = ["P0", "P1", "P2"]

    buckets = ["Easy", "Medium", "Hard"]

    plt.figure(figsize=(8, 5))
    for b in buckets:
        y = []
        for s in stages:
            v = df[(df["split"] == s) & (df["bucket"] == b)]["f1"].values
            y.append(float(v[0]) if len(v) else float("nan"))
        plt.plot(stage_labels, y, marker="o", label=b)

    plt.title(f"F1 vs Paraphrase Stage by Hardness Bucket ({track_name.capitalize()} Track)")
    plt.xlabel("Paraphrase stage")
    plt.ylabel("F1")
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print("Saved:", outfile)

def main():
    df = pd.read_csv("results/hardness_buckets_test.csv")

    # Standard track
    plot_track(df, "standard", "figures/f1_hardness_standard.png")

    # Simplified track
    plot_track(df, "simplified", "figures/f1_hardness_simplified.png")

if __name__ == "__main__":
    main()
