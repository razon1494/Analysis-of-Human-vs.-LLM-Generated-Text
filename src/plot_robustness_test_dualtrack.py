import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("results/robustness_test_dualtrack.csv")

    base = df[df["split"] == "P0_test"].iloc[0]
    p0_acc, p0_f1 = base["acc"], base["f1"]

    std = df[df["split"].isin(["P1_test_standard", "P2_test_standard"])].set_index("split")
    sim = df[df["split"].isin(["P1_test_simplified", "P2_test_simplified"])].set_index("split")

    stages = ["P0", "P1", "P2"]

    acc_std = [p0_acc, std.loc["P1_test_standard"]["acc"], std.loc["P2_test_standard"]["acc"]]
    acc_sim = [p0_acc, sim.loc["P1_test_simplified"]["acc"], sim.loc["P2_test_simplified"]["acc"]]

    f1_std = [p0_f1, std.loc["P1_test_standard"]["f1"], std.loc["P2_test_standard"]["f1"]]
    f1_sim = [p0_f1, sim.loc["P1_test_simplified"]["f1"], sim.loc["P2_test_simplified"]["f1"]]

    os.makedirs("figures", exist_ok=True)

    plt.figure()
    plt.plot(stages, acc_std, marker="o", label="Standard paraphrase")
    plt.plot(stages, acc_sim, marker="o", label="Simplified (non-expert) paraphrase")
    plt.xlabel("Paraphrase stage")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Under Paraphrasing: Standard vs Simplified (Test Set)")
    plt.grid(True, linewidth=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/accuracy_test_dualtrack.png", dpi=200)

    plt.figure()
    plt.plot(stages, f1_std, marker="o", label="Standard paraphrase")
    plt.plot(stages, f1_sim, marker="o", label="Simplified (non-expert) paraphrase")
    plt.xlabel("Paraphrase stage")
    plt.ylabel("F1")
    plt.title("F1 Under Paraphrasing: Standard vs Simplified (Test Set)")
    plt.grid(True, linewidth=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/f1_test_dualtrack.png", dpi=200)

    print("Saved: figures/accuracy_test_dualtrack.png")
    print("Saved: figures/f1_test_dualtrack.png")

if __name__ == "__main__":
    main()
