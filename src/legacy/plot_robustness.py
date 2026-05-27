import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("results/robustness_test.csv")
    df = df.set_index("split").loc[["P0_test", "P1_test", "P2_test"]].reset_index()

    # Accuracy plot
    plt.figure()
    plt.plot(df["split"], df["acc"], marker="o")
    plt.xlabel("Condition")
    plt.ylabel("Accuracy")
    plt.title("Detector Accuracy Under Iterative Paraphrasing (Test Set)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig("figures/accuracy_vs_paraphrase_test.png", dpi=200)

    # F1 plot
    plt.figure()
    plt.plot(df["split"], df["f1"], marker="o")
    plt.xlabel("Condition")
    plt.ylabel("F1")
    plt.title("Detector F1 Under Iterative Paraphrasing (Test Set)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig("figures/f1_vs_paraphrase_test.png", dpi=200)

    print("Saved: figures/accuracy_vs_paraphrase_test.png")
    print("Saved: figures/f1_vs_paraphrase_test.png")

if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    main()
