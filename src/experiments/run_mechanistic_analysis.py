"""
run_mechanistic_analysis.py  (P8)
----------------------------------
Qualitative + mechanistic explanation for the Hard-bucket F1 collapse under
simplified paraphrasing.

Questions answered
------------------
  Q1. Which flip cases drive the Hard-bucket collapse?
      (correctly predicted at P0 → misclassified at P2_simplified)
  Q2. Which char/word n-gram features drove P0 correct predictions?
      (SHAP Linear attribution on the two TF-IDF detectors)
  Q3. Do those features survive into P2 text?
      (feature survival rate)
  Q4. What *kind* of features get destroyed?
      (qualitative labelling + aggregate feature drift)
  Q5. Why is RoBERTa more robust?
      (brief vocabulary richness comparison)

Outputs (results/eval/)
-----------------------
  mechanistic_flip_cases.csv      — flip examples with P0/P2 texts + probs
  mechanistic_feature_drift.csv   — top-K feature mean TF-IDF weight P0→P2
  mechanistic_shap_summary.csv    — per-detector top SHAP features (Hard flips)

Figures
-------
  figures/fig20_shap_top_features.png     — top SHAP features at P0 vs P2
  figures/fig21_feature_survival.png      — % top features surviving P0→P2
  figures/fig22_vocab_drift.png           — vocabulary richness Hard bucket
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lib import paths
from lib.io import load_jsonl, load_test_ids, to_xy, label_to_int
from lib.detectors import build_word_tfidf_lr, build_char_tfidf_lr
from lib.hardness import hardness_readability

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200,
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

TOP_K      = 20   # top features to analyse per detector
N_BG       = 200  # background sample size for SHAP LinearExplainer


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data():
    test_ids  = load_test_ids(paths.TEST_IDS)
    train_ids = load_test_ids(paths.TRAIN_IDS)
    p0_all    = load_jsonl(paths.P0_PATH)

    p0_test   = [r for r in p0_all if r["id"] in test_ids]
    train_rows = [r for r in p0_all if r["id"] in train_ids]

    p2_sim_path = ROOT / "data" / "p2" / "p2_test_simplified.jsonl"
    p1_sim_path = ROOT / "data" / "p1" / "p1_test_simplified.jsonl"
    p2_std_path = ROOT / "data" / "p2" / "p2_test.jsonl"

    p2_sim = load_jsonl(p2_sim_path)
    p1_sim = load_jsonl(p1_sim_path)
    p2_std = load_jsonl(p2_std_path)

    return train_rows, p0_test, p1_sim, p2_sim, p2_std


# ── Hard bucket IDs ───────────────────────────────────────────────────────────

def get_hard_ids(p0_test: list) -> set:
    texts = [r["text"] for r in p0_test]
    fk    = hardness_readability(texts)
    return {r["id"] for r, b in zip(p0_test, fk.buckets) if b == "Hard"}


# ── Flip case finder ──────────────────────────────────────────────────────────

def find_flips(p0_rows: list, p2_rows: list, detector, hard_ids: set) -> list:
    """Return rows where prediction is correct at P0 but wrong at P2 (Hard bucket)."""
    id_to_p2 = {r["id"]: r for r in p2_rows}

    flips = []
    for r0 in p0_rows:
        rid = r0["id"]
        if rid not in hard_ids or rid not in id_to_p2:
            continue
        r2 = id_to_p2[rid]

        y_true = label_to_int(r0["label"])   # 0=human, 1=LLM
        p0_prob = float(detector.predict_proba([r0["text"]])[0])
        p2_prob = float(detector.predict_proba([r2["text"]])[0])
        p0_pred = int(p0_prob >= 0.5)
        p2_pred = int(p2_prob >= 0.5)

        if p0_pred == y_true and p2_pred != y_true:
            flips.append({
                "id":       rid,
                "label":    y_true,
                "text_p0":  r0["text"],
                "text_p2":  r2["text"],
                "prob_p0":  round(p0_prob, 4),
                "prob_p2":  round(p2_prob, 4),
                "delta":    round(p2_prob - p0_prob, 4),
            })
    return flips


# ── SHAP attribution ─────────────────────────────────────────────────────────

def shap_top_features(
    bundle,
    texts_p0: list[str],
    texts_p2: list[str],
    bg_texts: list[str],
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Compute mean |SHAP| for the top-k most impactful features at P0 and P2.

    For a linear model, SHAP values are exact:
        shap_i(x) = coef_i * (x_i - E[x_i])
    where E[x_i] is the mean feature value over the background set.
    This avoids all SHAP library API compatibility issues.
    """
    vec        = bundle.vectorizer
    clf        = bundle.classifier
    coef       = clf.coef_[0]                            # (n_features,)
    feat_names = np.array(vec.get_feature_names_out())

    # Compute background mean from training texts
    n_bg   = min(N_BG, len(bg_texts))
    bg_mat = vec.transform(bg_texts[:n_bg])
    bg_mean = np.asarray(bg_mat.mean(axis=0)).flatten()  # (n_features,)

    rows = []
    for stage, texts in [("P0", texts_p0), ("P2_simplified", texts_p2)]:
        if not texts:
            continue
        Xv        = vec.transform(texts)
        Xd        = np.asarray(Xv.todense())             # (n, n_features)
        # SHAP values: coef * (x - E[x])
        sv        = coef * (Xd - bg_mean)                # (n, n_features)
        mean_shap = np.abs(sv).mean(axis=0)              # (n_features,)
        top_idx   = np.argsort(mean_shap)[::-1][:top_k]
        for rank, idx in enumerate(top_idx, 1):
            rows.append({
                "stage":      stage,
                "rank":       rank,
                "feature":    feat_names[idx],
                "mean_shap":  round(float(mean_shap[idx]), 6),
            })
    return pd.DataFrame(rows)


# ── Feature survival analysis ─────────────────────────────────────────────────

def feature_survival(
    bundle,
    flip_rows: list,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    For each flip case, check whether the top-K features at P0 still appear
    (non-zero TF-IDF weight) in the P2 version of the same text.
    """
    if not flip_rows:
        return pd.DataFrame()

    vec        = bundle.vectorizer
    feat_names = np.array(vec.get_feature_names_out())
    coef       = bundle.classifier.coef_[0]                  # (n_features,)

    # Top-K features by |coef| — these are the "detector cues"
    top_idx = np.argsort(np.abs(coef))[::-1][:top_k]
    top_feat_names = feat_names[top_idx]

    rows = []
    for f in flip_rows:
        x0 = vec.transform([f["text_p0"]])
        x2 = vec.transform([f["text_p2"]])

        for idx, fname in zip(top_idx, top_feat_names):
            w0 = float(x0[0, idx])
            w2 = float(x2[0, idx])
            rows.append({
                "id":         f["id"],
                "label":      f["label"],
                "feature":    fname,
                "weight_p0":  round(w0, 6),
                "weight_p2":  round(w2, 6),
                "survived":   w2 > 0,
                "coef":       round(float(coef[idx]), 6),
            })
    return pd.DataFrame(rows)


# ── Feature mean weight drift ─────────────────────────────────────────────────

def feature_drift_table(
    bundle,
    hard_p0: list,
    hard_p1: list,
    hard_p2: list,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Mean TF-IDF weight of top-K |coef| features across P0/P1/P2 (Hard LLM rows)."""
    vec       = bundle.vectorizer
    coef      = bundle.classifier.coef_[0]
    fn        = np.array(vec.get_feature_names_out())
    top_idx   = np.argsort(np.abs(coef))[::-1][:top_k]

    # Keep only LLM-generated Hard rows for the drift analysis (label==1)
    def llm_texts(rows):
        return [r["text"] for r in rows if r["label"] == "llm"]

    rows = []
    for stage, rlist in [("P0", hard_p0), ("P1_simplified", hard_p1), ("P2_simplified", hard_p2)]:
        texts = llm_texts(rlist)
        if not texts:
            continue
        Xv     = vec.transform(texts)
        Xd     = Xv.toarray()                      # (n, n_features)
        means  = Xd[:, top_idx].mean(axis=0)       # (top_k,)
        for rank, (idx, m) in enumerate(zip(top_idx, means), 1):
            rows.append({
                "stage":        stage,
                "rank":         rank,
                "feature":      fn[idx],
                "mean_weight":  round(float(m), 6),
                "coef":         round(float(coef[idx]), 6),
                "llm_positive": coef[idx] > 0,
            })
    return pd.DataFrame(rows)


# ── Vocabulary richness ───────────────────────────────────────────────────────

def vocab_richness(rows: list) -> dict:
    """Type-Token Ratio and mean word length for a list of rows."""
    import re
    ttrs, wlens = [], []
    for r in rows:
        toks = re.findall(r"\b[a-z]+\b", r["text"].lower())
        if toks:
            ttrs.append(len(set(toks)) / len(toks))
            wlens.append(np.mean([len(t) for t in toks]))
    return {
        "ttr_mean": round(float(np.mean(ttrs)), 4),
        "ttr_std":  round(float(np.std(ttrs)), 4),
        "wlen_mean": round(float(np.mean(wlens)), 4),
        "wlen_std":  round(float(np.std(wlens)), 4),
    }


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_shap_top_features(df_shap: dict) -> None:
    """
    For each detector: side-by-side horizontal bars of top SHAP features
    at P0 vs P2_simplified (Hard flip cases).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (det_name, df) in zip(axes, df_shap.items()):
        if df.empty:
            ax.set_visible(False)
            continue

        p0   = df[df["stage"] == "P0"].set_index("feature")["mean_shap"]
        p2   = df[df["stage"] == "P2_simplified"].set_index("feature")["mean_shap"]

        # Union of top features from both stages
        all_feats = list(dict.fromkeys(
            list(df[df["stage"] == "P0"]["feature"]) +
            list(df[df["stage"] == "P2_simplified"]["feature"])
        ))[:TOP_K]

        y   = np.arange(len(all_feats))
        h   = 0.35
        v0  = [float(p0.get(f, 0)) for f in all_feats]
        v2  = [float(p2.get(f, 0)) for f in all_feats]

        ax.barh(y + h/2, v0, h, label="P0 (original)",       color="#1f77b4", alpha=0.85)
        ax.barh(y - h/2, v2, h, label="P2 simplified",       color="#d62728", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([repr(f) for f in all_feats], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean |SHAP value| on Hard flip cases")
        ax.set_title(f"{det_name.replace('_', ' ')}\n"
                     f"Top-{TOP_K} features by SHAP importance")
        ax.legend(fontsize=9)
        ax.grid(True, axis="x", alpha=0.25)

    fig.suptitle("Feature attribution shift: Hard-bucket flip cases\n"
                 "(P0 = correct prediction, P2_simplified = wrong prediction)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = paths.FIGURES / "fig20_shap_top_features.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig_feature_survival(survival_data: dict) -> None:
    """
    Bar chart: % of top-K features (by |coef|) that survive into P2 text,
    split by human vs LLM examples among the flip cases.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, (det_name, df) in zip(axes, survival_data.items()):
        if df.empty:
            ax.set_visible(False)
            continue

        groups = {"Human (label=0)": 0, "LLM (label=1)": 1}
        colors = {"Human (label=0)": "#ff7f0e", "LLM (label=1)": "#2ca02c"}
        xs, ys, errs, cols = [], [], [], []

        for gname, glabel in groups.items():
            sub  = df[df["label"] == glabel]
            if sub.empty:
                continue
            rate = sub.groupby("id")["survived"].mean()
            xs.append(gname)
            ys.append(float(rate.mean()))
            errs.append(float(rate.std()) if len(rate) > 1 else 0)
            cols.append(colors[gname])

        x = np.arange(len(xs))
        ax.bar(x, ys, color=cols, alpha=0.85, width=0.5)
        ax.errorbar(x, ys, yerr=errs, fmt="none", color="black", capsize=4)
        for xi, yi in zip(x, ys):
            ax.text(xi, yi + 0.02, f"{yi:.1%}", ha="center", fontsize=10, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(xs)
        ax.set_ylabel(f"Fraction of top-{TOP_K} features surviving into P2")
        ax.set_ylim(0, 1.15)
        ax.set_title(det_name.replace("_", " "))
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.4)
        ax.grid(True, axis="y", alpha=0.2)

    fig.suptitle(f"Feature survival rate: P0 → P2_simplified\n"
                 f"(top-{TOP_K} features by |LR coefficient|, Hard flip cases)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = paths.FIGURES / "fig21_feature_survival.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig_vocab_drift(vocab_df: pd.DataFrame) -> None:
    """TTR and mean word length across P0/P1/P2 for Hard bucket (human vs LLM)."""
    if vocab_df.empty:
        print("Skipping fig22: no vocab drift data.")
        return

    stages  = ["P0", "P1_simplified", "P2_simplified"]
    x       = np.arange(len(stages))
    metrics = [("ttr_mean", "Type-Token Ratio (TTR)"),
               ("wlen_mean", "Mean Word Length")]
    groups  = [("Human", "#ff7f0e"), ("LLM", "#2ca02c")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for group, color in groups:
            sub = vocab_df[vocab_df["group"] == group]
            vals = [float(sub[sub["stage"] == s][metric].values[0])
                    if len(sub[sub["stage"] == s]) else np.nan for s in stages]
            ax.plot(stages, vals, marker="o", color=color,
                    label=group, linewidth=2)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Vocabulary richness in Hard bucket: Human vs LLM across paraphrase stages\n"
                 "(simplified paraphrasing homogenises TTR — erases detector cues)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = paths.FIGURES / "fig22_vocab_drift.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = paths.RESULTS / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading data...")
    train_rows, p0_test, p1_sim, p2_sim, p2_std = load_data()
    X_tr, y_tr = to_xy(train_rows)

    print(f"  P0_test: {len(p0_test)} | P1_sim: {len(p1_sim)} | "
          f"P2_sim: {len(p2_sim)} | train: {len(train_rows)}")

    print("[2/6] Identifying Hard bucket (readability_fk)...")
    hard_ids = get_hard_ids(p0_test)
    print(f"  Hard bucket: {len(hard_ids)} examples")

    id_to_p1 = {r["id"]: r for r in p1_sim}
    id_to_p2 = {r["id"]: r for r in p2_sim}

    hard_p0   = [r for r in p0_test if r["id"] in hard_ids]
    hard_p1   = [id_to_p1[r["id"]] for r in hard_p0 if r["id"] in id_to_p1]
    hard_p2   = [id_to_p2[r["id"]] for r in hard_p0 if r["id"] in id_to_p2]

    print(f"  Hard P0/P1/P2 aligned: {len(hard_p0)}/{len(hard_p1)}/{len(hard_p2)}")

    print("[3/6] Training detectors...")
    detectors = {
        "char_tfidf_lr": build_char_tfidf_lr(X_tr, y_tr),
        "word_tfidf_lr": build_word_tfidf_lr(X_tr, y_tr),
    }

    print("[4/6] Finding Hard-bucket flip cases (P0 correct → P2 wrong)...")
    all_flips_rows = []
    flip_stats = {}
    for det_name, det in detectors.items():
        flips = find_flips(p0_test, p2_sim, det, hard_ids)
        flip_stats[det_name] = len(flips)
        print(f"  {det_name}: {len(flips)} flip cases "
              f"(human flips: {sum(f['label']==0 for f in flips)}, "
              f"LLM flips: {sum(f['label']==1 for f in flips)})")
        for f in flips:
            f["detector"] = det_name
        all_flips_rows.extend(flips)

    df_flips = pd.DataFrame(all_flips_rows)
    if not df_flips.empty:
        df_flips.drop(columns=["text_p0", "text_p2"]).to_csv(
            out_dir / "mechanistic_flip_cases.csv", index=False
        )
        print(f"  Saved: mechanistic_flip_cases.csv ({len(all_flips_rows)} rows)")

    print("[5/6] SHAP attribution + feature survival (Hard flip cases)...")
    shap_dfs    = {}
    survival_dfs = {}
    drift_dfs    = {}

    bg_texts    = [r["text"] for r in train_rows]   # background for SHAP

    for det_name, det in detectors.items():
        det_flips = [f for f in all_flips_rows if f["detector"] == det_name]
        print(f"\n  [{det_name}] {len(det_flips)} flip cases")

        # ── SHAP on Hard flip texts ──────────────────────────────────────────
        if det_flips:
            flip_p0_texts = [f["text_p0"] for f in det_flips]
            flip_p2_texts = [f["text_p2"] for f in det_flips]
            try:
                df_shap = shap_top_features(det, flip_p0_texts, flip_p2_texts,
                                            bg_texts, top_k=TOP_K)
                shap_dfs[det_name] = df_shap
                print(f"    SHAP computed ({len(df_shap)} rows)")
            except Exception as e:
                print(f"    SHAP failed: {e}")
                shap_dfs[det_name] = pd.DataFrame()

            # ── Feature survival ─────────────────────────────────────────────
            df_surv = feature_survival(det, det_flips, top_k=TOP_K)
            survival_dfs[det_name] = df_surv
            if not df_surv.empty:
                rate = df_surv.groupby("label")["survived"].mean()
                for lbl, r in rate.items():
                    print(f"    Feature survival (label={lbl}): {r:.1%}")
        else:
            shap_dfs[det_name]    = pd.DataFrame()
            survival_dfs[det_name] = pd.DataFrame()

        # ── Feature drift on all Hard rows ───────────────────────────────────
        df_drift = feature_drift_table(det, hard_p0, hard_p1, hard_p2, top_k=TOP_K)
        drift_dfs[det_name] = df_drift

    # Save SHAP summary
    if any(not d.empty for d in shap_dfs.values()):
        shap_all = pd.concat(
            [df.assign(detector=dn) for dn, df in shap_dfs.items() if not df.empty],
            ignore_index=True,
        )
        shap_all.to_csv(out_dir / "mechanistic_shap_summary.csv", index=False)
        print(f"\n  Saved: mechanistic_shap_summary.csv")

    # Save feature drift
    if any(not d.empty for d in drift_dfs.values()):
        drift_all = pd.concat(
            [df.assign(detector=dn) for dn, df in drift_dfs.items() if not df.empty],
            ignore_index=True,
        )
        drift_all.to_csv(out_dir / "mechanistic_feature_drift.csv", index=False)
        print(f"  Saved: mechanistic_feature_drift.csv")

    print("[6/6] Vocabulary richness analysis + figures...")

    # Vocab richness across stages
    vocab_rows = []
    for stage, rows in [("P0", hard_p0), ("P1_simplified", hard_p1), ("P2_simplified", hard_p2)]:
        for group, label in [("Human", "human"), ("LLM", "llm")]:
            sub = [r for r in rows if r["label"] == label]
            if not sub:
                continue
            v = vocab_richness(sub)
            vocab_rows.append({"stage": stage, "group": group, "n": len(sub), **v})
    vocab_df = pd.DataFrame(vocab_rows)
    if not vocab_df.empty:
        vocab_df.to_csv(out_dir / "mechanistic_vocab_richness.csv", index=False)
        print(f"  Saved: mechanistic_vocab_richness.csv")
        print(vocab_df.to_string(index=False))

    # Figures
    if any(not d.empty for d in shap_dfs.values()):
        fig_shap_top_features(shap_dfs)
    if any(not d.empty for d in survival_dfs.values()):
        fig_feature_survival(survival_dfs)
    fig_vocab_drift(vocab_df)

    # ── Summary printout ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("P8 MECHANISTIC FINDINGS")
    print("=" * 65)
    print(f"\nHard bucket size: {len(hard_ids)} examples")
    print("\nFlip counts (P0-correct → P2-sim-wrong):")
    for det_name, n in flip_stats.items():
        pct = n / len(hard_ids) * 100
        print(f"  {det_name:<22} {n:>3} flips  ({pct:.1f}% of Hard bucket)")

    print("\nVocabulary richness (Hard bucket, P0 vs P2_simplified):")
    if not vocab_df.empty:
        for _, row in vocab_df.iterrows():
            print(f"  {row['group']:<6} {row['stage']:<20} "
                  f"TTR={row['ttr_mean']:.3f}  wlen={row['wlen_mean']:.2f}")

    print("\nTop SHAP features — char_tfidf_lr (P0, Hard flips):")
    if "char_tfidf_lr" in shap_dfs and not shap_dfs["char_tfidf_lr"].empty:
        top5_p0 = (shap_dfs["char_tfidf_lr"][shap_dfs["char_tfidf_lr"]["stage"] == "P0"]
                   .sort_values("rank").head(5))
        for _, r in top5_p0.iterrows():
            print(f"  #{int(r['rank']):>2}  {repr(r['feature']):<20}  "
                  f"mean_shap={r['mean_shap']:.4f}")

    print("\nTop SHAP features — char_tfidf_lr (P2_simplified, Hard flips):")
    if "char_tfidf_lr" in shap_dfs and not shap_dfs["char_tfidf_lr"].empty:
        top5_p2 = (shap_dfs["char_tfidf_lr"][shap_dfs["char_tfidf_lr"]["stage"] == "P2_simplified"]
                   .sort_values("rank").head(5))
        for _, r in top5_p2.iterrows():
            print(f"  #{int(r['rank']):>2}  {repr(r['feature']):<20}  "
                  f"mean_shap={r['mean_shap']:.4f}")

    print("\nDone. P8 complete.")


if __name__ == "__main__":
    main()
