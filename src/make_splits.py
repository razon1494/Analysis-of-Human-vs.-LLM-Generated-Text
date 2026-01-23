import os
import json
import random


def load_ids(path: str):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
    return ids


def main():
    P0_PATH = os.path.join("data", "p0", "p0.jsonl")
    OUT_DIR = os.path.join("data", "splits")
    os.makedirs(OUT_DIR, exist_ok=True)

    SEED = 42
    TRAIN_FRAC = 0.8
    VAL_FRAC = 0.1
    TEST_FRAC = 0.1

    ids = load_ids(P0_PATH)
    rng = random.Random(SEED)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    n_test = n - n_train - n_val

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    assert len(train_ids) + len(val_ids) + len(test_ids) == n

    with open(os.path.join(OUT_DIR, "train_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(train_ids) + "\n")

    with open(os.path.join(OUT_DIR, "val_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(val_ids) + "\n")

    with open(os.path.join(OUT_DIR, "test_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(test_ids) + "\n")

    print(f"Saved splits to: {OUT_DIR}")
    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")


if __name__ == "__main__":
    main()
