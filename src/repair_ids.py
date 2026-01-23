import json
import os
from collections import Counter


def main():
    IN_PATH = os.path.join("data", "p0", "p0.jsonl")
    OUT_PATH = os.path.join("data", "p0", "p0_fixed.jsonl")

    rows = []
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    ids = [r["id"] for r in rows]
    c = Counter(ids)
    dup_ids = {k for k, v in c.items() if v > 1}

    if not dup_ids:
        print("No duplicate IDs found. Nothing to fix.")
        return

    used = set()
    fixed = []
    changed = 0

    for r in rows:
        _id = r["id"]
        if _id not in used:
            used.add(_id)
            fixed.append(r)
            continue

        # Duplicate found -> assign a new unique ID while preserving label
        changed += 1
        base = _id

        # Safer naming: keep original as prefix + add suffix counter
        k = 2
        new_id = f"{base}__dup{k}"
        while new_id in used:
            k += 1
            new_id = f"{base}__dup{k}"

        r["id"] = new_id
        used.add(new_id)
        fixed.append(r)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in fixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved: {OUT_PATH}")
    print(f"Total rows: {len(fixed)}")
    print(f"IDs unique: {len({r['id'] for r in fixed})}")
    print(f"Duplicates repaired: {changed}")


if __name__ == "__main__":
    main()
