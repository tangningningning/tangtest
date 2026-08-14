#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import random
from pathlib import Path


def find_root():
    p = Path(__file__).resolve()
    if p.parent.name == "data":
        return p.parent.parent
    return Path.cwd()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = find_root()
    src = root / "data" / "sumo" / "processed" / "mobility_episode_400.csv"
    out = root / "data" / "sumo" / "processed" / "vehicle_role_pool.csv"

    if not src.is_file():
        raise FileNotFoundError(src)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} already exists; use --overwrite")

    ids = set()
    with src.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ids.add(row["vehicle_id"])

    def key(x):
        try:
            return (0, int(x))
        except ValueError:
            return (1, x)

    ids = sorted(ids, key=key)

    if len(ids) != 60:
        raise RuntimeError(f"Expected 60 vehicles, got {len(ids)}")

    rng = random.Random(args.seed)
    rng.shuffle(ids)

    half = len(ids) // 2
    tv_ids = ids[:half]
    sv_ids = ids[half:]

    rows = []
    for rank, vid in enumerate(tv_ids, start=1):
        rows.append({
            "vehicle_id": vid,
            "role_pool": "TV",
            "role_rank": rank,
            "role_seed": args.seed,
        })

    for rank, vid in enumerate(sv_ids, start=1):
        rows.append({
            "vehicle_id": vid,
            "role_pool": "SV",
            "role_rank": rank,
            "role_seed": args.seed,
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["vehicle_id", "role_pool", "role_rank", "role_seed"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Role pool created")
    print("input  =", src)
    print("output =", out)
    print("seed   =", args.seed)
    print("TV candidates =", len(tv_ids))
    print("SV candidates =", len(sv_ids))
    print("TV first 10 =", ", ".join(tv_ids[:10]))
    print("SV first 10 =", ", ".join(sv_ids[:10]))
    print()
    print("Use TV(k) = first k TV-ranked vehicles")
    print("Use SV(k) = first k SV-ranked vehicles")


if __name__ == "__main__":
    main()
