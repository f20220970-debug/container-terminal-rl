"""Seed robustness + reward ablation. Evaluates 3 training seeds of each reward variant on the same held-out
seeds as evaluate.py and compares them with the best rule per scenario.
Usage: python evaluate_seeds.py  -> results/seed_summary.csv, results/seed_summary.json"""
import json
import numpy as np
import pandas as pd
import torch

from evaluate import N_SEEDS, SCENARIOS, SEED0, run
from terminal_rl.ppo import CraneScorer, policy_fn

N = 30   # held-out episodes per scenario per checkpoint (same seeds for every policy)


def load(tag):
    net = CraneScorer(7, 3)
    net.load_state_dict(torch.load(f"results/{tag}.pt"))
    return policy_fn(net)


def main():
    base = pd.read_csv("results/eval_episodes.csv")
    seeds = [SEED0 + s for s in range(N)]
    rows = []
    for variant in ["idle", "mk"]:
        for s in [0, 1, 2]:
            pol = load(f"ppo_{variant}_seed{s}")
            for sc in SCENARIOS:
                for sd in seeds:
                    k = run(pol, sc, sd)
                    rows.append({"variant": variant, "train_seed": s, "scenario": sc, "seed": sd, **k})
    ep = pd.DataFrame(rows)
    ep.to_csv("results/seed_episodes.csv", index=False)
    out = []
    for sc in SCENARIOS:
        b = base[(base.scenario == sc) & (base.seed.isin(seeds)) & (base.policy.isin(
            ["dedicated", "fewest_assigned", "most_starved"]))]
        best = b.groupby("policy").crane_moves_per_hour.mean()
        best_rule, best_val = best.idxmax(), best.max()
        best_mk = b[b.policy == best_rule].makespan_h.mean()
        for variant in ["idle", "mk"]:
            per_seed = ep[(ep.scenario == sc) & (ep.variant == variant)].groupby("train_seed")
            mph = per_seed.crane_moves_per_hour.mean()
            out.append({"scenario": sc, "reward": "idle time" if variant == "idle" else "idle + turnaround",
                        "best_rule": best_rule, "best_rule_mph": round(best_val, 2),
                        "ppo_mph_mean": round(mph.mean(), 2), "ppo_mph_min_seed": round(mph.min(), 2),
                        "ppo_mph_max_seed": round(mph.max(), 2), "seeds_beating_best_rule": int((mph > best_val).sum()),
                        "ppo_makespan_h": round(per_seed.makespan_h.mean().mean(), 3),
                        "best_rule_makespan_h": round(best_mk, 3)})
    df = pd.DataFrame(out)
    df.to_csv("results/seed_summary.csv", index=False)
    json.dump(out, open("results/seed_summary.json", "w"), indent=1)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
