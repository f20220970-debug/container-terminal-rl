"""Held-out evaluation: PPO vs dispatching rules on seeds never used in training, across 6 scenarios.
Usage: python evaluate.py   -> results/eval_episodes.csv, results/eval_summary.csv, results/summary.json
"""
import json
import numpy as np
import pandas as pd
import torch

from terminal_rl.env import make
from terminal_rl.heuristics import RULES
from terminal_rl.ppo import CraneScorer, policy_fn

SCENARIOS = ["base", "fewer_trucks", "more_trucks", "volatile_cranes", "breakdown", "six_cranes"]
TRAINED_ON = {"base", "fewer_trucks", "more_trucks"}     # fleet 8-14 and cycle cv 0.25-0.5 were in training
N_SEEDS, SEED0 = 50, 900_000                            # training seeds are random draws < 1e9 from rng(0)


def run(policy, scenario, seed):
    env = make(scenario)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    done, info = False, {}
    while not done:
        _, _, done, _, info = env.step(policy(env, rng))
    return info


def boot_ci(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    net = CraneScorer(7, 3)
    net.load_state_dict(torch.load("results/ppo_dispatcher.pt"))
    policies = {**RULES, "ppo": policy_fn(net)}
    rows = []
    for sc in SCENARIOS:
        for s in range(N_SEEDS):
            for name, pol in policies.items():
                k = run(pol, sc, SEED0 + s)
                rows.append({"scenario": sc, "seed": SEED0 + s, "policy": name, **k})
    ep = pd.DataFrame(rows)
    ep.to_csv("results/eval_episodes.csv", index=False)

    summ, headline = [], {}
    for sc, g in ep.groupby("scenario", sort=False):
        piv = g.pivot(index="seed", columns="policy", values="crane_moves_per_hour")
        rules_mean = piv[list(RULES)].mean()
        best_rule = rules_mean.idxmax()
        diff = (piv["ppo"] - piv[best_rule]).values
        lo, hi = boot_ci(diff)
        for pol in piv.columns:
            summ.append({"scenario": sc, "in_training_distribution": sc in TRAINED_ON, "policy": pol,
                         "crane_moves_per_hour": round(piv[pol].mean(), 2),
                         "crane_idle_pct": round(g[g.policy == pol].crane_idle_pct.mean(), 1),
                         "makespan_h": round(g[g.policy == pol].makespan_h.mean(), 2)})
        headline[sc] = {"best_rule": best_rule, "best_rule_mph": round(rules_mean.max(), 2),
                        "ppo_mph": round(piv["ppo"].mean(), 2), "ppo_minus_best_rule": round(diff.mean(), 2),
                        "ci95": [round(lo, 2), round(hi, 2)], "ppo_wins_pct": round(100 * (diff > 0).mean(), 0),
                        "ppo_vs_nearest_pct": round(100 * (piv["ppo"].mean() / piv["nearest"].mean() - 1), 1),
                        "in_training_distribution": sc in TRAINED_ON}
    pd.DataFrame(summ).to_csv("results/eval_summary.csv", index=False)
    json.dump(headline, open("results/summary.json", "w"), indent=2)
    print(pd.DataFrame(headline).T.to_string())


if __name__ == "__main__":
    main()
