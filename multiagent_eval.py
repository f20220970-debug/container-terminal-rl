"""Central dispatcher vs decentralised truck agents, at three information levels, on held-out ships.
Usage: python multiagent_eval.py -> results/multiagent_summary.csv"""
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from evaluate import SEED0
from terminal_rl.env import make
from terminal_rl.heuristics import RULES, public_board
from terminal_rl.ppo import CraneScorer, policy_fn

SCEN = ["fewer_trucks", "base", "more_trucks", "six_cranes"]
N = 30


def load(tag):
    net = CraneScorer(7, 3)
    net.load_state_dict(torch.load(f"results/{tag}.pt"))
    return policy_fn(net)


def run(policy, scenario, seed, obs_mode):
    env = make(scenario)
    env.cfg = replace(env.cfg, obs_mode=obs_mode)
    env.reset(seed=seed)
    done, info, rng = False, {}, np.random.default_rng(seed)
    while not done:
        _, _, done, _, info = env.step(policy(env, rng))
    return info


POLICIES = [  # name, information level, policy, obs_mode
    ("rule: public board (greedy)", "local", public_board, "local"),
    ("PPO truck agents, local info", "local", None, "local"),
    ("rule: fewest assigned", "broadcast counts", RULES["fewest_assigned"], "comm"),
    ("PPO truck agents + broadcast", "broadcast counts", None, "comm"),
    ("rule: most starved", "full (central)", RULES["most_starved"], "full"),
    ("PPO central dispatcher", "full (central)", None, "full"),
]


def main():
    learned = {"local": load("ppo_local_seed0"), "comm": load("ppo_comm_seed0"), "full": load("ppo_idle_seed0")}
    rows = []
    for name, info_level, pol, mode in POLICIES:
        p = pol or learned[mode]
        for sc in SCEN:
            for s in range(N):
                k = run(p, sc, SEED0 + s, mode)
                rows.append({"policy": name, "information": info_level, "scenario": sc, "seed": SEED0 + s, **k})
    df = pd.DataFrame(rows)
    df.to_csv("results/multiagent_episodes.csv", index=False)
    summ = df.pivot_table(index=["information", "policy"], columns="scenario", values="crane_moves_per_hour",
                          aggfunc="mean").round(2)[SCEN]
    summ.to_csv("results/multiagent_summary.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
