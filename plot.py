"""Charts for the README: results/eval_summary.csv + training curve -> docs/results.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

s = pd.read_csv("results/eval_summary.csv")
h = json.load(open("results/training_curve.json"))
sc = ["base", "fewer_trucks", "more_trucks", "volatile_cranes", "breakdown", "six_cranes"]
pols = [("dedicated", "#bdbdbd"), ("fewest_assigned", "#9dc3e6"), ("most_starved", "#2e75b6"), ("ppo", "#c00000")]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [2.2, 1]})
x = np.arange(len(sc))
for i, (p, col) in enumerate(pols):
    v = [s[(s.scenario == c) & (s.policy == p)].crane_moves_per_hour.iloc[0] for c in sc]
    ax[0].bar(x + (i - 1.5) * 0.2, v, 0.2, color=col, label=p.replace("_", " ") if p != "ppo" else "PPO (ours)")
ax[0].set_xticks(x, [c.replace("_", "\n") + ("\n(unseen)" if c in ("volatile_cranes", "breakdown", "six_cranes") else "") for c in sc])
ax[0].set_ylim(18, 29)
ax[0].set_ylabel("crane moves per hour (higher is better)")
ax[0].set_title("Held-out evaluation, 50 seeds per scenario")
ax[0].legend(ncol=4, fontsize=8, loc="upper left")
ax[1].plot([r["step"] / 1000 for r in h], [r["mean_return_last20"] for r in h], color="#c00000")
ax[1].set(title="PPO training", xlabel="dispatch decisions (thousands)", ylabel="episode return = -crane idle minutes")
plt.tight_layout()
plt.savefig("docs/results.png", dpi=130)
