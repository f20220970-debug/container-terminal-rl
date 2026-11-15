"""Robustness runs: PPO with 3 training seeds, for the idle-time reward and for idle time + a turnaround penalty.
Usage: python train_seeds.py <seed> <makespan_coef>   (each run ~11 min on 2 cores)"""
import json
import sys
import torch
from terminal_rl.ppo import PPOConfig, train

if __name__ == "__main__":
    seed, coef = int(sys.argv[1]), float(sys.argv[2])
    torch.set_num_threads(1)
    net, hist = train(PPOConfig(total_steps=450_000, seed=seed, makespan_coef=coef), log=lambda s: None)
    tag = f"ppo_{'mk' if coef > 0 else 'idle'}_seed{seed}"
    torch.save(net.state_dict(), f"results/{tag}.pt")
    json.dump(hist, open(f"results/{tag}_curve.json", "w"))
    print("done", tag, hist[-1])
