"""Multi-agent runs: each truck is an agent with a SHARED policy (parameter sharing, independent PPO style) that
sees only local information. Usage: python train_ma.py <obs_mode: local|comm> <seed>"""
import json
import sys
import torch
from terminal_rl.ppo import PPOConfig, train

if __name__ == "__main__":
    mode, seed = sys.argv[1], int(sys.argv[2])
    torch.set_num_threads(1)
    net, hist = train(PPOConfig(total_steps=450_000, seed=seed, obs_mode=mode), log=lambda s: None)
    torch.save(net.state_dict(), f"results/ppo_{mode}_seed{seed}.pt")
    json.dump(hist, open(f"results/ppo_{mode}_seed{seed}_curve.json", "w"))
    print("done", mode, seed, hist[-1])
