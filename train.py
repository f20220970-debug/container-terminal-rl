"""Train the PPO dispatcher. Usage: python train.py  (about 6 minutes on 2 CPU cores)"""
import json
import torch
from terminal_rl.ppo import PPOConfig, train

if __name__ == "__main__":
    torch.set_num_threads(2)
    cfg = PPOConfig(total_steps=450_000)
    net, history = train(cfg)
    torch.save(net.state_dict(), "results/ppo_dispatcher.pt")
    json.dump(history, open("results/training_curve.json", "w"), indent=1)
