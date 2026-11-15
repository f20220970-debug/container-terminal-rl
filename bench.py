"""Throughput benchmark: simulated dispatch decisions per second with 1..N parallel worker processes.
Usage: python bench.py [max_workers]  -> results/benchmark.json"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from terminal_rl.env import make
from terminal_rl.heuristics import RULES

EPISODES_PER_WORKER = 20


def worker(wid):
    steps, rule = 0, RULES["most_starved"]
    for e in range(EPISODES_PER_WORKER):
        env = make("base")
        env.reset(seed=wid * 1000 + e)
        done = False
        while not done:
            _, _, done, _, _ = env.step(rule(env))
            steps += 1
    return steps


def main(max_workers):
    res = []
    for n in range(1, max_workers + 1):
        t = time.perf_counter()
        with mp.Pool(n) as pool:
            steps = sum(pool.map(worker, range(n)))
        dt = time.perf_counter() - t
        res.append({"workers": n, "decisions": steps, "seconds": round(dt, 2), "decisions_per_s": round(steps / dt)})
        print(res[-1], flush=True)
    base = res[0]["decisions_per_s"]
    for r in res:
        r["speedup"] = round(r["decisions_per_s"] / base, 2)
        r["efficiency"] = round(r["speedup"] / r["workers"], 2)
    json.dump({"cpu_count": os.cpu_count(), "runs": res}, open("results/benchmark.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count())
