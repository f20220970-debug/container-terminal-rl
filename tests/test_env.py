"""Sanity checks on the simulator and the baselines."""
import numpy as np
from terminal_rl.env import make
from terminal_rl.heuristics import RULES


def play(rule, scenario="base", seed=1):
    env = make(scenario)
    env.reset(seed=seed)
    rng, done, total_r, info = np.random.default_rng(seed), False, 0.0, {}
    while not done:
        _, r, done, _, info = env.step(rule(env, rng))
        total_r += r
    return env, total_r, info


def test_every_container_is_delivered():
    for sc in ["base", "breakdown", "six_cranes"]:
        env, _, info = play(RULES["fewest_assigned"], sc)
        assert info["delivered"] == env.total


def test_return_equals_minus_total_crane_idle():
    env, total_r, _ = play(RULES["most_starved"])
    assert abs(total_r + env.idle_total / 60) < 1e-6


def test_same_seed_same_result():
    a = play(RULES["dedicated"], seed=5)[2]
    b = play(RULES["dedicated"], seed=5)[2]
    assert {k: v for k, v in a.items() if k != "action_mask"} == {k: v for k, v in b.items() if k != "action_mask"}


def test_productivity_never_exceeds_physical_limit():
    env, _, info = play(RULES["most_starved"], "more_trucks")
    limit = 3600 / (env.cfg.crane_cycle_mean_s * 0.8 + env.cfg.handover_s)   # generous upper bound
    assert info["crane_moves_per_hour"] < limit


def test_action_mask_blocks_finished_cranes():
    env = make("base")
    env.reset(seed=3)
    env.c_remaining[0] = 0
    env.c_holding[0] = False
    assert not env._valid()[0]
