"""Masked-random rollouts: engine stays legal until termination."""

from __future__ import annotations

import random

import numpy as np
import pytest

from puerto_rico.engine import PuertoRicoEngine
from puerto_rico.env import PuertoRicoEnv, _ACTION_TO_INDEX


def _masked_random_episode(*, num_players: int, seed: int, max_steps: int = 500_000) -> None:
    rng = random.Random(seed)
    eng = PuertoRicoEngine()
    eng.reset(num_players, seed=seed)
    steps = 0
    while not eng.is_terminal() and steps < max_steps:
        actor: int | None = None
        for i in range(num_players):
            if eng.legal_actions(i):
                actor = i
                break
        assert actor is not None, "non-terminal state must have a legal actor"
        legal = eng.legal_actions(actor)
        choice = rng.choice(legal)
        assert eng.is_legal(actor, choice)
        eng.apply(actor, choice)
        steps += 1
    assert eng.is_terminal(), "episode should end within max_steps"


@pytest.mark.parametrize("num_players", [3, 4, 5])
@pytest.mark.parametrize("seed", [0, 1, 2, 42, 99])
def test_engine_random_play_terminates(num_players: int, seed: int) -> None:
    _masked_random_episode(num_players=num_players, seed=seed + num_players * 1000)


def test_env_masked_random_play() -> None:
    """Env steps using only actions allowed by ``action_mask``."""
    rng = random.Random(31415)
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=88)
    steps = 0
    while not any(env.terminations.values()) and steps < 500_000:
        agent = env.agent_selection
        obs = env.observe(agent)
        mask = obs["action_mask"]
        idx = int(rng.choice(np.flatnonzero(mask)))
        assert mask[idx] == 1
        env.step(idx)
        steps += 1
    assert all(env.terminations.values())
    assert env._engine.is_terminal()  # noqa: SLF001


def test_registry_covers_all_legal_actions_in_smoke() -> None:
    """During a short rollout, every legal engine action maps to the discrete registry."""
    rng = random.Random(0)
    eng = PuertoRicoEngine()
    eng.reset(4, seed=123)
    for _ in range(5_000):
        if eng.is_terminal():
            break
        actor = next(i for i in range(eng.state.num_players) if eng.legal_actions(i))
        for a in eng.legal_actions(actor):
            assert a in _ACTION_TO_INDEX, f"missing from registry: {a!r}"
        eng.apply(actor, rng.choice(eng.legal_actions(actor)))
