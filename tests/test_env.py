"""PettingZoo ``PuertoRicoEnv`` API and action-mask tests."""

from __future__ import annotations

import numpy as np
import pytest

from pettingzoo.test import api_test

from puerto_rico.env import PuertoRicoEnv, _ACTION_REGISTRY, _ACTION_TO_INDEX


@pytest.mark.parametrize("num_players,num_cycles", [(3, 30), (4, 25), (5, 20)])
def test_pettingzoo_api_test(num_players: int, num_cycles: int) -> None:
    """``pettingzoo.test.api_test`` (keep ``num_cycles`` modest — each cycle is one agent turn)."""
    env = PuertoRicoEnv(num_players=num_players)
    api_test(env, num_cycles=num_cycles, verbose_progress=False)


def test_observation_contains_action_mask() -> None:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    for agent in env.possible_agents:
        obs = env.observe(agent)
        assert "action_mask" in obs
        assert isinstance(obs["action_mask"], np.ndarray)
        assert obs["action_mask"].shape == (len(_ACTION_REGISTRY),)


def test_action_mask_matches_legal_engine_actions() -> None:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=42)
    agent = env.agent_selection
    pid = int(agent.split("_", 1)[1])
    obs = env.observe(agent)
    mask = obs["action_mask"]
    legal = env._engine.legal_actions(pid)  # noqa: SLF001 — test env/engine coupling
    assert sum(mask) == len(legal)
    for a in legal:
        idx = _ACTION_TO_INDEX[a]
        assert mask[idx] == 1
    for i, v in enumerate(mask):
        if v:
            assert _ACTION_REGISTRY[i] in legal


def test_non_actor_zero_mask() -> None:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=1)
    acting = env.agent_selection
    for agent in env.possible_agents:
        obs = env.observe(agent)
        if agent == acting:
            assert obs["action_mask"].sum() >= 1
        else:
            assert obs["action_mask"].sum() == 0


def test_step_rejects_masked_out_action() -> None:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    agent = env.agent_selection
    obs = env.observe(agent)
    mask = obs["action_mask"]
    zeros = np.flatnonzero(mask == 0)
    assert zeros.size > 0
    illegal_idx = int(zeros[0])
    with pytest.raises(ValueError, match="illegal"):
        env.step(illegal_idx)
