"""Masked-random rollouts: engine and env stay legal until termination."""

from __future__ import annotations

import random

import pytest

from puerto_rico.engine import PuertoRicoEngine
from puerto_rico.env import PuertoRicoEnv
from tests._rollout_helpers import MAX_RANDOM_EPISODE_STEPS, assert_engine_invariants, sample_engine_action


def _masked_random_episode(*, num_players: int, seed: int, max_steps: int = MAX_RANDOM_EPISODE_STEPS) -> None:
    rng = random.Random(seed)
    eng = PuertoRicoEngine()
    eng.reset(num_players, seed=seed)
    steps = 0
    assert_engine_invariants(eng)
    while not eng.is_terminal() and steps < max_steps:
        actor = eng.acting_player()
        assert actor is not None, "non-terminal state must have a legal actor"
        choice = sample_engine_action(eng, actor, rng)
        assert eng.is_legal(actor, choice)
        eng.apply(actor, choice)
        assert_engine_invariants(eng)
        steps += 1
    assert steps > 0
    assert eng.is_terminal(), "episode should end within max_steps"


@pytest.mark.parametrize("num_players,seed", [(3, 0), (4, 1), (5, 2)])
def test_engine_random_play_terminates(num_players: int, seed: int) -> None:
    _masked_random_episode(num_players=num_players, seed=seed + num_players * 1000)


def test_env_masked_random_play() -> None:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=88)
    steps = 0
    assert_engine_invariants(env._engine)  # noqa: SLF001 - smoke test checks env/engine consistency
    while not any(env.terminations.values()) and steps < MAX_RANDOM_EPISODE_STEPS:
        agent = env.agent_selection
        obs = env.observe(agent)
        action = env.action_space(agent).sample(obs["action_mask"])
        decoded = env._decode_action(action)  # noqa: SLF001 - intentional env/engine coupling
        player_id = int(agent.split("_", 1)[1])
        assert env._engine.is_legal(player_id, decoded)  # noqa: SLF001
        env.step(action)
        assert_engine_invariants(env._engine)  # noqa: SLF001 - smoke test checks env/engine consistency
        steps += 1
    assert steps > 0
    assert all(env.terminations.values())
    assert env._engine.is_terminal()  # noqa: SLF001
