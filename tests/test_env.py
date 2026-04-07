"""PettingZoo ``PuertoRicoEnv`` API and structured action-mask tests."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pettingzoo.test import api_test

from puerto_rico.engine import MayorSubmitPlacement, PickRole
from puerto_rico.env import DECISION_TYPES, PuertoRicoEnv
from puerto_rico.state import (
    Building,
    BuilderPhasePending,
    CaptainPhasePending,
    CraftsmanPhasePending,
    Good,
    IslandSpace,
    IslandTile,
    MayorPhasePending,
    Phase,
    PlacedBuilding,
    Role,
    SettlerPhasePending,
    TraderPhasePending,
    TradingHouseState,
    normalize_goods_counts,
)

_EXPECTED_API_WARNING_SNIPPETS = (
    "Observation space for each agent probably should be",
    "Action space for each agent probably should be",
    "Observation is not a NumPy array",
)


def _acting_agent(env: PuertoRicoEnv) -> str:
    return env.agent_selection


def _first_masked_out_index(mask: np.ndarray) -> int:
    masked_out = np.flatnonzero(mask == 0)
    assert masked_out.size > 0
    return int(masked_out[0])


def _assert_masked_branch_rejected(env: PuertoRicoEnv, branch: str) -> None:
    agent = _acting_agent(env)
    obs = env.observe(agent)
    illegal_index = _first_masked_out_index(np.asarray(obs["action_mask"][branch]))
    with pytest.raises(ValueError, match="illegal"):
        env.step({branch: illegal_index})


def _make_settler_env() -> PuertoRicoEnv:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    env._engine._state = dataclasses.replace(  # noqa: SLF001 - targeted env setup
        env._engine.state,
        phase=Phase.SETTLER,
        round_role_order=((Role.SETTLER, 0),),
        current_role_execution_index=0,
        face_up_plantations=(IslandTile.INDIGO,),
        pending=SettlerPhasePending(settler_role_chooser=0, next_actor_index=0),
    )
    env.agent_selection = "player_0"
    return env


def _make_mayor_privilege_env(*, colonist_supply: int = 0) -> PuertoRicoEnv:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    env._engine._state = dataclasses.replace(  # noqa: SLF001 - targeted env setup
        env._engine.state,
        colonist_supply=colonist_supply,
        phase=Phase.MAYOR,
        round_role_order=((Role.MAYOR, 0),),
        current_role_execution_index=0,
        pending=MayorPhasePending(
            mayor_role_chooser=0,
            subphase="privilege",
            placement_pools=(0, 0, 0),
            placement_next=None,
        ),
    )
    env.agent_selection = "player_0"
    return env


def _make_mayor_placement_env() -> PuertoRicoEnv:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    island = list(env._engine.state.players[0].island_spaces)  # noqa: SLF001 - targeted env setup
    island[0] = IslandSpace(tile=IslandTile.QUARRY, colonists=0)
    player0 = dataclasses.replace(
        env._engine.state.players[0],  # noqa: SLF001
        island_spaces=tuple(island),
        city_buildings=(PlacedBuilding(building=Building.SMALL_MARKET, anchor_slot=0, colonists=(0,)),),
    )
    env._engine._state = dataclasses.replace(  # noqa: SLF001 - targeted env setup
        env._engine.state,
        players=(player0,) + env._engine.state.players[1:],
        phase=Phase.MAYOR,
        round_role_order=((Role.MAYOR, 0),),
        current_role_execution_index=0,
        pending=MayorPhasePending(
            mayor_role_chooser=0,
            subphase="placement",
            placement_pools=(3, 0, 0),
            placement_next=0,
        ),
    )
    env.agent_selection = "player_0"
    return env


def _make_builder_env() -> PuertoRicoEnv:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    player0 = dataclasses.replace(env._engine.state.players[0], doubloons=10)  # noqa: SLF001
    env._engine._state = dataclasses.replace(  # noqa: SLF001 - targeted env setup
        env._engine.state,
        players=(player0,) + env._engine.state.players[1:],
        phase=Phase.BUILDER,
        round_role_order=((Role.BUILDER, 0),),
        current_role_execution_index=0,
        pending=BuilderPhasePending(role_chooser=0, next_actor=0),
    )
    env.agent_selection = "player_0"
    return env


def _make_craftsman_env() -> PuertoRicoEnv:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    island = list(env._engine.state.players[0].island_spaces)  # noqa: SLF001 - targeted env setup
    island[0] = IslandSpace(tile=IslandTile.CORN, colonists=1)
    player0 = dataclasses.replace(env._engine.state.players[0], island_spaces=tuple(island))  # noqa: SLF001
    env._engine._state = dataclasses.replace(  # noqa: SLF001 - targeted env setup
        env._engine.state,
        players=(player0,) + env._engine.state.players[1:],
        phase=Phase.CRAFTSMAN,
        round_role_order=((Role.CRAFTSMAN, 0),),
        current_role_execution_index=0,
        pending=CraftsmanPhasePending(role_chooser=0, next_actor=0),
    )
    env.agent_selection = "player_0"
    return env


def _make_trader_env() -> PuertoRicoEnv:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    player0 = dataclasses.replace(
        env._engine.state.players[0],  # noqa: SLF001
        goods=normalize_goods_counts({Good.CORN: 1}),
    )
    env._engine._state = dataclasses.replace(  # noqa: SLF001 - targeted env setup
        env._engine.state,
        players=(player0,) + env._engine.state.players[1:],
        phase=Phase.TRADER,
        round_role_order=((Role.TRADER, 0),),
        current_role_execution_index=0,
        trading_house=TradingHouseState(goods=()),
        pending=TraderPhasePending(role_chooser=0, next_actor=0),
    )
    env.agent_selection = "player_0"
    return env


def _make_captain_loading_env() -> PuertoRicoEnv:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    player0 = dataclasses.replace(
        env._engine.state.players[0],  # noqa: SLF001
        goods=normalize_goods_counts({Good.CORN: 1}),
    )
    env._engine._state = dataclasses.replace(  # noqa: SLF001 - targeted env setup
        env._engine.state,
        players=(player0,) + env._engine.state.players[1:],
        phase=Phase.CAPTAIN,
        round_role_order=((Role.CAPTAIN, 0),),
        current_role_execution_index=0,
        pending=CaptainPhasePending(
            captain_role_chooser=0,
            active_player_index=0,
            captain_privilege_vp_awarded=False,
            wharf_used=(False, False, False),
            subphase="loading",
            storage_next_actor=None,
            storage_done=(False, False, False),
            ship_full_credit=(None, None, None),
        ),
    )
    env.agent_selection = "player_0"
    return env


def _make_captain_storage_env() -> PuertoRicoEnv:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    player0 = dataclasses.replace(
        env._engine.state.players[0],  # noqa: SLF001
        goods=normalize_goods_counts({Good.CORN: 1, Good.INDIGO: 1}),
    )
    env._engine._state = dataclasses.replace(  # noqa: SLF001 - targeted env setup
        env._engine.state,
        players=(player0,) + env._engine.state.players[1:],
        phase=Phase.CAPTAIN,
        round_role_order=((Role.CAPTAIN, 0),),
        current_role_execution_index=0,
        pending=CaptainPhasePending(
            captain_role_chooser=0,
            active_player_index=0,
            captain_privilege_vp_awarded=False,
            wharf_used=(False, False, False),
            subphase="storage",
            storage_next_actor=0,
            storage_done=(False, False, False),
            ship_full_credit=(None, None, None),
        ),
    )
    env.agent_selection = "player_0"
    return env


def _make_prospector_env() -> PuertoRicoEnv:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    env._engine._state = dataclasses.replace(  # noqa: SLF001 - targeted env setup
        env._engine.state,
        phase=Phase.PROSPECTOR,
        round_role_order=((Role.PROSPECTOR, 0),),
        current_role_execution_index=0,
        pending=None,
    )
    env.agent_selection = "player_0"
    return env


def _make_reward_env(reward_mode: str) -> PuertoRicoEnv:
    env = PuertoRicoEnv(num_players=3, reward_mode=reward_mode)
    env.reset(seed=0)
    player0 = dataclasses.replace(
        env._engine.state.players[0],  # noqa: SLF001
        goods=normalize_goods_counts({Good.CORN: 2}),
    )
    env._engine._state = dataclasses.replace(  # noqa: SLF001 - targeted env setup
        env._engine.state,
        players=(player0,) + env._engine.state.players[1:],
        phase=Phase.CAPTAIN,
        round_role_order=((Role.CAPTAIN, 0),),
        current_role_execution_index=0,
        pending=CaptainPhasePending(
            captain_role_chooser=0,
            active_player_index=0,
            captain_privilege_vp_awarded=False,
            wharf_used=(False, False, False),
            subphase="loading",
            storage_next_actor=None,
            storage_done=(False, False, False),
            ship_full_credit=(None, None, None),
        ),
    )
    env.agent_selection = "player_0"
    return env


@pytest.mark.parametrize("num_players,num_cycles", [(3, 30), (4, 25), (5, 20)])
def test_pettingzoo_api_test(num_players: int, num_cycles: int) -> None:
    env = PuertoRicoEnv(num_players=num_players)
    with pytest.warns(UserWarning) as captured:
        api_test(env, num_cycles=num_cycles, verbose_progress=False)
    messages = {str(warning.message) for warning in captured.list}
    assert len(messages) == len(_EXPECTED_API_WARNING_SNIPPETS)
    for snippet in _EXPECTED_API_WARNING_SNIPPETS:
        assert any(snippet in message for message in messages)


def test_observation_contains_structured_action_mask() -> None:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    for agent in env.possible_agents:
        obs = env.observe(agent)
        assert "observation" in obs
        assert "action_mask" in obs
        assert isinstance(obs["observation"], np.ndarray)
        assert isinstance(obs["action_mask"], dict)
        assert tuple(obs["action_mask"]) == DECISION_TYPES


def test_initial_role_mask_matches_legal_role_choices() -> None:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=42)
    agent = env.agent_selection
    pid = int(agent.split("_", 1)[1])
    obs = env.observe(agent)
    role_mask = obs["action_mask"]["role"]
    legal = env._engine.legal_actions(pid)  # noqa: SLF001 - test env/engine coupling
    expected_roles = {action.role for action in legal if isinstance(action, PickRole)}
    enabled_roles = {
        env._decode_action({"role": role_index}).role  # noqa: SLF001 - intentional env/engine coupling
        for role_index, enabled in enumerate(role_mask.tolist())
        if int(enabled) == 1
    }
    assert enabled_roles == expected_roles


def test_non_actor_masks_are_zeroed() -> None:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=1)
    acting = env.agent_selection
    for agent in env.possible_agents:
        mask = env.observe(agent)["action_mask"]
        totals = [
            int(mask["role"].sum()),
            int(mask["settler"].sum()),
            int(mask["mayor_privilege"].sum()),
            int(mask["builder"].sum()),
            int(mask["craftsman"].sum()),
            int(mask["trader"].sum()),
            int(mask["captain_loading"].sum()),
            int(mask["captain_storage"].sum()),
            int(mask["prospector"].sum()),
            int(mask["mayor_placement"]["total_pool"][0]),
        ]
        if agent == acting:
            assert sum(totals) >= 1
        else:
            assert sum(totals) == 0


def test_mayor_placement_mask_samples_legal_final_allocation() -> None:
    env = _make_mayor_placement_env()

    obs = env.observe("player_0")
    sample = env.action_space("player_0").sample(obs["action_mask"])
    decoded = env._decode_action(sample)  # noqa: SLF001 - intentional env/engine coupling
    assert isinstance(decoded, MayorSubmitPlacement)
    assert env._engine.is_legal(0, decoded)  # noqa: SLF001


def test_step_rejects_masked_out_role_action() -> None:
    env = PuertoRicoEnv(num_players=3)
    env.reset(seed=0)
    illegal_role = _first_masked_out_index(env.observe(_acting_agent(env))["action_mask"]["role"])
    with pytest.raises(ValueError, match="illegal"):
        env.step({"role": illegal_role})


@pytest.mark.parametrize(
    ("make_env", "branch"),
    [
        (_make_settler_env, "settler"),
        (_make_mayor_privilege_env, "mayor_privilege"),
        (_make_builder_env, "builder"),
        (_make_craftsman_env, "craftsman"),
        (_make_trader_env, "trader"),
        (_make_captain_loading_env, "captain_loading"),
        (_make_captain_storage_env, "captain_storage"),
    ],
)
def test_step_rejects_masked_out_branch_action(make_env, branch: str) -> None:
    env = make_env()
    _assert_masked_branch_rejected(env, branch)


def test_step_rejects_illegal_mayor_placement_allocation() -> None:
    env = _make_mayor_placement_env()
    obs = env.observe("player_0")
    sample = env.action_space("player_0").sample(obs["action_mask"])
    sample["mayor_placement"]["san_juan"] = int(sample["mayor_placement"]["san_juan"]) + 1

    with pytest.raises(ValueError, match="illegal"):
        env.step(sample)


def test_step_rejects_out_of_range_prospector_choice() -> None:
    env = _make_prospector_env()

    with pytest.raises(ValueError, match="prospector out of range"):
        env.step({"prospector": 1})


@pytest.mark.parametrize("reward_mode,expected_reward", [("vp_delta", 3.0), ("zero", 0.0)])
def test_step_rewards_follow_reward_mode(reward_mode: str, expected_reward: float) -> None:
    env = _make_reward_env(reward_mode)
    agent = _acting_agent(env)
    player_id = int(agent.split("_", 1)[1])
    obs = env.observe(agent)
    action = env.action_space(agent).sample(obs["action_mask"])
    vp_before = env._engine.state.players[player_id].vp_from_chips + env._engine.state.players[player_id].vp_on_paper  # noqa: SLF001

    env.step(action)

    vp_after = env._engine.state.players[player_id].vp_from_chips + env._engine.state.players[player_id].vp_on_paper  # noqa: SLF001
    assert vp_after - vp_before == 3
    assert env.rewards[agent] == expected_reward
    for other_agent in env.possible_agents:
        if other_agent != agent:
            assert env.rewards[other_agent] == 0.0
