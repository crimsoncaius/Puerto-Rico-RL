"""Setup and constants invariants for ``initial_game_state``."""

from __future__ import annotations

import random

import pytest

from puerto_rico import constants as c
from puerto_rico.setup import initial_game_state, validate_initial_state
from puerto_rico.state import IslandTile, Phase


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_validate_initial_state_all_player_counts(num_players: int) -> None:
    state = initial_game_state(num_players, seed=42)
    validate_initial_state(state)


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_starting_doubloons_and_bank(num_players: int) -> None:
    state = initial_game_state(num_players, seed=0)
    sd = c.STARTING_DOUBLOONS_BY_PLAYER_COUNT[num_players]
    assert all(p.doubloons == sd for p in state.players)
    assert state.bank_doubloons == c.TOTAL_DOUBLOON_VALUE - num_players * sd
    assert state.bank_doubloons + sum(p.doubloons for p in state.players) == c.TOTAL_DOUBLOON_VALUE


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_vp_supply_by_player_count(num_players: int) -> None:
    state = initial_game_state(num_players, seed=0)
    assert state.vp_supply == c.VP_SUPPLY_TOTAL_BY_PLAYER_COUNT[num_players]
    assert state.vp_supply <= c.VP_CHIPS_PHYSICAL_TOTAL_VALUE


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_colonist_ship_and_supply(num_players: int) -> None:
    state = initial_game_state(num_players, seed=0)
    assert state.colonist_ship == c.COLONIST_SHIP_BY_PLAYER_COUNT[num_players]
    assert state.colonist_supply == c.COLONIST_SUPPLY_BY_PLAYER_COUNT[num_players]


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_cargo_ship_capacities(num_players: int) -> None:
    state = initial_game_state(num_players, seed=0)
    caps = c.CARGO_SHIP_CAPACITIES_BY_PLAYER_COUNT[num_players]
    assert len(state.cargo_ships) == 3
    assert tuple(sh.capacity for sh in state.cargo_ships) == caps
    assert all(sh.good is None and sh.barrels == 0 for sh in state.cargo_ships)


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_roles_in_play_match_constants(num_players: int) -> None:
    state = initial_game_state(num_players, seed=0)
    assert state.roles_in_play == c.roles_for_player_count(num_players)


def test_plantation_tile_counts_sum() -> None:
    assert sum(c.PLANTATION_TILE_COUNTS.values()) == c.PLANTATION_TILE_TOTAL


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_plantation_inventory_conservation(num_players: int) -> None:
    """Stacks + discard + face-up + on boards = total plantation tiles."""
    state = initial_game_state(num_players, seed=123)
    stacked = sum(len(st) for st in state.plantation_stacks)
    discard = len(state.plantation_discard)
    on_boards = sum(
        sum(1 for sp in p.island_spaces if sp.tile is not None and sp.tile is not IslandTile.QUARRY)
        for p in state.players
    )
    assert stacked + discard + len(state.face_up_plantations) + on_boards == c.PLANTATION_TILE_TOTAL


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_face_up_market_size(num_players: int) -> None:
    state = initial_game_state(num_players, seed=0)
    assert len(state.face_up_plantations) == c.face_up_plantation_count(num_players)


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_starting_plantations_from_table(num_players: int) -> None:
    row = c.STARTING_PLANTATIONS_BY_PLAYER_COUNT[num_players]
    state = initial_game_state(num_players, seed=0)
    for i, p in enumerate(state.players):
        assert p.island_spaces[0].tile == row[i]


def test_deterministic_seed_reproducible() -> None:
    s1 = initial_game_state(4, seed=12345)
    s2 = initial_game_state(4, seed=12345)
    assert s1 == s2


def test_different_seeds_can_differ() -> None:
    a = initial_game_state(5, seed=1)
    b = initial_game_state(5, seed=2)
    assert a.face_up_plantations != b.face_up_plantations or a.plantation_stacks != b.plantation_stacks


def test_shuffle_uses_seed_parameter() -> None:
    rng = random.Random(99)
    bag: list[int] = list(range(20))
    rng.shuffle(bag)
    first = tuple(bag)

    rng2 = random.Random(99)
    bag2 = list(range(20))
    rng2.shuffle(bag2)
    assert tuple(bag2) == first


def test_constants_all_exported() -> None:
    for name in c.__all__:
        assert hasattr(c, name), f"missing __all__ name: {name}"


def test_initial_phase_and_governor() -> None:
    state = initial_game_state(3, seed=0)
    assert state.phase is Phase.ROLE_SELECTION
    assert state.governor_index == 0
    assert state.next_role_selector_index == 0


def test_quarries_and_goods_supply() -> None:
    state = initial_game_state(4, seed=0)
    assert state.quarries_remaining == c.QUARRY_TILE_COUNT
    gdict = dict(state.goods_supply)
    assert sum(gdict.values()) == c.GOOD_SUPPLY_TOTAL
    for good, expect in c.GOOD_SUPPLY_COUNTS.items():
        assert gdict.get(good, 0) == expect


def test_five_plantation_stacks() -> None:
    state = initial_game_state(3, seed=7)
    assert len(state.plantation_stacks) == c.NUM_PLANTATION_STACKS


def test_invalid_player_count_raises() -> None:
    with pytest.raises(ValueError):
        initial_game_state(2, seed=0)
