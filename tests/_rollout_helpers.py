"""Shared helpers for randomized engine/env rollout tests."""

from __future__ import annotations

import random

from puerto_rico.constants import COLONIST_DISCS_TOTAL, TOTAL_DOUBLOON_VALUE
from puerto_rico.engine import MayorSubmitPlacement, PuertoRicoEngine
from puerto_rico.state import MayorPhasePending, Phase, island_tile_max_colonists, total_colonists_on_board

MAX_RANDOM_EPISODE_STEPS = 100_000


def sample_mayor_placement(eng: PuertoRicoEngine, player_id: int, rng: random.Random) -> MayorSubmitPlacement:
    pending = eng.state.pending
    assert isinstance(pending, MayorPhasePending)
    pool = pending.placement_pools[player_id]
    player = eng.state.players[player_id]

    island_caps = [0 if space.tile is None else island_tile_max_colonists(space.tile) for space in player.island_spaces]
    building_caps = [len(pb.colonists) for pb in player.city_buildings]
    building_caps.extend(0 for _ in range(12 - len(building_caps)))
    board_caps = island_caps + building_caps

    if pool >= sum(board_caps):
        board_alloc = board_caps
    else:
        slots: list[int] = []
        for idx, capacity in enumerate(board_caps):
            slots.extend(idx for _ in range(capacity))
        picks = rng.sample(slots, pool)
        board_alloc = [0 for _ in board_caps]
        for idx in picks:
            board_alloc[idx] += 1

    return MayorSubmitPlacement(
        island_targets=tuple(board_alloc[:12]),
        building_targets=tuple(board_alloc[12:]),
        san_juan=pool - sum(board_alloc),
    )


def sample_engine_action(eng: PuertoRicoEngine, player_id: int, rng: random.Random):
    pending = eng.state.pending
    if eng.state.phase is Phase.MAYOR and isinstance(pending, MayorPhasePending) and pending.subphase == "placement":
        return sample_mayor_placement(eng, player_id, rng)
    legal = eng.legal_actions(player_id)
    assert legal
    return rng.choice(legal)


def assert_engine_invariants(eng: PuertoRicoEngine) -> None:
    state = eng.state
    assert state.bank_doubloons + sum(player.doubloons for player in state.players) == TOTAL_DOUBLOON_VALUE
    for ship in state.cargo_ships:
        assert 0 <= ship.barrels <= ship.capacity
        if ship.good is None:
            assert ship.barrels == 0
    assert len(state.trading_house.goods) <= 4

    colonists_modeled = state.colonist_ship + state.colonist_supply + sum(
        total_colonists_on_board(player) for player in state.players
    )
    assert 0 <= colonists_modeled <= COLONIST_DISCS_TOTAL
