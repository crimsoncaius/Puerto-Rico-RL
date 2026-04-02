# Rule coverage checklist (Setup — mapping to `initial_game_state` / validation):
# ---------------------------------------------------------------------------
# [x] General: board central — implicit; unused colonists returned to box — not modeled.
# [x] Supply: quarries all face-up → quarries_remaining == QUARRY_TILE_COUNT.
# [x] Plantations: shuffle 50 tiles into 5 stacks → plantation_stacks (order = bottom→top).
# [x] Reveal (players + 1) face-up → face_up_plantations (round-robin from stack tops).
# [x] Goods: sorted into five piles → goods_supply from GOOD_SUPPLY_COUNTS.
# [x] Trading house nearby → empty TradingHouseState.
# [x] Colonist ship 3/4/5 → colonist_ship.
# [x] Colonist supply 55/75/95 → colonist_supply.
# [x] Roles 6/7/8 → roles_in_play via roles_for_player_count.
# [x] Cargo ships capacities by player count → cargo_ships.
# [x] VP pool 75/100/122 → vp_supply (total VP value remaining).
# [x] Starting doubloons 2/3/4 → players[].doubloons; bank remainder → bank_doubloons.
# [x] Governor + starting plantations by seating → governor_index 0; island space 0 tiles.
# ---------------------------------------------------------------------------

from __future__ import annotations

import dataclasses
import random

from .constants import (
    BUILDING_SPECS,
    COLONIST_DISCS_TOTAL,
    COLONIST_SHIP_BY_PLAYER_COUNT,
    COLONIST_SUPPLY_BY_PLAYER_COUNT,
    CARGO_SHIP_CAPACITIES_BY_PLAYER_COUNT,
    GOOD_SUPPLY_COUNTS,
    GOOD_SUPPLY_TOTAL,
    NUM_PLANTATION_STACKS,
    PLANTATION_TILE_COUNTS,
    PLANTATION_TILE_TOTAL,
    QUARRY_TILE_COUNT,
    STARTING_DOUBLOONS_BY_PLAYER_COUNT,
    STARTING_PLANTATIONS_BY_PLAYER_COUNT,
    SUPPORTED_PLAYER_COUNTS,
    TOTAL_DOUBLOON_VALUE,
    VP_CHIPS_PHYSICAL_TOTAL_VALUE,
    VP_SUPPLY_TOTAL_BY_PLAYER_COUNT,
    face_up_plantation_count,
    roles_for_player_count,
)
from .state import (
    BuildingSupplyCounts,
    CargoShipState,
    GameState,
    Good,
    IslandSpace,
    IslandTile,
    Phase,
    PlayerState,
    TradingHouseState,
    normalize_building_supply,
    normalize_goods_counts,
    total_colonists_on_board,
)


def _empty_goods() -> "tuple[tuple[Good, int], ...]":
    return normalize_goods_counts({})


def _initial_building_supply() -> BuildingSupplyCounts:
    return normalize_building_supply({spec.building: spec.copies_in_supply for spec in BUILDING_SPECS})


def _multiset_remove(tiles: list[IslandTile], remove: IslandTile) -> None:
    try:
        tiles.remove(remove)
    except ValueError as e:
        raise AssertionError(f"Tile {remove} not available in plantation multiset") from e


def _build_plantation_multiset() -> list[IslandTile]:
    out: list[IslandTile] = []
    for tile, n in PLANTATION_TILE_COUNTS.items():
        out.extend([tile] * n)
    assert len(out) == PLANTATION_TILE_TOTAL
    return out


def _split_into_stacks(tile_list: list[IslandTile], num_stacks: int) -> tuple[tuple[IslandTile, ...], ...]:
    """Split into `num_stacks` stacks of nearly equal size; each tuple is bottom → top."""

    if num_stacks <= 0:
        raise ValueError("num_stacks must be positive")
    n = len(tile_list)
    if n == 0:
        return tuple(() for _ in range(num_stacks))
    base = n // num_stacks
    rem = n % num_stacks
    stacks: list[list[IslandTile]] = []
    idx = 0
    for s in range(num_stacks):
        size = base + (1 if s < rem else 0)
        chunk = tile_list[idx : idx + size]
        idx += size
        stacks.append(chunk)
    assert sum(len(st) for st in stacks) == n
    return tuple(tuple(st) for st in stacks)


def _stacks_to_mutable(stacks: tuple[tuple[IslandTile, ...], ...]) -> list[list[IslandTile]]:
    return [list(st) for st in stacks]


def _draw_face_up_round_robin(
    stacks: list[list[IslandTile]],
    count: int,
) -> tuple[tuple[IslandTile, ...], list[list[IslandTile]]]:
    """Draw `count` tiles from stack tops, round-robin order 0..N-1, 0.. (top = list end)."""

    if count < 0:
        raise ValueError("count must be nonnegative")
    face: list[IslandTile] = []
    cursor = 0
    while len(face) < count:
        before = len(face)
        for _ in range(len(stacks)):
            if len(face) >= count:
                break
            si = cursor % len(stacks)
            cursor += 1
            if stacks[si]:
                face.append(stacks[si].pop())
        if len(face) == before:
            break
    return tuple(face), stacks


def _finalize_stacks_from_mutable(stacks: list[list[IslandTile]]) -> tuple[tuple[IslandTile, ...], ...]:
    return tuple(tuple(st) for st in stacks)


def validate_initial_state(state: GameState) -> None:
    """Assert invariants for a freshly constructed match (setup validation)."""

    n = state.num_players
    assert n in SUPPORTED_PLAYER_COUNTS
    assert len(state.players) == n

    assert state.phase is Phase.ROLE_SELECTION
    assert state.governor_index == 0
    assert state.round_number == 1
    assert state.pending is None
    assert state.next_role_selector_index == state.governor_index
    assert state.round_role_order == ()
    assert state.current_role_execution_index is None
    assert state.game_end_colonists is False
    assert state.game_end_city12 is False
    assert state.game_end_vp is False
    assert sum(c for _, c in state.building_supply) == sum(s.copies_in_supply for s in BUILDING_SPECS)

    assert state.roles_in_play == roles_for_player_count(n)
    assert state.player_roles_this_round == tuple(None for _ in range(n))
    assert state.role_card_doubloons == ()

    sd = STARTING_DOUBLOONS_BY_PLAYER_COUNT[n]
    assert all(p.doubloons == sd for p in state.players)
    assert state.bank_doubloons == TOTAL_DOUBLOON_VALUE - n * sd
    assert state.bank_doubloons + sum(p.doubloons for p in state.players) == TOTAL_DOUBLOON_VALUE

    assert state.quarries_remaining == QUARRY_TILE_COUNT

    assert state.colonist_ship == COLONIST_SHIP_BY_PLAYER_COUNT[n]
    assert state.colonist_supply == COLONIST_SUPPLY_BY_PLAYER_COUNT[n]
    colonists_modeled = state.colonist_ship + state.colonist_supply + sum(
        total_colonists_on_board(p) for p in state.players
    )
    unused_in_box = COLONIST_DISCS_TOTAL - colonists_modeled
    assert unused_in_box >= 0
    assert colonists_modeled + unused_in_box == COLONIST_DISCS_TOTAL

    assert state.vp_supply == VP_SUPPLY_TOTAL_BY_PLAYER_COUNT[n]
    assert state.vp_supply <= VP_CHIPS_PHYSICAL_TOTAL_VALUE

    caps = CARGO_SHIP_CAPACITIES_BY_PLAYER_COUNT[n]
    assert len(state.cargo_ships) == 3
    assert tuple(cs.capacity for cs in state.cargo_ships) == caps
    assert all(cs.good is None and cs.barrels == 0 for cs in state.cargo_ships)

    assert state.trading_house.goods == ()

    gdict = dict(state.goods_supply)
    assert set(gdict.keys()) <= set(Good)
    assert sum(gdict.values()) == GOOD_SUPPLY_TOTAL
    for g, expect in GOOD_SUPPLY_COUNTS.items():
        assert gdict.get(g, 0) == expect

    assert len(state.face_up_plantations) == face_up_plantation_count(n)

    assert len(state.plantation_stacks) == NUM_PLANTATION_STACKS
    stacked = sum(len(st) for st in state.plantation_stacks)
    discard = len(state.plantation_discard)
    on_boards = sum(
        sum(1 for sp in p.island_spaces if sp.tile is not None and sp.tile is not IslandTile.QUARRY)
        for p in state.players
    )
    assert stacked + discard + len(state.face_up_plantations) + on_boards == PLANTATION_TILE_TOTAL

    for p in state.players:
        assert p.vp_from_chips == 0
        assert p.vp_on_paper == 0
        assert p.vp_chips_1 == 0 and p.vp_chips_5 == 0
        assert p.city_buildings == ()
        assert p.goods == _empty_goods()
        assert len(p.island_spaces) == 12
        filled = sum(1 for sp in p.island_spaces if sp.tile is not None)
        assert filled == 1
        assert p.island_spaces[0].colonists == 0


def initial_game_state(num_players: int, seed: int | None = None) -> GameState:
    """Create the canonical pre-first-round state (governor is seat 0).

    Starting plantations are removed from the plantation bag before shuffling. The remaining
    tiles are shuffled with ``random.Random(seed)``; ``seed is None`` uses ``Random``'s default
    (system-dependent, not reproducible across runs).
    """

    if num_players not in SUPPORTED_PLAYER_COUNTS:
        raise ValueError(f"num_players must be one of {SUPPORTED_PLAYER_COUNTS}, got {num_players}")

    rng = random.Random(seed)

    roles = roles_for_player_count(num_players)
    starting_row = STARTING_PLANTATIONS_BY_PLAYER_COUNT[num_players]
    assert len(starting_row) == num_players

    plantation_bag = _build_plantation_multiset()
    for t in starting_row:
        _multiset_remove(plantation_bag, t)

    rng.shuffle(plantation_bag)
    stacks_mutable = _stacks_to_mutable(_split_into_stacks(plantation_bag, NUM_PLANTATION_STACKS))
    n_face = face_up_plantation_count(num_players)
    face_up, stacks_after = _draw_face_up_round_robin(stacks_mutable, n_face)
    plantation_stacks = _finalize_stacks_from_mutable(stacks_after)

    players: list[PlayerState] = []
    for i in range(num_players):
        tile = starting_row[i]
        island = [IslandSpace(tile=None, colonists=0) for _ in range(12)]
        island[0] = IslandSpace(tile=tile, colonists=0)
        players.append(
            PlayerState(
                doubloons=STARTING_DOUBLOONS_BY_PLAYER_COUNT[num_players],
                vp_from_chips=0,
                vp_on_paper=0,
                san_juan_colonists=0,
                island_spaces=tuple(island),
                city_buildings=(),
                goods=_empty_goods(),
                vp_chips_1=0,
                vp_chips_5=0,
            )
        )

    cargo_caps = CARGO_SHIP_CAPACITIES_BY_PLAYER_COUNT[num_players]
    cargo_ships = tuple(CargoShipState(capacity=c, good=None, barrels=0) for c in cargo_caps)

    goods_supply = normalize_goods_counts(dict(GOOD_SUPPLY_COUNTS))
    building_supply = _initial_building_supply()

    state = GameState(
        num_players=num_players,
        phase=Phase.ROLE_SELECTION,
        governor_index=0,
        players=tuple(players),
        roles_in_play=roles,
        role_card_doubloons=(),
        player_roles_this_round=tuple(None for _ in range(num_players)),
        next_role_selector_index=0,
        round_role_order=(),
        current_role_execution_index=None,
        building_supply=building_supply,
        game_end_colonists=False,
        game_end_city12=False,
        game_end_vp=False,
        plantation_stacks=plantation_stacks,
        face_up_plantations=tuple(face_up),
        plantation_discard=(),
        quarries_remaining=QUARRY_TILE_COUNT,
        colonist_ship=COLONIST_SHIP_BY_PLAYER_COUNT[num_players],
        colonist_supply=COLONIST_SUPPLY_BY_PLAYER_COUNT[num_players],
        bank_doubloons=TOTAL_DOUBLOON_VALUE - num_players * STARTING_DOUBLOONS_BY_PLAYER_COUNT[num_players],
        goods_supply=goods_supply,
        vp_supply=VP_SUPPLY_TOTAL_BY_PLAYER_COUNT[num_players],
        cargo_ships=cargo_ships,
        trading_house=TradingHouseState(goods=()),
        round_number=1,
        pending=None,
    )

    validate_initial_state(state)
    return state


def refresh_face_up_plantations_after_settler(state: GameState) -> GameState:
    """Discard current face-up market to discard pile; draw (players+1) new face-up from stacks."""

    discard = list(state.plantation_discard) + list(state.face_up_plantations)
    stacks = [list(st) for st in state.plantation_stacks]
    n = face_up_plantation_count(state.num_players)
    face, stacks2 = _draw_face_up_round_robin(stacks, n)
    return dataclasses.replace(
        state,
        face_up_plantations=tuple(face),
        plantation_stacks=tuple(tuple(x) for x in stacks2),
        plantation_discard=tuple(discard),
    )


__all__ = ["initial_game_state", "refresh_face_up_plantations_after_settler", "validate_initial_state"]
