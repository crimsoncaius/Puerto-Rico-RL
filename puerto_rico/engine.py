# State machine design (compact):
# ---------------------------------------------------------------------------
# ROLE_SELECTION: next_role_selector_index chooses in governor→clockwise order.
#   Each PickRole appends (Role, chooser) to round_role_order and pays doubloons stacked
#   on that role card. When len == num_players, execution starts at round_role_order[0].
#
# ROLE_EXECUTION (phase is one of SETTLER..PROSPECTOR): current_role_execution_index selects
#   which tuple in round_role_order is active. Chooser acts first, then clockwise (pending
#   structs carry cursors). When a role finishes, index increments or ROUND_CLEANUP begins.
#
# ROUND_CLEANUP: unused roles gain +1 doubloon on their stacks; role cards clear; governor
#   passes clockwise; if any game_end_* flag, GAME_OVER else ROLE_SELECTION for next round.
#
# MAYOR micro: privilege (mayor only) → draft one colonist at a time from ship → placement
#   one colonist at a time → automatic ship refill (colonist shortage may set game_end_colonists).
#
# CAPTAIN micro: loading loop (mandatory load when any legal cargo load exists; Wharf may
#   satisfy obligation once per player) → storage trims per player → unload full ships.
#
# Edge-case flags (game_end_*) are set when triggers fire; GAME_OVER is entered at round cleanup
# except where rules require immediate tracking (VP on paper after chips run out).
# ---------------------------------------------------------------------------

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from itertools import product
from enum import Enum
from typing import Callable, Final, Iterable, Optional, Sequence, TypeAlias, Union

from .constants import (
    BUILDING_METADATA,
    GOOD_SUPPLY_COUNTS,
    LARGE_UNIQUE_BUILDINGS,
    PRODUCTION_BUILDINGS,
    VIOLET_BUILDINGS,
    VIOLET_SMALL_BUILDINGS,
    BuildingCategory,
    roles_for_player_count,
)
from .setup import initial_game_state, refresh_face_up_plantations_after_settler
from .state import (
    Building,
    BuildingSupplyCounts,
    CaptainPhasePending,
    CargoShipState,
    CraftsmanPhasePending,
    GameState,
    Good,
    GoodsCounts,
    IslandSpace,
    IslandTile,
    MayorPhasePending,
    Phase,
    PlacedBuilding,
    PlayerState,
    Role,
    SettlerPhasePending,
    TraderPhasePending,
    TradingHouseState,
    BuilderPhasePending,
    building_city_spaces,
    building_occupied,
    building_printed_cost,
    building_printed_vp,
    building_worker_circles,
    clockwise_indices,
    count_empty_building_circles,
    count_filled_island_spaces,
    count_occupied_quarries,
    goods_dict,
    good_count,
    island_space_occupied,
    island_tile_max_colonists,
    max_quarry_discount_for_city_slot,
    normalize_goods_counts,
    normalize_building_supply,
    role_action_order,
    total_colonists_on_board,
    goods_total,
)


# ---------------------------------------------------------------------------
# Actions (typed; engine is the sole interpreter)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PickRole:
    """Take one available role card from the table (governor→clockwise). Doubloons on the card move with the pick."""

    role: Role


@dataclass(frozen=True, slots=True)
class SettlerTakeFaceUp:
    """Take one face-up plantation from the market (index into sorted face-up list)."""

    face_up_index: int


@dataclass(frozen=True, slots=True)
class SettlerTakeQuarryPrivilege:
    """Settler privilege: take one quarry instead of a face-up plantation (chooser only)."""


@dataclass(frozen=True, slots=True)
class SettlerTakeQuarryConstructionHut:
    """Construction Hut: take a quarry instead of a face-up plantation (non-chooser)."""


@dataclass(frozen=True, slots=True)
class SettlerTakeHacienda:
    """Hacienda: take the top face-down plantation before the normal settler pick."""


@dataclass(frozen=True, slots=True)
class SettlerPass:
    """Skip settler action on your turn (optional action for non-captain roles)."""


@dataclass(frozen=True, slots=True)
class MayorPrivilegeTake:
    """Mayor privilege: take one colonist from supply before the ship is drafted."""


@dataclass(frozen=True, slots=True)
class MayorPrivilegeSkip:
    """Mayor may skip privilege (still legal if supply empty — handled by legality)."""


@dataclass(frozen=True, slots=True)
class MayorSubmitPlacement:
    """Submit the player's final mayor allocation for their full pooled colonists."""

    island_targets: tuple[int, ...]
    building_targets: tuple[int, ...]
    san_juan: int


@dataclass(frozen=True, slots=True)
class BuilderBuild:
    """Pay cost and place one building at anchor_slot (privilege discount applied for chooser)."""

    building: Building
    anchor_slot: int


@dataclass(frozen=True, slots=True)
class BuilderPass:
    """Skip building this round (no builder privilege discount if you do not build)."""


@dataclass(frozen=True, slots=True)
class BuilderNoOp:
    """Internal: non-chooser has no affordable building; skip turn and advance pending actor."""


@dataclass(frozen=True, slots=True)
class CraftsmanTurn:
    """Resolve craftsman production for the active player; privilege_good is chooser's extra barrel."""

    privilege_good: Optional[Good]
    hacienda_good: Optional[Good] = None


@dataclass(frozen=True, slots=True)
class TraderSell:
    """Sell one good to the trading house at the standard price (+ trader privilege if chooser)."""

    good: Good


@dataclass(frozen=True, slots=True)
class TraderPass:
    """Skip selling on your trader turn."""


@dataclass(frozen=True, slots=True)
class CaptainLoad:
    """Load one kind onto a cargo ship; amount is maximal for that ship/good choice (rules)."""

    good: Good
    ship_index: int


@dataclass(frozen=True, slots=True)
class CaptainUseWharf:
    """Wharf: once per phase, send all barrels of one kind to supply and score as if shipped."""

    good: Good


@dataclass(frozen=True, slots=True)
class CaptainPassLoading:
    """Pass loading turn only when no legal cargo load and Wharf cannot satisfy obligation."""


@dataclass(frozen=True, slots=True)
class CaptainStorageCommit:
    """After loading ends: choose barrels to keep on windrose; rest return to supply."""

    keep_counts: GoodsCounts


@dataclass(frozen=True, slots=True)
class ProspectorCollect:
    """Take privilege doubloon(s) from bank (role-card doubloons were taken with PickRole)."""


@dataclass(frozen=True, slots=True)
class RoundCleanupAdvance:
    """Internal sentinel — round cleanup is automatic; exposed for a uniform apply API if needed."""

    pass


EngineAction: TypeAlias = Union[
    PickRole,
    SettlerTakeHacienda,
    SettlerTakeFaceUp,
    SettlerTakeQuarryPrivilege,
    SettlerTakeQuarryConstructionHut,
    SettlerPass,
    MayorPrivilegeTake,
    MayorPrivilegeSkip,
    MayorSubmitPlacement,
    BuilderBuild,
    BuilderPass,
    BuilderNoOp,
    CraftsmanTurn,
    TraderSell,
    TraderPass,
    CaptainLoad,
    CaptainUseWharf,
    CaptainPassLoading,
    CaptainStorageCommit,
    ProspectorCollect,
]


# ---------------------------------------------------------------------------
# Pricing & small static tables
# ---------------------------------------------------------------------------

_TRADER_PRICE: Final[dict[Good, int]] = {
    Good.CORN: 0,
    Good.INDIGO: 1,
    Good.SUGAR: 2,
    Good.TOBACCO: 3,
    Good.COFFEE: 4,
}

_ISLAND_TILE_TO_GOOD: Final[dict[IslandTile, Good]] = {
    IslandTile.CORN: Good.CORN,
    IslandTile.INDIGO: Good.INDIGO,
    IslandTile.SUGAR: Good.SUGAR,
    IslandTile.TOBACCO: Good.TOBACCO,
    IslandTile.COFFEE: Good.COFFEE,
}


def _role_to_execution_phase(role: Role) -> Phase:
    if role in (Role.PROSPECTOR, Role.PROSPECTOR_A, Role.PROSPECTOR_B):
        return Phase.PROSPECTOR
    return Phase(str(role.value))


def _prospectors_equivalent(a: Role, b: Role) -> bool:
    sa = {Role.PROSPECTOR, Role.PROSPECTOR_A, Role.PROSPECTOR_B}
    return a == b or (a in sa and b in sa)


def _goods_add(g: MappingLikeGoods, good: Good, n: int) -> dict[Good, int]:
    d = dict(goods_dict(g)) if not isinstance(g, dict) else dict(g)
    d[good] = d.get(good, 0) + n
    return d


def _goods_sub(g: MappingLikeGoods, good: Good, n: int) -> dict[Good, int]:
    d = dict(goods_dict(g)) if not isinstance(g, dict) else dict(g)
    d[good] = max(0, d.get(good, 0) - n)
    if d[good] == 0:
        del d[good]
    return d


MappingLikeGoods = Union[GoodsCounts, dict[Good, int]]


def _bank_pay(state: GameState, player_index: int, amount: int) -> tuple[GameState, None | str]:
    if amount < 0:
        return state, "negative payment"
    p = state.players[player_index]
    if p.doubloons < amount:
        return state, "insufficient doubloons"
    new_players = list(state.players)
    new_players[player_index] = PlayerState(
        doubloons=p.doubloons - amount,
        vp_from_chips=p.vp_from_chips,
        vp_on_paper=p.vp_on_paper,
        san_juan_colonists=p.san_juan_colonists,
        island_spaces=p.island_spaces,
        city_buildings=p.city_buildings,
        goods=p.goods,
        vp_chips_1=p.vp_chips_1,
        vp_chips_5=p.vp_chips_5,
    )
    return (
        dataclasses.replace(
            state,
            players=tuple(new_players),
            bank_doubloons=state.bank_doubloons + amount,
        ),
        None,
    )


def _bank_receive(state: GameState, player_index: int, amount: int) -> GameState:
    p = state.players[player_index]
    new_players = list(state.players)
    new_players[player_index] = PlayerState(
        doubloons=p.doubloons + amount,
        vp_from_chips=p.vp_from_chips,
        vp_on_paper=p.vp_on_paper,
        san_juan_colonists=p.san_juan_colonists,
        island_spaces=p.island_spaces,
        city_buildings=p.city_buildings,
        goods=p.goods,
        vp_chips_1=p.vp_chips_1,
        vp_chips_5=p.vp_chips_5,
    )
    return dataclasses.replace(
        state,
        players=tuple(new_players),
        bank_doubloons=state.bank_doubloons - amount,
    )


def _role_doubloons_map(state: GameState) -> dict[Role, int]:
    return {r: n for r, n in state.role_card_doubloons}


def _set_role_doubloons(state: GameState, m: dict[Role, int]) -> GameState:
    tup = tuple(sorted(((r, m[r]) for r in m if m[r] > 0), key=lambda x: x[0].value))
    return dataclasses.replace(state, role_card_doubloons=tup)


def _replace_player(state: GameState, idx: int, p: PlayerState) -> GameState:
    pl = list(state.players)
    pl[idx] = p
    return dataclasses.replace(state, players=tuple(pl))


def _occupied_plantation_counts(player: PlayerState) -> dict[IslandTile, int]:
    out: dict[IslandTile, int] = {}
    for sp in player.island_spaces:
        if sp.tile is None or sp.tile is IslandTile.QUARRY:
            continue
        if not island_space_occupied(sp):
            continue
        out[sp.tile] = out.get(sp.tile, 0) + 1
    return out


def _building_produces_good(b: Building) -> Optional[Good]:
    m = {
        Building.SMALL_INDIGO_PLANT: Good.INDIGO,
        Building.LARGE_INDIGO_PLANT: Good.INDIGO,
        Building.SMALL_SUGAR_MILL: Good.SUGAR,
        Building.LARGE_SUGAR_MILL: Good.SUGAR,
        Building.TOBACCO_STORAGE: Good.TOBACCO,
        Building.COFFEE_ROASTER: Good.COFFEE,
    }
    return m.get(b)


def _production_capacity_for_good(player: PlayerState, good: Good) -> int:
    """Max barrels from buildings (circles with colonists), capped by matching plantations."""
    plant_tile = {
        Good.INDIGO: IslandTile.INDIGO,
        Good.SUGAR: IslandTile.SUGAR,
        Good.TOBACCO: IslandTile.TOBACCO,
        Good.COFFEE: IslandTile.COFFEE,
        Good.CORN: IslandTile.CORN,
    }[good]
    plantations = _occupied_plantation_counts(player).get(plant_tile, 0)
    if good == Good.CORN:
        # Corn has no production building; one barrel per occupied corn plantation.
        return plantations

    cap = 0
    for pb in player.city_buildings:
        if not building_occupied(pb):
            continue
        if _building_produces_good(pb.building) != good:
            continue
        # Each occupied worker circle can produce at most one barrel, limited by plantations.
        cap += sum(1 for c in pb.colonists if c > 0)
    return min(cap, plantations)


def _compute_craftsman_production(player: PlayerState, goods_supply: MappingLikeGoods) -> dict[Good, int]:
    """Barrels actually produced this craftsman phase (supply-limited per kind)."""
    out: dict[Good, int] = {}
    sup = goods_dict(goods_supply)
    for g in Good:
        cap = _production_capacity_for_good(player, g)
        take = min(cap, sup.get(g, 0))
        if take:
            out[g] = take
    return out


def _factory_bonus(kinds: int) -> int:
    if kinds <= 1:
        return 0
    return {2: 1, 3: 2, 4: 3, 5: 5}.get(kinds, 5)


def _city_occupied_slots(player: PlayerState) -> set[int]:
    occ: set[int] = set()
    for pb in player.city_buildings:
        w = building_city_spaces(pb.building)
        for i in range(w):
            occ.add(pb.anchor_slot + i)
    return occ


def _can_place_building_at(player: PlayerState, building: Building, anchor: int) -> bool:
    w = building_city_spaces(building)
    if anchor < 0 or anchor + w > 12:
        return False
    occ = _city_occupied_slots(player)
    for i in range(w):
        if anchor + i in occ:
            return False
    # Horizontal adjacency only (matches state modeling notes: anchor and anchor+1).
    if w == 2 and anchor % 4 == 3:
        return False
    return True


def _repack_city_buildings(
    existing: Sequence[PlacedBuilding],
    new_building: Building,
    new_anchor: int,
) -> Optional[tuple[PlacedBuilding, ...]]:
    """Return a valid city layout after adding `new_building`, relocating others if needed.

    The rules permit moving buildings within the city to make room for a large building.
    This helper only repacks geometry; building ownership and colonists stay attached
    to their tiles.
    """

    new_width = building_city_spaces(new_building)
    if new_anchor < 0 or new_anchor + new_width > 12:
        return None
    if new_width == 2 and new_anchor % 4 == 3:
        return None

    reserved = {new_anchor + i for i in range(new_width)}
    if any(slot < 0 or slot >= 12 for slot in reserved):
        return None

    to_place = sorted(existing, key=lambda pb: (-building_city_spaces(pb.building), pb.anchor_slot, pb.building.value))
    placed: list[PlacedBuilding] = []
    occupied = set(reserved)

    def backtrack(idx: int) -> bool:
        if idx >= len(to_place):
            return True
        pb = to_place[idx]
        width = building_city_spaces(pb.building)
        for anchor in range(12):
            if anchor + width > 12:
                continue
            if width == 2 and anchor % 4 == 3:
                continue
            slots = {anchor + i for i in range(width)}
            if slots & occupied:
                continue
            occupied.update(slots)
            placed.append(dataclasses.replace(pb, anchor_slot=anchor))
            if backtrack(idx + 1):
                return True
            placed.pop()
            occupied.difference_update(slots)
        return False

    if not backtrack(0):
        return None

    return tuple(
        list(placed)
        + [
            PlacedBuilding(
                building=new_building,
                anchor_slot=new_anchor,
                colonists=tuple(0 for _ in range(building_worker_circles(new_building))),
            )
        ]
    )


def _builder_discount(player: PlayerState, anchor_slot: int, building: Building) -> int:
    """Quarry discount capped by building column (1–4)."""
    col_cap = max_quarry_discount_for_city_slot(anchor_slot)
    quarries = count_occupied_quarries(player)
    return min(quarries, col_cap)


def _player_has_building(player: PlayerState, b: Building) -> bool:
    return any(pb.building == b for pb in player.city_buildings)


def _player_get_building(player: PlayerState, b: Building) -> Optional[PlacedBuilding]:
    return next((pb for pb in player.city_buildings if pb.building == b), None)


def _building_supply_take(state: GameState, b: Building) -> GameState | None:
    m = dict(state.building_supply)
    if m.get(b, 0) <= 0:
        return None
    m[b] -= 1
    if m[b] == 0:
        del m[b]
    return dataclasses.replace(state, building_supply=normalize_building_supply(m))


def _count_filled_city_slots(player: PlayerState) -> int:
    return sum(building_city_spaces(pb.building) for pb in player.city_buildings)


def _builder_doubloon_cost_base(
    player: PlayerState, building: Building, anchor_slot: int, *, quarry_discount_applies: bool
) -> int:
    """max(1, printed_cost - quarry_discount). Quarry discount applies only to the builder (chooser)."""
    printed = building_printed_cost(building)
    qd = _builder_discount(player, anchor_slot, building) if quarry_discount_applies else 0
    return max(1, printed - qd)


def _occupied_building_bonus_eligible(player: PlayerState, b: Building) -> bool:
    pb = _player_get_building(player, b)
    if pb is None:
        return False
    return building_occupied(pb)


def _take_one_colonist_from_supply_or_ship(state: GameState) -> tuple[GameState, bool]:
    if state.colonist_supply > 0:
        return dataclasses.replace(state, colonist_supply=state.colonist_supply - 1), True
    if state.colonist_ship > 0:
        return dataclasses.replace(state, colonist_ship=state.colonist_ship - 1), True
    return state, False


def _draw_hacienda_plantation(state: GameState) -> tuple[GameState, Optional[IslandTile]]:
    stacks = [list(st) for st in state.plantation_stacks]
    for idx, stack in enumerate(stacks):
        if not stack:
            continue
        tile = stack.pop()
        return dataclasses.replace(state, plantation_stacks=tuple(tuple(st) for st in stacks)), tile
    return state, None


def _place_colonist_on_new_island_tile(
    state: GameState,
    player_index: int,
    island_slot: int,
) -> GameState:
    state2, got_colonist = _take_one_colonist_from_supply_or_ship(state)
    if not got_colonist:
        return state
    p = state2.players[player_index]
    spaces = list(p.island_spaces)
    sp = spaces[island_slot]
    spaces[island_slot] = IslandSpace(tile=sp.tile, colonists=sp.colonists + 1)
    return _replace_player(state2, player_index, dataclasses.replace(p, island_spaces=tuple(spaces)))


def _maybe_apply_hospice(state: GameState, player_index: int, island_slot: int) -> GameState:
    p = state.players[player_index]
    if not _occupied_building_bonus_eligible(p, Building.HOSPICE):
        return state
    return _place_colonist_on_new_island_tile(state, player_index, island_slot)


def _maybe_apply_university(state: GameState, player_index: int, built_building: Building) -> GameState:
    p = state.players[player_index]
    uni = _player_get_building(p, Building.UNIVERSITY)
    if uni is None:
        return state
    if built_building is not Building.UNIVERSITY and not building_occupied(uni):
        return state
    state2, got_colonist = _take_one_colonist_from_supply_or_ship(state)
    if not got_colonist:
        return state
    p2 = state2.players[player_index]
    city = list(p2.city_buildings)
    for i, pb in enumerate(city):
        if pb.building != Building.UNIVERSITY:
            continue
        cols = list(pb.colonists)
        for j, c in enumerate(cols):
            if c == 0:
                cols[j] = 1
                city[i] = dataclasses.replace(pb, colonists=tuple(cols))
                return _replace_player(state2, player_index, dataclasses.replace(p2, city_buildings=tuple(city)))
        break
    return state


def _collect_player_colonists_for_mayor(player: PlayerState) -> tuple[PlayerState, int]:
    """Return player with board/San Juan cleared and all colonists moved into a placement pool."""

    pooled = player.san_juan_colonists

    island_spaces: list[IslandSpace] = []
    for sp in player.island_spaces:
        pooled += sp.colonists
        island_spaces.append(IslandSpace(tile=sp.tile, colonists=0 if sp.tile is not None else 0))

    city_buildings: list[PlacedBuilding] = []
    for pb in player.city_buildings:
        pooled += sum(pb.colonists)
        city_buildings.append(dataclasses.replace(pb, colonists=tuple(0 for _ in pb.colonists)))

    return (
        dataclasses.replace(
            player,
            san_juan_colonists=0,
            island_spaces=tuple(island_spaces),
            city_buildings=tuple(city_buildings),
        ),
        pooled,
    )


def _prepare_mayor_placement_state(
    state: GameState,
    pend: MayorPhasePending,
) -> tuple[GameState, MayorPhasePending, Optional[int]]:
    """During mayor, auto-distribute the ship and pool all colonists for final placement."""

    pools = list(pend.placement_pools[: state.num_players])
    if len(pools) < state.num_players:
        pools.extend(0 for _ in range(state.num_players - len(pools)))
    for offset in range(state.colonist_ship):
        pools[(pend.mayor_role_chooser + offset) % state.num_players] += 1
    players = list(state.players)
    for idx, player in enumerate(players):
        cleared_player, pooled = _collect_player_colonists_for_mayor(player)
        players[idx] = cleared_player
        pools[idx] += pooled
    pend2 = dataclasses.replace(pend, subphase="placement", placement_pools=tuple(pools))
    state2 = dataclasses.replace(state, players=tuple(players), colonist_ship=0, pending=pend2)
    placement_next = PuertoRicoEngine._mayor_next_player_with_pool(
        pend2.mayor_role_chooser, state2.num_players, pend2.placement_pools
    )
    return state2, pend2, placement_next


def _final_scoring_bonus(player: PlayerState) -> int:
    bonus = 0
    occupied_large = {
        pb.building
        for pb in player.city_buildings
        if pb.building in LARGE_UNIQUE_BUILDINGS and building_occupied(pb)
    }
    if Building.GUILD_HALL in occupied_large:
        for pb in player.city_buildings:
            if pb.building not in PRODUCTION_BUILDINGS:
                continue
            bonus += 1 if pb.building in (Building.SMALL_INDIGO_PLANT, Building.SMALL_SUGAR_MILL) else 2
    if Building.RESIDENCE in occupied_large:
        filled = count_filled_island_spaces(player)
        bonus += {9: 4, 10: 5, 11: 6}.get(filled, 7 if filled >= 12 else 0)
    if Building.FORTRESS in occupied_large:
        bonus += total_colonists_on_board(player) // 3
    if Building.CUSTOMS_HOUSE in occupied_large:
        bonus += player.vp_from_chips // 4
    if Building.CITY_HALL in occupied_large:
        bonus += sum(1 for pb in player.city_buildings if pb.building in VIOLET_BUILDINGS)
    return bonus


def _player_can_afford_any_build(state: GameState, player_id: int, chooser: int) -> bool:
    p = state.players[player_id]
    priv = 1 if player_id == chooser else 0
    quarry_ok = player_id == chooser
    for b in sorted(BUILDING_METADATA.keys(), key=lambda x: x.value):
        if dict(state.building_supply).get(b, 0) <= 0:
            continue
        if _player_has_building(p, b):
            continue
        for slot in range(12):
            if building_city_spaces(b) == 2:
                if _repack_city_buildings(p.city_buildings, b, slot) is None:
                    continue
            elif not _can_place_building_at(p, b, slot):
                continue
            cost = _builder_doubloon_cost_base(p, b, slot, quarry_discount_applies=quarry_ok)
            if p.doubloons + priv >= cost:
                return True
    return False


def _pay_building_vp_from_supply(
    state: GameState, player_index: int, printed_vp: int
) -> tuple[GameState, None | str]:
    """Printed VP is paid from the VP supply; shortfall is vp_on_paper (endgame still uses tile printed VP)."""
    if printed_vp < 0:
        return state, "negative building vp"
    take = min(printed_vp, state.vp_supply)
    paper = printed_vp - take
    new_vp_supply = state.vp_supply - take
    game_end_vp = state.game_end_vp
    if state.vp_supply > 0 and new_vp_supply == 0:
        game_end_vp = True
    p = state.players[player_index]
    pl = list(state.players)
    pl[player_index] = PlayerState(
        doubloons=p.doubloons,
        vp_from_chips=p.vp_from_chips,
        vp_on_paper=p.vp_on_paper + paper,
        san_juan_colonists=p.san_juan_colonists,
        island_spaces=p.island_spaces,
        city_buildings=p.city_buildings,
        goods=p.goods,
        vp_chips_1=p.vp_chips_1,
        vp_chips_5=p.vp_chips_5,
    )
    return (
        dataclasses.replace(
            state,
            players=tuple(pl),
            vp_supply=new_vp_supply,
            game_end_vp=game_end_vp,
        ),
        None,
    )


def _trading_house_allows_good(house: TradingHouseState, good: Good, player: PlayerState) -> bool:
    if good in house.goods:
        return _occupied_building_bonus_eligible(player, Building.OFFICE)
    return True


def _captain_ship_accepts_good(
    ships: tuple[CargoShipState, ...], ship_index: int, good: Good
) -> bool:
    """Empty ship may not take a good that is already on another ship (no duplicate kinds across ships)."""
    sh = ships[ship_index]
    if sh.good is not None:
        return sh.good == good
    for j, other in enumerate(ships):
        if j != ship_index and other.good == good:
            return False
    return True


def _captain_max_load_on_ship(
    ships: tuple[CargoShipState, ...],
    ship_index: int,
    good: Good,
    player_goods: dict[Good, int],
) -> int:
    """Max barrels of `good` on this ship this turn (one kind per turn; duplicate kinds across ships forbidden)."""
    sh = ships[ship_index]
    have = player_goods.get(good, 0)
    if have <= 0:
        return 0
    if not _captain_ship_accepts_good(ships, ship_index, good):
        return 0
    if sh.good is not None and sh.good != good:
        return 0
    space = sh.capacity - sh.barrels
    if space <= 0:
        return 0
    return min(have, space)


def _best_ship_and_amount_for_good(
    ships: tuple[CargoShipState, ...],
    good: Good,
    player_goods: dict[Good, int],
) -> tuple[int, int]:
    """Rulebook: among ships that can take this good, load where the amount is maximal (ties → lowest ship index)."""
    best_i = -1
    best_amt = 0
    for i in range(len(ships)):
        amt = _captain_max_load_on_ship(ships, i, good, player_goods)
        if amt > best_amt:
            best_amt = amt
            best_i = i
    return best_i, best_amt


def _legal_cargo_loads_for_player(
    state: GameState,
    player_index: int,
) -> list[tuple[Good, int, int]]:
    """One legal cargo load per good type: (good, best_ship_index, amount). Wharf is separate."""
    p = state.players[player_index]
    gd = goods_dict(p.goods)
    ships = state.cargo_ships
    out: list[tuple[Good, int, int]] = []
    for g in Good:
        if gd.get(g, 0) <= 0:
            continue
        si, amt = _best_ship_and_amount_for_good(ships, g, gd)
        if amt > 0 and si >= 0:
            out.append((g, si, amt))
    return out


def _any_legal_cargo_load(state: GameState, player_index: int) -> bool:
    return len(_legal_cargo_loads_for_player(state, player_index)) > 0


def _harbor_bonus(player: PlayerState) -> bool:
    return _occupied_building_bonus_eligible(player, Building.HARBOR)


def _wharf_available(state: GameState, player_index: int, pending: CaptainPhasePending) -> bool:
    if player_index < 0 or player_index >= len(pending.wharf_used):
        wharf_already_used = False
    else:
        wharf_already_used = pending.wharf_used[player_index]
    if wharf_already_used:
        return False
    return _occupied_building_bonus_eligible(state.players[player_index], Building.WHARF)


def _occupied_violet_building(player: PlayerState, b: Building) -> bool:
    return _occupied_building_bonus_eligible(player, b)


def _trader_sell_price(player: PlayerState, good: Good) -> int:
    price = _TRADER_PRICE[good]
    if _occupied_violet_building(player, Building.SMALL_MARKET):
        price += 1
    if _occupied_violet_building(player, Building.LARGE_MARKET):
        price += 2
    return price


def _storage_keep_is_valid(player: PlayerState, keep: dict[Good, int], have: dict[Good, int]) -> bool:
    """End of captain: default keep 1 barrel; warehouses extend retention (rulebook storage section)."""
    if any(keep.get(g, 0) > have.get(g, 0) for g in Good):
        return False
    total_have = sum(have.get(g, 0) for g in Good)
    total_keep = sum(keep.values())
    # You must retain at least one barrel if you still have goods after loading (default windrose rule).
    if total_have > 0 and total_keep < 1:
        return False
    if total_keep == 0:
        return total_have == 0
    sm = _occupied_violet_building(player, Building.SMALL_WAREHOUSE)
    lw = _occupied_violet_building(player, Building.LARGE_WAREHOUSE)
    full_kinds = {g for g in Good if have.get(g, 0) > 0 and keep.get(g, 0) == have[g]}
    if not sm and not lw:
        return total_keep <= 1
    if sm and not lw:
        # Small: keep all of one chosen kind, plus the normal 1-barrel windrose allowance.
        if len(full_kinds) > 1:
            return False
        if not full_kinds:
            return total_keep <= 1
        (gk,) = tuple(full_kinds)
        return total_keep - keep.get(gk, 0) <= 1
    if lw and not sm:
        if len(full_kinds) > 2:
            return False
        if len(full_kinds) < 2:
            return total_keep <= 1
        g1, g2 = tuple(full_kinds)
        return total_keep - keep[g1] - keep[g2] <= 1
    # Both warehouses: keep all of three kinds + 1 barrel (rulebook: three kinds + default).
    if len(full_kinds) > 3:
        return False
    if len(full_kinds) < 3:
        return total_keep <= 1
    return total_keep - sum(keep[g] for g in full_kinds) <= 1


def _enumerate_valid_storage_keeps(player: PlayerState) -> list[dict[Good, int]]:
    have = goods_dict(player.goods)
    if not any(have.values()):
        return [{}]
    out: list[dict[Good, int]] = []
    gorder = tuple(sorted(Good, key=lambda x: x.value))
    ranges = [range(0, have.get(g, 0) + 1) for g in gorder]
    for tup in product(*ranges):
        cand = {gorder[i]: tup[i] for i in range(len(gorder))}
        if _storage_keep_is_valid(player, cand, have):
            out.append(cand)
    return out


def _global_anyone_can_load(state: GameState, pend: CaptainPhasePending) -> bool:
    """Loading continues while at least one player can still legally ship (cargo or Wharf)."""
    for i in range(state.num_players):
        if _any_legal_cargo_load(state, i):
            return True
        if _wharf_available(state, i, pend) and goods_total(goods_dict(state.players[i].goods)) > 0:
            return True
    return False


def _replace_cargo_ships(state: GameState, ships: tuple[CargoShipState, ...]) -> GameState:
    return dataclasses.replace(state, cargo_ships=ships)


def _award_vp_from_supply(
    state: GameState, player_index: int, amount: int
) -> tuple[GameState, None | str]:
    """VP chips first; remainder is paper VP (still counts for Customs House). Last chip taken ends game."""
    if amount < 0:
        return state, "negative vp"
    take = min(amount, state.vp_supply)
    paper = amount - take
    new_vp_supply = state.vp_supply - take
    game_end_vp = state.game_end_vp
    if state.vp_supply > 0 and new_vp_supply == 0:
        game_end_vp = True
    p = state.players[player_index]
    new_vp_total = p.vp_from_chips + take + paper
    pl = list(state.players)
    pl[player_index] = PlayerState(
        doubloons=p.doubloons,
        vp_from_chips=new_vp_total,
        vp_on_paper=p.vp_on_paper,
        san_juan_colonists=p.san_juan_colonists,
        island_spaces=p.island_spaces,
        city_buildings=p.city_buildings,
        goods=p.goods,
        vp_chips_1=p.vp_chips_1,
        vp_chips_5=p.vp_chips_5,
    )
    return (
        dataclasses.replace(
            state,
            players=tuple(pl),
            vp_supply=new_vp_supply,
            game_end_vp=game_end_vp,
        ),
        None,
    )


def _goods_supply_take(state: GameState, good: Good, n: int) -> tuple[GameState, Optional[str]]:
    if n <= 0:
        return state, "nonpositive goods take"
    d = goods_dict(state.goods_supply)
    if d.get(good, 0) < n:
        return state, "insufficient goods supply"
    d[good] = d.get(good, 0) - n
    if d[good] == 0:
        del d[good]
    return dataclasses.replace(state, goods_supply=normalize_goods_counts(d)), None


def _goods_supply_add(state: GameState, good: Good, n: int) -> GameState:
    if n <= 0:
        return state
    d = goods_dict(state.goods_supply)
    d[good] = d.get(good, 0) + n
    return dataclasses.replace(state, goods_supply=normalize_goods_counts(d))


def _player_add_goods(state: GameState, player_index: int, good: Good, n: int) -> GameState:
    if n <= 0:
        return state
    p = state.players[player_index]
    d = goods_dict(p.goods)
    d[good] = d.get(good, 0) + n
    return _replace_player(state, player_index, dataclasses.replace(p, goods=normalize_goods_counts(d)))


def _player_sub_goods(state: GameState, player_index: int, good: Good, n: int) -> tuple[GameState, Optional[str]]:
    if n < 0:
        return state, "negative sub goods"
    p = state.players[player_index]
    d = goods_dict(p.goods)
    if d.get(good, 0) < n:
        return state, "insufficient goods on windrose"
    d[good] = d.get(good, 0) - n
    if d[good] == 0:
        del d[good]
    return _replace_player(state, player_index, dataclasses.replace(p, goods=normalize_goods_counts(d))), None


def _windrose_storage_capacity(player: PlayerState) -> int:
    """Default 1 barrel; warehouses add full kinds."""
    base = 1
    # Small warehouse: +all of one kind — modeled as extra capacity via storage phase, not a simple +N.
    # We use explicit CaptainStorageCommit for all trimming.
    if _player_has_building(player, Building.SMALL_WAREHOUSE) and building_occupied(
        next(pb for pb in player.city_buildings if pb.building == Building.SMALL_WAREHOUSE)
    ):
        base = max(base, 999)  # resolved in storage commit by kind
    return base


class PuertoRicoEngine:
    """Pure rules engine: transitions, legality, terminal checks, and final scoring."""

    __slots__ = ("_rng", "_state")

    def __init__(self) -> None:
        self._rng: random.Random = random.Random()
        self._state: GameState = initial_game_state(3, seed=0)

    # -- public API ---------------------------------------------------------

    def reset(self, num_players: int, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self._state = initial_game_state(num_players, seed=seed)

    @property
    def state(self) -> GameState:
        return self._state

    def legal_actions(self, player_id: int) -> tuple[EngineAction, ...]:
        return tuple(self._legal_actions_impl(player_id))

    def is_legal(self, player_id: int, action: EngineAction) -> bool:
        if self._state.phase is Phase.MAYOR and isinstance(self._state.pending, MayorPhasePending):
            if self._state.pending.subphase == "placement":
                return self._validate_mayor_placement(player_id, action, self._state.pending) is None
        return action in self.legal_actions(player_id)

    def acting_player(self) -> Optional[int]:
        s = self._state
        if s.phase in (Phase.GAME_OVER, Phase.ROUND_CLEANUP):
            return None
        if s.phase is Phase.ROLE_SELECTION:
            return s.next_role_selector_index
        idx = s.current_role_execution_index
        if idx is None or idx < 0 or idx >= len(s.round_role_order):
            return None
        _role, chooser = s.round_role_order[idx]
        if s.phase is Phase.SETTLER and isinstance(s.pending, SettlerPhasePending):
            return s.pending.next_actor_index
        if s.phase is Phase.MAYOR and isinstance(s.pending, MayorPhasePending):
            if s.pending.subphase == "privilege":
                return s.pending.mayor_role_chooser
            return s.pending.placement_next
        if s.phase is Phase.BUILDER and isinstance(s.pending, BuilderPhasePending):
            return s.pending.next_actor
        if s.phase is Phase.CRAFTSMAN and isinstance(s.pending, CraftsmanPhasePending):
            return s.pending.next_actor
        if s.phase is Phase.TRADER and isinstance(s.pending, TraderPhasePending):
            return s.pending.next_actor
        if s.phase is Phase.CAPTAIN and isinstance(s.pending, CaptainPhasePending):
            if s.pending.subphase == "loading":
                return s.pending.active_player_index
            if s.pending.subphase == "storage":
                return s.pending.storage_next_actor
            return None
        if s.phase is Phase.PROSPECTOR:
            return chooser
        return None

    def apply(self, player_id: int, action: EngineAction) -> None:
        err = self._apply_impl(player_id, action)
        if err:
            raise ValueError(err)

    def is_terminal(self) -> bool:
        return self._state.phase is Phase.GAME_OVER

    def final_scores(self) -> tuple[int, ...]:
        if not self.is_terminal():
            raise RuntimeError("final_scores() is only valid in GAME_OVER")
        return tuple(self._final_score_player(i) for i in range(self._state.num_players))

    def _final_score_player(self, player_index: int) -> int:
        p = self._state.players[player_index]
        printed_vp = sum(building_printed_vp(pb.building) for pb in p.city_buildings)
        return int(p.vp_from_chips + printed_vp + _final_scoring_bonus(p))

    # -- legality ------------------------------------------------------------

    def _legal_actions_impl(self, player_id: int) -> list[EngineAction]:
        s = self._state
        n = s.num_players
        if player_id < 0 or player_id >= n:
            return []

        if s.phase is Phase.GAME_OVER:
            return []

        if s.phase is Phase.ROLE_SELECTION:
            if s.next_role_selector_index != player_id:
                return []
            return self._legal_pick_roles(player_id)

        if s.phase is Phase.ROUND_CLEANUP:
            return []

        # Role phases — only one active player depending on pending
        idx = s.current_role_execution_index or 0
        if idx < 0 or idx >= len(s.round_role_order):
            return []
        role, chooser = s.round_role_order[idx]

        if s.phase is Phase.SETTLER and isinstance(s.pending, SettlerPhasePending):
            return self._legal_settler(player_id, s.pending)

        if s.phase is Phase.MAYOR and isinstance(s.pending, MayorPhasePending):
            return self._legal_mayor(player_id, s.pending)

        if s.phase is Phase.BUILDER and isinstance(s.pending, BuilderPhasePending):
            return self._legal_builder(player_id, s.pending)

        if s.phase is Phase.CRAFTSMAN and isinstance(s.pending, CraftsmanPhasePending):
            return self._legal_craftsman(player_id, s.pending)

        if s.phase is Phase.TRADER and isinstance(s.pending, TraderPhasePending):
            return self._legal_trader(player_id, s.pending)

        if s.phase is Phase.CAPTAIN and isinstance(s.pending, CaptainPhasePending):
            return self._legal_captain(player_id, s.pending)

        if s.phase is Phase.PROSPECTOR:
            if player_id != chooser:
                return []
            return [ProspectorCollect()]

        return []

    def _legal_pick_roles(self, player_id: int) -> list[EngineAction]:
        s = self._state
        taken = {r for r in s.player_roles_this_round if r is not None}
        avail = [r for r in sorted(s.roles_in_play, key=lambda x: x.value) if r not in taken]
        return [PickRole(r) for r in avail]

    def _legal_settler(self, player_id: int, pend: SettlerPhasePending) -> list[EngineAction]:
        s = self._state
        chooser = pend.settler_role_chooser
        cur = pend.next_actor_index
        if cur is None or cur != player_id:
            return []
        p = s.players[player_id]
        acts: list[EngineAction] = []
        has_space = count_filled_island_spaces(p) < 12

        if (
            not pend.awaiting_normal_pick
            and has_space
            and _occupied_violet_building(p, Building.HACIENDA)
            and any(s.plantation_stacks)
        ):
            acts.append(SettlerTakeHacienda())

        if player_id == chooser:
            if s.quarries_remaining > 0 and has_space:
                acts.append(SettlerTakeQuarryPrivilege())
            for i, _t in enumerate(s.face_up_plantations):
                if has_space:
                    acts.append(SettlerTakeFaceUp(i))
        else:
            if _player_has_building(p, Building.CONSTRUCTION_HUT) and s.quarries_remaining > 0 and has_space:
                hut = next(pb for pb in p.city_buildings if pb.building == Building.CONSTRUCTION_HUT)
                if building_occupied(hut):
                    acts.append(SettlerTakeQuarryConstructionHut())
            for i, _t in enumerate(s.face_up_plantations):
                if has_space:
                    acts.append(SettlerTakeFaceUp(i))
        if not acts or not pend.awaiting_normal_pick:
            acts.append(SettlerPass())
        return acts

    def _legal_mayor(self, player_id: int, pend: MayorPhasePending) -> list[EngineAction]:
        return self._mayor_legal_actions_for_player(player_id, pend)

    @staticmethod
    def _mayor_board_capacity(player: PlayerState) -> int:
        island_capacity = sum(
            island_tile_max_colonists(space.tile) for space in player.island_spaces if space.tile is not None
        )
        building_capacity = sum(len(pb.colonists) for pb in player.city_buildings)
        return island_capacity + building_capacity

    def _validate_mayor_placement(
        self,
        player_id: int,
        action: EngineAction,
        pend: MayorPhasePending,
    ) -> Optional[str]:
        if pend.subphase != "placement" or pend.placement_next != player_id:
            return "not placement turn"
        if not isinstance(action, MayorSubmitPlacement):
            return "bad placement action"
        if player_id < 0 or player_id >= len(pend.placement_pools):
            return "invalid mayor placement pool index"
        pool = pend.placement_pools[player_id]
        p = self._state.players[player_id]
        if len(action.island_targets) != len(p.island_spaces):
            return "bad island target length"
        if len(action.building_targets) < len(p.city_buildings):
            return "bad building target length"
        if any(target != 0 for target in action.building_targets[len(p.city_buildings) :]):
            return "target for missing building"

        allocated_on_board = 0
        for idx, target in enumerate(action.island_targets):
            if target < 0:
                return "negative island target"
            space = p.island_spaces[idx]
            capacity = 0 if space.tile is None else island_tile_max_colonists(space.tile)
            if target > capacity:
                return "island target exceeds capacity"
            allocated_on_board += target

        for idx, pb in enumerate(p.city_buildings):
            target = action.building_targets[idx]
            if target < 0:
                return "negative building target"
            capacity = len(pb.colonists)
            if target > capacity:
                return "building target exceeds capacity"
            allocated_on_board += target

        if action.san_juan < 0:
            return "negative san juan target"
        if allocated_on_board + action.san_juan != pool:
            return "placement total mismatch"

        required_on_board = min(pool, self._mayor_board_capacity(p))
        if allocated_on_board != required_on_board:
            return "must fill board before San Juan"
        return None

    def _legal_storage_commits(self, player_id: int) -> list[EngineAction]:
        p = self._state.players[player_id]
        return [CaptainStorageCommit(keep_counts=normalize_goods_counts(k)) for k in _enumerate_valid_storage_keeps(p)]

    def _builder_legal_actions_for_player(self, player_id: int, pend: BuilderPhasePending) -> list[EngineAction]:
        s = self._state
        p = s.players[player_id]
        chooser = pend.role_chooser
        acts: list[EngineAction] = []
        quarry_ok = player_id == chooser
        for b in sorted(BUILDING_METADATA.keys(), key=lambda x: x.value):
            if dict(s.building_supply).get(b, 0) <= 0:
                continue
            if _player_has_building(p, b):
                continue
            for slot in range(12):
                if building_city_spaces(b) == 2:
                    if _repack_city_buildings(p.city_buildings, b, slot) is None:
                        continue
                elif not _can_place_building_at(p, b, slot):
                    continue
                cost = _builder_doubloon_cost_base(p, b, slot, quarry_discount_applies=quarry_ok)
                priv = 1 if player_id == chooser else 0
                if p.doubloons + priv >= cost:
                    acts.append(BuilderBuild(b, slot))
        if not acts:
            # Non-choosers MUST build if they can afford anything; they have no pass option.
            # Use BuilderNoOp to advance the pending actor.
            return [BuilderNoOp()] if player_id != chooser else [BuilderPass()]
        if player_id == chooser:
            return acts + [BuilderPass()]
        return acts

    def _legal_builder(self, player_id: int, pend: BuilderPhasePending) -> list[EngineAction]:
        if pend.next_actor != player_id:
            return []
        return self._builder_legal_actions_for_player(player_id, pend)

    def _craftsman_legal_actions_for_player(self, player_id: int, pend: CraftsmanPhasePending) -> list[EngineAction]:
        s = self._state
        p = s.players[player_id]
        chooser = pend.role_chooser
        prod = _compute_craftsman_production(p, s.goods_supply)
        sup_after_base = goods_dict(s.goods_supply)
        for g, n in prod.items():
            sup_after_base[g] = sup_after_base.get(g, 0) - n
        acts: list[EngineAction] = []
        prod_kinds = {g for g, n in prod.items() if n > 0}
        priv_opts: list[Optional[Good]] = [None]
        if player_id == chooser and prod_kinds:
            for g in sorted(prod_kinds, key=lambda x: x.value):
                if sup_after_base.get(g, 0) >= 1:
                    priv_opts.append(g)
        for pg in priv_opts:
            acts.append(CraftsmanTurn(privilege_good=pg, hacienda_good=None))
        out: list[EngineAction] = []
        seen: set[CraftsmanTurn] = set()
        for a in acts:
            if isinstance(a, CraftsmanTurn) and a not in seen:
                seen.add(a)
                out.append(a)
        return out if out else [CraftsmanTurn(privilege_good=None)]

    def _legal_craftsman(self, player_id: int, pend: CraftsmanPhasePending) -> list[EngineAction]:
        if pend.next_actor != player_id:
            return []
        return self._craftsman_legal_actions_for_player(player_id, pend)

    def _trader_legal_actions_for_player(self, player_id: int, pend: TraderPhasePending) -> list[EngineAction]:
        s = self._state
        p = s.players[player_id]
        house = s.trading_house
        acts: list[EngineAction] = [TraderPass()]
        if len(house.goods) >= 4:
            return acts
        gd = goods_dict(p.goods)
        for g in Good:
            if gd.get(g, 0) <= 0:
                continue
            if not _trading_house_allows_good(house, g, p):
                continue
            acts.append(TraderSell(g))
        return acts

    def _legal_trader(self, player_id: int, pend: TraderPhasePending) -> list[EngineAction]:
        if pend.next_actor != player_id:
            return []
        return self._trader_legal_actions_for_player(player_id, pend)

    def _captain_legal_actions_for_player(self, player_id: int, pend: CaptainPhasePending) -> list[EngineAction]:
        s = self._state
        if pend.subphase == "loading":
            if pend.active_player_index != player_id:
                return []
            p = s.players[player_id]
            loads = _legal_cargo_loads_for_player(s, player_id)
            acts: list[EngineAction] = []
            for g, si, _amt in loads:
                acts.append(CaptainLoad(good=g, ship_index=si))
            wharf_ok = _wharf_available(s, player_id, pend)
            if wharf_ok:
                for g in Good:
                    if good_count(p.goods, g) > 0:
                        acts.append(CaptainUseWharf(good=g))
            has_goods = goods_total(goods_dict(p.goods)) > 0
            can_cargo = _any_legal_cargo_load(s, player_id)
            # Captain: must load onto a cargo ship whenever legally able, unless you use Wharf for this turn instead.
            must_act = has_goods and (can_cargo or wharf_ok)
            if must_act:
                return acts
            return [CaptainPassLoading()]
        if pend.subphase == "storage":
            if pend.storage_next_actor != player_id:
                return []
            return self._legal_storage_commits(player_id)
        if pend.subphase == "unload_full_ships":
            return []
        return []

    def _legal_captain(self, player_id: int, pend: CaptainPhasePending) -> list[EngineAction]:
        return self._captain_legal_actions_for_player(player_id, pend)

    def _mayor_legal_actions_for_player(self, player_id: int, pend: MayorPhasePending) -> list[EngineAction]:
        s = self._state
        m: list[EngineAction] = []
        if pend.subphase == "privilege":
            if player_id != pend.mayor_role_chooser:
                return []
            if s.colonist_supply > 0:
                m.append(MayorPrivilegeTake())
            m.append(MayorPrivilegeSkip())
            return m
        if pend.subphase == "placement":
            return []
        return []

    # -- apply --------------------------------------------------------------

    def _apply_impl(self, player_id: int, action: EngineAction) -> Optional[str]:
        if not self.is_legal(player_id, action):
            return "illegal action"
        s = self._state

        if isinstance(action, PickRole):
            return self._apply_pick_role(player_id, action)

        if s.phase is Phase.SETTLER and isinstance(s.pending, SettlerPhasePending):
            return self._apply_settler(player_id, action, s.pending)

        if s.phase is Phase.MAYOR and isinstance(s.pending, MayorPhasePending):
            return self._apply_mayor(player_id, action, s.pending)

        if s.phase is Phase.BUILDER and isinstance(s.pending, BuilderPhasePending):
            return self._apply_builder(player_id, action, s.pending)

        if s.phase is Phase.CRAFTSMAN and isinstance(s.pending, CraftsmanPhasePending):
            return self._apply_craftsman(player_id, action, s.pending)

        if s.phase is Phase.TRADER and isinstance(s.pending, TraderPhasePending):
            return self._apply_trader(player_id, action, s.pending)

        if s.phase is Phase.CAPTAIN and isinstance(s.pending, CaptainPhasePending):
            return self._apply_captain(player_id, action, s.pending)

        if s.phase is Phase.PROSPECTOR and isinstance(action, ProspectorCollect):
            return self._apply_prospector(player_id)

        return "unhandled"

    def _apply_pick_role(self, player_id: int, action: PickRole) -> Optional[str]:
        s = self._state
        role = action.role
        taken = {r for r in s.player_roles_this_round if r is not None}
        if role in taken:
            return "role already taken"
        dm = _role_doubloons_map(s)
        coins = dm.get(role, 0)
        new_dm = dict(dm)
        if role in new_dm:
            del new_dm[role]
        s1 = _set_role_doubloons(s, new_dm)
        pl = list(s1.player_roles_this_round)
        pl[player_id] = role
        order = list(s1.round_role_order)
        order.append((role, player_id))
        s2 = dataclasses.replace(
            s1,
            player_roles_this_round=tuple(pl),
            round_role_order=tuple(order),
            next_role_selector_index=(player_id + 1) % s.num_players
            if len(order) < s.num_players
            else None,
        )
        s3 = _bank_receive(s2, player_id, coins) if coins else s2
        if len(order) < s.num_players:
            self._state = s3
            return None
        self._state = self._begin_role_execution(s3, 0)
        return None

    def _begin_role_execution(self, state: GameState, exec_index: int) -> GameState:
        role, chooser = state.round_role_order[exec_index]
        phase = _role_to_execution_phase(role)
        if phase is Phase.PROSPECTOR:
            return dataclasses.replace(
                state,
                phase=Phase.PROSPECTOR,
                current_role_execution_index=exec_index,
                pending=None,
            )
        if phase is Phase.SETTLER:
            pend = SettlerPhasePending(
                settler_role_chooser=chooser,
                next_actor_index=chooser,
                awaiting_normal_pick=False,
            )
            return dataclasses.replace(
                state,
                phase=Phase.SETTLER,
                current_role_execution_index=exec_index,
                pending=pend,
            )
        if phase is Phase.MAYOR:
            mp = MayorPhasePending(
                mayor_role_chooser=chooser,
                subphase="privilege",
                placement_pools=tuple(0 for _ in range(state.num_players)),
                placement_next=None,
            )
            return dataclasses.replace(
                state,
                phase=Phase.MAYOR,
                current_role_execution_index=exec_index,
                pending=mp,
            )
        if phase is Phase.BUILDER:
            bp = BuilderPhasePending(role_chooser=chooser, next_actor=chooser)
            return dataclasses.replace(
                state,
                phase=Phase.BUILDER,
                current_role_execution_index=exec_index,
                pending=bp,
            )
        if phase is Phase.CRAFTSMAN:
            cp = CraftsmanPhasePending(role_chooser=chooser, next_actor=chooser)
            return dataclasses.replace(
                state,
                phase=Phase.CRAFTSMAN,
                current_role_execution_index=exec_index,
                pending=cp,
            )
        if phase is Phase.TRADER:
            tp = TraderPhasePending(role_chooser=chooser, next_actor=chooser)
            return dataclasses.replace(
                state,
                phase=Phase.TRADER,
                current_role_execution_index=exec_index,
                pending=tp,
            )
        if phase is Phase.CAPTAIN:
            cap = CaptainPhasePending(
                captain_role_chooser=chooser,
                active_player_index=chooser,
                captain_privilege_vp_awarded=False,
                wharf_used=tuple(False for _ in range(state.num_players)),
                subphase="loading",
                storage_next_actor=None,
                storage_done=tuple(False for _ in range(state.num_players)),
                ship_full_credit=(None, None, None),
            )
            return dataclasses.replace(
                state,
                phase=Phase.CAPTAIN,
                current_role_execution_index=exec_index,
                pending=cap,
            )
        raise AssertionError(f"unknown role phase {phase}")

    def _advance_role_queue(self, state: GameState) -> GameState:
        idx = (state.current_role_execution_index or 0) + 1
        if idx >= state.num_players:
            return self._begin_round_cleanup(state)
        return self._begin_role_execution(state, idx)

    def _begin_round_cleanup(self, state: GameState) -> GameState:
        # Place +1 doubloon on each unused role card (stacks).
        chosen = {r for r in state.player_roles_this_round if r is not None}
        dm = _role_doubloons_map(state)
        for r in state.roles_in_play:
            if r not in chosen:
                dm[r] = dm.get(r, 0) + 1
        s1 = _set_role_doubloons(state, dm)
        # Return role cards
        s2 = dataclasses.replace(
            s1,
            phase=Phase.ROUND_CLEANUP,
            player_roles_this_round=tuple(None for _ in range(s1.num_players)),
            round_role_order=(),
            current_role_execution_index=None,
            next_role_selector_index=None,
            pending=None,
        )
        # Governor passes clockwise
        gov = (s2.governor_index + 1) % s2.num_players
        game_over = s2.game_end_colonists or s2.game_end_city12 or s2.game_end_vp
        s3 = dataclasses.replace(
            s2,
            governor_index=gov,
            next_role_selector_index=gov,
            round_number=s2.round_number + 1,
            phase=Phase.GAME_OVER if game_over else Phase.ROLE_SELECTION,
        )
        self._state = s3
        return s3

    def _apply_settler(self, player_id: int, action: EngineAction, pend: SettlerPhasePending) -> Optional[str]:
        s = self._state
        if isinstance(action, SettlerPass):
            if pend.awaiting_normal_pick and self._legal_settler(player_id, pend) != [SettlerPass()]:
                return "must complete normal settler pick after hacienda"
            self._settler_finish_turn(s, pend, player_id)
            return None
        p = s.players[player_id]
        empty_slot = next((i for i, sp in enumerate(p.island_spaces) if sp.tile is None), None)
        if empty_slot is None:
            return "no empty island slot"
        if isinstance(action, SettlerTakeHacienda):
            if pend.awaiting_normal_pick:
                return "hacienda already used this turn"
            if not _occupied_violet_building(p, Building.HACIENDA):
                return "no hacienda"
            s2, tile = _draw_hacienda_plantation(s)
            if tile is None:
                return "no plantations for hacienda"
            new_pl = list(p.island_spaces)
            new_pl[empty_slot] = IslandSpace(tile=tile, colonists=0)
            s3 = _replace_player(s2, player_id, dataclasses.replace(p, island_spaces=tuple(new_pl)))
            if count_filled_island_spaces(s3.players[player_id]) >= 12:
                self._settler_finish_turn(s3, dataclasses.replace(pend, awaiting_normal_pick=False), player_id)
                return None
            self._state = dataclasses.replace(
                s3,
                pending=dataclasses.replace(pend, awaiting_normal_pick=True),
            )
            return None
        if isinstance(action, SettlerTakeFaceUp):
            if action.face_up_index < 0 or action.face_up_index >= len(s.face_up_plantations):
                return "bad face-up index"
            fu = list(s.face_up_plantations)
            t = fu.pop(action.face_up_index)
            new_pl = list(p.island_spaces)
            new_pl[empty_slot] = IslandSpace(tile=t, colonists=0)
            s2 = _replace_player(s, player_id, dataclasses.replace(p, island_spaces=tuple(new_pl)))
            s3 = dataclasses.replace(s2, face_up_plantations=tuple(fu))
            s4 = _maybe_apply_hospice(s3, player_id, empty_slot)
            self._settler_finish_turn(s4, dataclasses.replace(pend, awaiting_normal_pick=False), player_id)
            return None
        if isinstance(action, (SettlerTakeQuarryPrivilege, SettlerTakeQuarryConstructionHut)):
            if s.quarries_remaining <= 0:
                return "no quarries"
            new_pl = list(p.island_spaces)
            new_pl[empty_slot] = IslandSpace(tile=IslandTile.QUARRY, colonists=0)
            s2 = _replace_player(
                s,
                player_id,
                dataclasses.replace(p, island_spaces=tuple(new_pl)),
            )
            s3 = dataclasses.replace(s2, quarries_remaining=s2.quarries_remaining - 1)
            s4 = _maybe_apply_hospice(s3, player_id, empty_slot)
            self._settler_finish_turn(s4, dataclasses.replace(pend, awaiting_normal_pick=False), player_id)
            return None
        return "bad settler action"

    def _settler_finish_turn(self, state: GameState, pend: SettlerPhasePending, player_id: int) -> None:
        """Chooser acts first, then clockwise; after the last player in order, the phase ends."""
        order = role_action_order(pend.settler_role_chooser, state.num_players)
        idx = order.index(player_id)
        if idx == len(order) - 1:
            s2 = refresh_face_up_plantations_after_settler(state)
            s3 = dataclasses.replace(s2, pending=None)
            # _advance_role_queue returns the next-state; assign it
            self._state = self._advance_role_queue(s3)
            return
        nxt = order[idx + 1]
        self._state = dataclasses.replace(
            state,
            pending=SettlerPhasePending(pend.settler_role_chooser, nxt, False),
        )

    def _builder_finish_turn(self, state: GameState, pend: BuilderPhasePending, player_id: int) -> None:
        order = role_action_order(pend.role_chooser, state.num_players)
        idx = order.index(player_id)
        if idx == len(order) - 1:
            self._state = self._advance_role_queue(dataclasses.replace(state, pending=None))
            return
        nxt = order[idx + 1]
        self._state = dataclasses.replace(
            state,
            pending=BuilderPhasePending(role_chooser=pend.role_chooser, next_actor=nxt),
        )

    def _apply_builder(self, player_id: int, action: EngineAction, pend: BuilderPhasePending) -> Optional[str]:
        s = self._state
        chooser = pend.role_chooser
        if player_id != pend.next_actor:
            return "not builder turn"
        if isinstance(action, BuilderPass):
            if player_id != chooser:
                return "only chooser may pass builder"
            # Builder may always pass (they just forgo the building opportunity).
            # Non-choosers who can afford must build (BuilderNoOp handles this).
            self._builder_finish_turn(s, pend, player_id)
            return None
        if isinstance(action, BuilderNoOp):
            # Non-chooser has no affordable building; just advance to next player.
            self._builder_finish_turn(s, pend, player_id)
            return None
        if not isinstance(action, BuilderBuild):
            return "bad builder action"
        p = s.players[player_id]
        quarry_ok = player_id == chooser
        if _player_has_building(p, action.building):
            return "already own building"
        if building_city_spaces(action.building) == 2:
            new_city = _repack_city_buildings(p.city_buildings, action.building, action.anchor_slot)
            if new_city is None:
                return "illegal placement"
        else:
            if not _can_place_building_at(p, action.building, action.anchor_slot):
                return "illegal placement"
            w = building_worker_circles(action.building)
            new_city = tuple(
                list(p.city_buildings)
                + [
                    PlacedBuilding(
                        building=action.building,
                        anchor_slot=action.anchor_slot,
                        colonists=tuple(0 for _ in range(w)),
                    )
                ]
            )
        cost = _builder_doubloon_cost_base(p, action.building, action.anchor_slot, quarry_discount_applies=quarry_ok)
        priv = 1 if player_id == chooser else 0
        if p.doubloons + priv < cost:
            return "cannot afford"
        s1 = _building_supply_take(s, action.building)
        if s1 is None:
            return "building not in supply"
        s2 = _bank_receive(s1, player_id, 1) if player_id == chooser else s1
        s3, err = _bank_pay(s2, player_id, cost)
        if err:
            return err
        p3 = s3.players[player_id]
        s4 = _replace_player(s3, player_id, dataclasses.replace(p3, city_buildings=new_city))
        s5 = _maybe_apply_university(s4, player_id, action.building)
        if _count_filled_city_slots(s5.players[player_id]) >= 12:
            s5 = dataclasses.replace(s5, game_end_city12=True)
        self._builder_finish_turn(s5, pend, player_id)
        return None

    @staticmethod
    def _mayor_next_player_with_pool(mayor: int, num_players: int, pools: tuple[int, ...]) -> Optional[int]:
        for k in range(num_players):
            idx = (mayor + k) % num_players
            if idx < 0 or idx >= len(pools):
                continue
            if pools[idx] > 0:
                return idx
        return None

    @staticmethod
    def _mayor_next_player_after(num_players: int, pools: list[int], after_player: int) -> Optional[int]:
        for step in range(1, num_players + 1):
            idx = (after_player + step) % num_players
            if idx < 0 or idx >= len(pools):
                continue
            if pools[idx] > 0:
                return idx
        return None

    def _mayor_refill_ship_and_advance(self, state: GameState, pend: MayorPhasePending) -> None:
        need = max(sum(count_empty_building_circles(p) for p in state.players), state.num_players)
        take = min(need, state.colonist_supply)
        ge = take < need
        s1 = dataclasses.replace(
            state,
            colonist_ship=take,
            colonist_supply=state.colonist_supply - take,
            game_end_colonists=state.game_end_colonists or ge,
            pending=None,
        )
        self._state = self._advance_role_queue(s1)

    def _mayor_after_privilege(self, s: GameState, pend: MayorPhasePending) -> None:
        s2, pend2, placement_next = _prepare_mayor_placement_state(s, pend)
        if placement_next is None:
            self._mayor_refill_ship_and_advance(s2, pend2)
        else:
            self._state = dataclasses.replace(
                s2,
                pending=dataclasses.replace(
                    pend2,
                    placement_next=placement_next,
                ),
            )

    def _apply_mayor(self, player_id: int, action: EngineAction, pend: MayorPhasePending) -> Optional[str]:
        s = self._state
        mayor = pend.mayor_role_chooser
        if pend.subphase == "privilege":
            if player_id != mayor:
                return "mayor privilege only"
            if isinstance(action, MayorPrivilegeSkip):
                self._mayor_after_privilege(s, pend)
                return None
            if isinstance(action, MayorPrivilegeTake):
                if s.colonist_supply <= 0:
                    return "empty colonist supply"
                pools = list(pend.placement_pools[: s.num_players])
                if len(pools) < s.num_players:
                    pools.extend(0 for _ in range(s.num_players - len(pools)))
                pools[mayor] += 1
                pend2 = dataclasses.replace(pend, placement_pools=tuple(pools))
                s2 = dataclasses.replace(s, colonist_supply=s.colonist_supply - 1, pending=pend2)
                self._mayor_after_privilege(s2, pend2)
                return None
            return "bad mayor privilege action"
        if pend.subphase == "placement":
            err = self._validate_mayor_placement(player_id, action, pend)
            if err:
                return err
            p = s.players[player_id]
            assert isinstance(action, MayorSubmitPlacement)

            new_island = [
                IslandSpace(tile=space.tile, colonists=action.island_targets[idx])
                for idx, space in enumerate(p.island_spaces)
            ]
            new_city = []
            for idx, pb in enumerate(p.city_buildings):
                workers = action.building_targets[idx]
                colonists = tuple(1 if circle < workers else 0 for circle in range(len(pb.colonists)))
                new_city.append(dataclasses.replace(pb, colonists=colonists))
            updated_player = dataclasses.replace(
                p,
                san_juan_colonists=action.san_juan,
                island_spaces=tuple(new_island),
                city_buildings=tuple(new_city),
            )
            s2 = _replace_player(s, player_id, updated_player)

            pools = list(pend.placement_pools)
            pools[player_id] = 0
            if sum(pools) == 0:
                self._mayor_refill_ship_and_advance(s2, dataclasses.replace(pend, placement_pools=tuple(pools)))
                return None
            nxt = self._mayor_next_player_after(s.num_players, pools, player_id)
            if nxt is None:
                self._mayor_refill_ship_and_advance(
                    s2,
                    dataclasses.replace(pend, placement_pools=tuple(pools)),
                )
                return None
            self._state = dataclasses.replace(
                s2,
                pending=dataclasses.replace(
                    pend,
                    placement_pools=tuple(pools),
                    placement_next=nxt,
                ),
            )
            return None
        return "bad mayor subphase"

    def _craftsman_finish_turn(self, state: GameState, pend: CraftsmanPhasePending, player_id: int) -> None:
        order = role_action_order(pend.role_chooser, state.num_players)
        idx = order.index(player_id)
        if idx == len(order) - 1:
            self._state = self._advance_role_queue(dataclasses.replace(state, pending=None))
            return
        nxt = order[idx + 1]
        self._state = dataclasses.replace(
            state,
            pending=CraftsmanPhasePending(role_chooser=pend.role_chooser, next_actor=nxt),
        )

    def _apply_craftsman(self, player_id: int, action: EngineAction, pend: CraftsmanPhasePending) -> Optional[str]:
        if player_id != pend.next_actor:
            return "not craftsman turn"
        if not isinstance(action, CraftsmanTurn):
            return "bad craftsman action"
        s = self._state
        p = s.players[player_id]
        chooser = pend.role_chooser
        prod = _compute_craftsman_production(p, s.goods_supply)
        state = s
        for g, n in prod.items():
            for _ in range(n):
                state, err = _goods_supply_take(state, g, 1)
                if err:
                    return err
                state = _player_add_goods(state, player_id, g, 1)
        if action.hacienda_good is not None:
            return "hacienda is a settler-only ability"
        if action.privilege_good is not None:
            if player_id != chooser:
                return "privilege only for chooser"
            kinds = set(prod.keys())
            if action.privilege_good not in kinds:
                return "bad privilege good"
            state, err = _goods_supply_take(state, action.privilege_good, 1)
            if err:
                return err
            state = _player_add_goods(state, player_id, action.privilege_good, 1)
        if _occupied_violet_building(state.players[player_id], Building.FACTORY):
            state = _bank_receive(state, player_id, _factory_bonus(len(prod)))
        self._craftsman_finish_turn(state, pend, player_id)
        return None

    def _trader_finish_turn(self, state: GameState, pend: TraderPhasePending, player_id: int) -> None:
        order = role_action_order(pend.role_chooser, state.num_players)
        idx = order.index(player_id)
        if idx == len(order) - 1:
            house = state.trading_house
            s2 = dataclasses.replace(
                state,
                trading_house=TradingHouseState(goods=()) if len(house.goods) >= 4 else house,
                pending=None,
            )
            self._state = self._advance_role_queue(s2)
            return
        nxt = order[idx + 1]
        self._state = dataclasses.replace(
            state,
            pending=TraderPhasePending(role_chooser=pend.role_chooser, next_actor=nxt),
        )

    def _apply_trader(self, player_id: int, action: EngineAction, pend: TraderPhasePending) -> Optional[str]:
        s = self._state
        if player_id != pend.next_actor:
            return "not trader turn"
        if isinstance(action, TraderPass):
            self._trader_finish_turn(s, pend, player_id)
            return None
        if not isinstance(action, TraderSell):
            return "bad trader action"
        p = s.players[player_id]
        g = action.good
        if good_count(p.goods, g) <= 0:
            return "no good to sell"
        house = s.trading_house
        if len(house.goods) >= 4:
            return "trading house full"
        if not _trading_house_allows_good(house, g, p):
            return "office required for duplicate"
        price = _trader_sell_price(p, g)
        if player_id == pend.role_chooser:
            price += 1
        s2, err = _player_sub_goods(s, player_id, g, 1)
        if err:
            return err
        s3 = _bank_receive(s2, player_id, price)
        new_goods = tuple(house.goods) + (g,)
        s4 = dataclasses.replace(s3, trading_house=TradingHouseState(goods=new_goods))
        if len(new_goods) >= 4:
            s5 = dataclasses.replace(s4, trading_house=TradingHouseState(goods=()), pending=None)
            self._state = self._advance_role_queue(s5)
            return None
        self._trader_finish_turn(s4, pend, player_id)
        return None

    def _captain_enter_storage(self, state: GameState, pend: CaptainPhasePending) -> GameState:
        return dataclasses.replace(
            state,
            pending=CaptainPhasePending(
                captain_role_chooser=pend.captain_role_chooser,
                active_player_index=pend.active_player_index,
                captain_privilege_vp_awarded=pend.captain_privilege_vp_awarded,
                wharf_used=pend.wharf_used,
                subphase="storage",
                storage_next_actor=pend.captain_role_chooser,
                storage_done=tuple(False for _ in range(state.num_players)),
                ship_full_credit=pend.ship_full_credit,
            ),
        )

    def _captain_after_loading_action(self, state: GameState, pend: CaptainPhasePending) -> GameState:
        if _global_anyone_can_load(state, pend):
            nxt = (pend.active_player_index + 1) % state.num_players
            return dataclasses.replace(
                state,
                pending=dataclasses.replace(pend, active_player_index=nxt),
            )
        return self._captain_enter_storage(state, pend)

    def _captain_unload_full_ships(self, state: GameState, pend: CaptainPhasePending) -> GameState:
        ships = list(state.cargo_ships)
        st = state
        for i, sh in enumerate(ships):
            if sh.barrels < sh.capacity or sh.good is None:
                continue
            g = sh.good
            st = _goods_supply_add(st, g, sh.barrels)
            ships[i] = CargoShipState(capacity=sh.capacity, good=None, barrels=0)
        return dataclasses.replace(st, cargo_ships=tuple(ships))

    def _apply_captain(self, player_id: int, action: EngineAction, pend: CaptainPhasePending) -> Optional[str]:
        s = self._state
        if pend.subphase == "loading":
            if pend.active_player_index != player_id:
                return "not captain loading turn"
            if isinstance(action, CaptainPassLoading):
                if _any_legal_cargo_load(s, player_id) or (
                    _wharf_available(s, player_id, pend) and goods_total(goods_dict(s.players[player_id].goods)) > 0
                ):
                    return "cannot pass with legal load or wharf"
                self._state = self._captain_after_loading_action(s, pend)
                return None
            if isinstance(action, CaptainLoad):
                p = s.players[player_id]
                gd = goods_dict(p.goods)
                amt = _captain_max_load_on_ship(s.cargo_ships, action.ship_index, action.good, gd)
                if amt <= 0:
                    return "illegal load"
                s2, err = _player_sub_goods(s, player_id, action.good, amt)
                if err:
                    return err
                ships = list(s2.cargo_ships)
                sh = ships[action.ship_index]
                new_good = action.good if sh.good is None else sh.good
                new_barrels = sh.barrels + amt
                new_sh = CargoShipState(capacity=sh.capacity, good=new_good, barrels=new_barrels)
                ships[action.ship_index] = new_sh
                credit = list(pend.ship_full_credit)
                if new_barrels == sh.capacity and new_good is not None:
                    if action.ship_index < 0 or action.ship_index >= len(credit):
                        return "invalid captain ship_full_credit index"
                    credit[action.ship_index] = player_id
                pend2 = dataclasses.replace(pend, ship_full_credit=tuple(credit))
                s3 = dataclasses.replace(s2, cargo_ships=tuple(ships), pending=pend2)
                s3, errv = _award_vp_from_supply(s3, player_id, amt)
                if errv:
                    return errv
                if _harbor_bonus(s3.players[player_id]):
                    s3, errh = _award_vp_from_supply(s3, player_id, 1)
                    if errh:
                        return errh
                if player_id == pend.captain_role_chooser and not pend.captain_privilege_vp_awarded:
                    s3, errp = _award_vp_from_supply(s3, player_id, 1)
                    if errp:
                        return errp
                    pnd = s3.pending
                    if isinstance(pnd, CaptainPhasePending):
                        s3 = dataclasses.replace(
                            s3,
                            pending=dataclasses.replace(pnd, captain_privilege_vp_awarded=True),
                        )
                pnd2 = s3.pending
                if not isinstance(pnd2, CaptainPhasePending):
                    return "captain pending lost"
                self._state = self._captain_after_loading_action(s3, pnd2)
                return None
            if isinstance(action, CaptainUseWharf):
                wharf_already_used = (
                    pend.wharf_used[player_id] if 0 <= player_id < len(pend.wharf_used) else False
                )
                if wharf_already_used:
                    return "wharf already used"
                if not _wharf_available(s, player_id, pend):
                    return "no wharf"
                g = action.good
                n = good_count(s.players[player_id].goods, g)
                if n <= 0:
                    return "no goods for wharf"
                s2 = s
                for _ in range(n):
                    s2, err = _player_sub_goods(s2, player_id, g, 1)
                    if err:
                        return err
                    s2 = _goods_supply_add(s2, g, 1)
                s2, errv = _award_vp_from_supply(s2, player_id, n)
                if errv:
                    return errv
                wu = list(pend.wharf_used)
                if player_id < 0 or player_id >= len(wu):
                    return "invalid captain wharf_used index"
                wu[player_id] = True
                pend2 = dataclasses.replace(pend, wharf_used=tuple(wu))
                s3 = dataclasses.replace(s2, pending=pend2)
                if _harbor_bonus(s3.players[player_id]):
                    s3, errh = _award_vp_from_supply(s3, player_id, 1)
                    if errh:
                        return errh
                pnd3 = s3.pending
                if not isinstance(pnd3, CaptainPhasePending):
                    return "captain pending lost"
                self._state = self._captain_after_loading_action(s3, pnd3)
                return None
            return "bad captain loading action"
        if pend.subphase == "storage":
            if pend.storage_next_actor != player_id:
                return "not storage turn"
            if not isinstance(action, CaptainStorageCommit):
                return "bad storage action"
            p = s.players[player_id]
            have = goods_dict(p.goods)
            keep = goods_dict(action.keep_counts)
            if not _storage_keep_is_valid(p, keep, have):
                return "illegal storage"
            back: dict[Good, int] = {}
            for gg in Good:
                diff = have.get(gg, 0) - keep.get(gg, 0)
                if diff > 0:
                    back[gg] = diff
            s2 = s
            for gg, nn in back.items():
                s2, err = _player_sub_goods(s2, player_id, gg, nn)
                if err:
                    return err
                s2 = _goods_supply_add(s2, gg, nn)
            new_keep = normalize_goods_counts(keep)
            s2 = _replace_player(s2, player_id, dataclasses.replace(s2.players[player_id], goods=new_keep))
            done = list(pend.storage_done)
            if player_id < 0 or player_id >= len(done):
                return "invalid captain storage_done index"
            done[player_id] = True
            pend2 = dataclasses.replace(pend, storage_done=tuple(done))
            if all(pend2.storage_done):
                s3 = self._captain_unload_full_ships(s2, pend2)
                self._state = self._advance_role_queue(dataclasses.replace(s3, pending=None))
                return None
            nxt = (player_id + 1) % s2.num_players
            for step in range(s2.num_players):
                idx = (nxt + step) % s2.num_players
                if idx < 0 or idx >= len(pend2.storage_done):
                    continue
                if not pend2.storage_done[idx]:
                    pend3 = dataclasses.replace(pend2, storage_next_actor=idx)
                    self._state = dataclasses.replace(s2, pending=pend3)
                    return None
            return "storage advance failed"
        return "bad captain subphase"

    def _apply_prospector(self, player_id: int) -> Optional[str]:
        s = self._state
        idx = s.current_role_execution_index or 0
        if idx < 0 or idx >= len(s.round_role_order):
            return "invalid role execution index"
        _role, chooser = s.round_role_order[idx]
        if player_id != chooser:
            return "not prospector chooser"
        s1 = _bank_receive(s, player_id, 1)
        self._state = self._advance_role_queue(s1)
        return None
