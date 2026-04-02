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
class SettlerPass:
    """Skip settler action on your turn (optional action for non-captain roles)."""


@dataclass(frozen=True, slots=True)
class MayorPrivilegeTake:
    """Mayor privilege: take one colonist from supply before the ship is drafted."""


@dataclass(frozen=True, slots=True)
class MayorPrivilegeSkip:
    """Mayor may skip privilege (still legal if supply empty — handled by legality)."""


@dataclass(frozen=True, slots=True)
class MayorDraftTake:
    """Take the next colonist from the ship into your hand (only on your draft turn)."""


@dataclass(frozen=True, slots=True)
class MayorPlaceColonistIsland:
    """Place one colonist from your hand onto an island tile (next empty circle on that tile)."""

    island_slot: int


@dataclass(frozen=True, slots=True)
class MayorPlaceColonistBuilding:
    """Place one colonist from your hand onto an empty worker circle of a building."""

    building_index: int
    circle_index: int


@dataclass(frozen=True, slots=True)
class MayorPlaceColonistSanJuan:
    """Place one colonist from your hand into San Juan (only when no empty circles remain)."""


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
    SettlerTakeFaceUp,
    SettlerTakeQuarryPrivilege,
    SettlerTakeQuarryConstructionHut,
    SettlerPass,
    MayorPrivilegeTake,
    MayorPrivilegeSkip,
    MayorDraftTake,
    MayorPlaceColonistIsland,
    MayorPlaceColonistBuilding,
    MayorPlaceColonistSanJuan,
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


def _builder_discount(player: PlayerState, anchor_slot: int, building: Building) -> int:
    """Quarry discount capped by building column (1–4)."""
    col_cap = max_quarry_discount_for_city_slot(anchor_slot)
    quarries = count_occupied_quarries(player)
    return min(quarries, col_cap)


def _player_has_building(player: PlayerState, b: Building) -> bool:
    return any(pb.building == b for pb in player.city_buildings)


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
            if not _can_place_building_at(p, b, slot):
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
        return _player_has_building(player, Building.OFFICE)
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
    return _player_has_building(player, Building.HARBOR) and building_occupied(
        next(pb for pb in player.city_buildings if pb.building == Building.HARBOR)
    )


def _wharf_available(state: GameState, player_index: int, pending: CaptainPhasePending) -> bool:
    if player_index < 0 or player_index >= len(pending.wharf_used):
        wharf_already_used = False
    else:
        wharf_already_used = pending.wharf_used[player_index]
    if wharf_already_used:
        return False
    if not _player_has_building(state.players[player_index], Building.WHARF):
        return False
    pb = next(pb for pb in state.players[player_index].city_buildings if pb.building == Building.WHARF)
    return building_occupied(pb)


def _occupied_violet_building(player: PlayerState, b: Building) -> bool:
    if not _player_has_building(player, b):
        return False
    pb = next(x for x in player.city_buildings if x.building == b)
    return building_occupied(pb)


def _trader_sell_price(player: PlayerState, good: Good) -> int:
    price = _TRADER_PRICE[good]
    if _occupied_violet_building(player, Building.SMALL_MARKET):
        price += 1
    if _occupied_violet_building(player, Building.OFFICE):
        price += 1
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
        return action in self.legal_actions(player_id)

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
        return int(p.vp_from_chips + p.vp_on_paper)

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
        acts: list[EngineAction] = [SettlerPass()]

        if player_id == chooser:
            if s.quarries_remaining > 0 and count_filled_island_spaces(p) < 12:
                acts.append(SettlerTakeQuarryPrivilege())
            for i, _t in enumerate(s.face_up_plantations):
                if count_filled_island_spaces(p) < 12:
                    acts.append(SettlerTakeFaceUp(i))
        else:
            if _player_has_building(p, Building.CONSTRUCTION_HUT) and s.quarries_remaining > 0 and count_filled_island_spaces(p) < 12:
                hut = next(pb for pb in p.city_buildings if pb.building == Building.CONSTRUCTION_HUT)
                if building_occupied(hut):
                    acts.append(SettlerTakeQuarryConstructionHut())
            for i, _t in enumerate(s.face_up_plantations):
                if count_filled_island_spaces(p) < 12:
                    acts.append(SettlerTakeFaceUp(i))
        return acts

    def _legal_mayor(self, player_id: int, pend: MayorPhasePending) -> list[EngineAction]:
        return self._mayor_legal_actions_for_player(player_id, pend)

    def _legal_mayor_placements(self, player_id: int, hand: int) -> list[EngineAction]:
        s = self._state
        p = s.players[player_id]
        acts: list[EngineAction] = []
        # No San Juan dumping if any empty circle exists anywhere on your board (must fill if possible).
        has_empty = self._any_empty_circle(p)
        if has_empty:
            for si, sp in enumerate(p.island_spaces):
                if sp.tile is None:
                    continue
                if sp.colonists < island_tile_max_colonists(sp.tile):
                    acts.append(MayorPlaceColonistIsland(si))
            for bi, pb in enumerate(p.city_buildings):
                for ci, cc in enumerate(pb.colonists):
                    if cc == 0:
                        acts.append(MayorPlaceColonistBuilding(bi, ci))
        elif hand > 0:
            acts.append(MayorPlaceColonistSanJuan())
        return acts

    def _any_empty_circle(self, p: PlayerState) -> bool:
        for sp in p.island_spaces:
            if sp.tile is None:
                continue
            if sp.colonists < island_tile_max_colonists(sp.tile):
                return True
        for pb in p.city_buildings:
            for c in pb.colonists:
                if c == 0:
                    return True
        return False

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
                if not _can_place_building_at(p, b, slot):
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
        has_hacienda = _occupied_violet_building(p, Building.HACIENDA)
        prod = _compute_craftsman_production(p, s.goods_supply)
        sup = goods_dict(s.goods_supply)
        sup_after_base = dict(sup)
        for g, n in prod.items():
            sup_after_base[g] = sup_after_base.get(g, 0) - n
        hacienda_options: list[Optional[Good]] = [None]
        if has_hacienda and prod:
            for g in prod:
                if prod.get(g, 0) > 0 and sup_after_base.get(g, 0) >= 1:
                    hacienda_options.append(g)
        acts: list[EngineAction] = []
        for hg in hacienda_options:
            sup2 = dict(sup_after_base)
            if hg is not None:
                sup2[hg] = sup2.get(hg, 0) - 1
                if sup2.get(hg, 0) < 0:
                    continue
            prod_kinds = {g for g, n in prod.items() if n > 0}
            if hg is not None:
                prod_kinds.add(hg)
            priv_opts: list[Optional[Good]] = [None]
            if player_id == chooser and prod_kinds:
                for g in sorted(prod_kinds, key=lambda x: x.value):
                    if sup2.get(g, 0) >= 1:
                        priv_opts.append(g)
            for pg in priv_opts:
                acts.append(CraftsmanTurn(privilege_good=pg, hacienda_good=hg))
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
        if pend.subphase == "draft":
            k = pend.ship_size_at_start - s.colonist_ship
            nxt = (pend.mayor_role_chooser + k) % s.num_players
            if player_id != nxt or s.colonist_ship <= 0:
                return []
            return [MayorDraftTake()]
        if pend.subphase == "placement":
            if pend.placement_next is None or player_id != pend.placement_next:
                return []
            if player_id < 0 or player_id >= len(pend.colonists_hands):
                return []
            hand = pend.colonists_hands[player_id]
            if hand <= 0:
                return []
            return self._legal_mayor_placements(player_id, hand)
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
            pend = SettlerPhasePending(settler_role_chooser=chooser, next_actor_index=chooser)
            return dataclasses.replace(
                state,
                phase=Phase.SETTLER,
                current_role_execution_index=exec_index,
                pending=pend,
            )
        if phase is Phase.MAYOR:
            ship = state.colonist_ship
            mp = MayorPhasePending(
                mayor_role_chooser=chooser,
                ship_size_at_start=ship,
                colonists_from_ship_remaining=ship,
                colonists_hands=tuple(0 for _ in range(state.num_players)),
                subphase="privilege",
                privilege_done=False,
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
        order = role_action_order(pend.settler_role_chooser, s.num_players)
        if isinstance(action, SettlerPass):
            self._settler_finish_turn(s, pend, player_id)
            return None
        p = s.players[player_id]
        empty_slot = next((i for i, sp in enumerate(p.island_spaces) if sp.tile is None), None)
        if empty_slot is None:
            return "no empty island slot"
        if isinstance(action, SettlerTakeFaceUp):
            if action.face_up_index < 0 or action.face_up_index >= len(s.face_up_plantations):
                return "bad face-up index"
            fu = list(s.face_up_plantations)
            t = fu.pop(action.face_up_index)
            new_pl = list(p.island_spaces)
            new_pl[empty_slot] = IslandSpace(tile=t, colonists=0)
            s2 = _replace_player(s, player_id, dataclasses.replace(p, island_spaces=tuple(new_pl)))
            s3 = dataclasses.replace(s2, face_up_plantations=tuple(fu))
            self._settler_finish_turn(s3, pend, player_id)
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
            self._settler_finish_turn(s3, pend, player_id)
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
            pending=SettlerPhasePending(pend.settler_role_chooser, nxt),
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
        if not _can_place_building_at(p, action.building, action.anchor_slot):
            return "illegal placement"
        if _player_has_building(p, action.building):
            return "already own building"
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
        w = building_worker_circles(action.building)
        pb = PlacedBuilding(
            building=action.building,
            anchor_slot=action.anchor_slot,
            colonists=tuple(0 for _ in range(w)),
        )
        p3 = s3.players[player_id]
        new_city = tuple(list(p3.city_buildings) + [pb])
        s4 = _replace_player(s3, player_id, dataclasses.replace(p3, city_buildings=new_city))
        printed_vp = building_printed_vp(action.building)
        s5, err2 = _pay_building_vp_from_supply(s4, player_id, printed_vp)
        if err2:
            return err2
        if _count_filled_city_slots(s5.players[player_id]) >= 12:
            s5 = dataclasses.replace(s5, game_end_city12=True)
        self._builder_finish_turn(s5, pend, player_id)
        return None

    @staticmethod
    def _mayor_next_with_colonists_in_hand(mayor: int, num_players: int, hands: tuple[int, ...]) -> Optional[int]:
        for k in range(num_players):
            idx = (mayor + k) % num_players
            if idx < 0 or idx >= len(hands):
                continue
            if hands[idx] > 0:
                return idx
        return None

    @staticmethod
    def _mayor_next_placement_after(
        mayor: int, num_players: int, hands: list[int], after_player: int
    ) -> Optional[int]:
        for step in range(1, num_players + 1):
            idx = (after_player + step) % num_players
            if idx < 0 or idx >= len(hands):
                continue
            if hands[idx] > 0:
                return idx
        return None

    def _mayor_refill_ship_and_advance(self, state: GameState, pend: MayorPhasePending) -> None:
        need = pend.ship_size_at_start
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
        pend2 = dataclasses.replace(pend, privilege_done=True)
        if s.colonist_ship > 0:
            self._state = dataclasses.replace(s, pending=dataclasses.replace(pend2, subphase="draft"))
            return
        placement_next = self._mayor_next_with_colonists_in_hand(
            pend2.mayor_role_chooser, s.num_players, pend2.colonists_hands
        )
        if placement_next is None:
            self._mayor_refill_ship_and_advance(s, pend2)
        else:
            self._state = dataclasses.replace(
                s,
                pending=dataclasses.replace(
                    pend2,
                    subphase="placement",
                    placement_next=placement_next,
                ),
            )

    def _apply_mayor(self, player_id: int, action: EngineAction, pend: MayorPhasePending) -> Optional[str]:
        s = self._state
        mayor = pend.mayor_role_chooser
        n = s.num_players
        if pend.subphase == "privilege":
            if player_id != mayor:
                return "mayor privilege only"
            if isinstance(action, MayorPrivilegeSkip):
                self._mayor_after_privilege(s, pend)
                return None
            if isinstance(action, MayorPrivilegeTake):
                if s.colonist_supply <= 0:
                    return "empty colonist supply"
                hands = list(pend.colonists_hands)
                if mayor < 0 or mayor >= len(hands):
                    return "invalid mayor colonists_hands index"
                hands[mayor] += 1
                pend2 = dataclasses.replace(pend, colonists_hands=tuple(hands))
                s2 = dataclasses.replace(s, colonist_supply=s.colonist_supply - 1, pending=pend2)
                self._mayor_after_privilege(s2, pend2)
                return None
            return "bad mayor privilege action"
        if pend.subphase == "draft":
            k = pend.ship_size_at_start - s.colonist_ship
            nxt = (mayor + k) % n
            if player_id != nxt or s.colonist_ship <= 0:
                return "not draft turn"
            if not isinstance(action, MayorDraftTake):
                return "bad draft action"
            hands = list(pend.colonists_hands)
            if player_id < 0 or player_id >= len(hands):
                return "invalid mayor colonists_hands index"
            hands[player_id] += 1
            s2 = dataclasses.replace(s, colonist_ship=s.colonist_ship - 1)
            rem = s2.colonist_ship
            pend2 = dataclasses.replace(
                pend,
                colonists_hands=tuple(hands),
                colonists_from_ship_remaining=rem,
            )
            if rem == 0:
                placement_next = self._mayor_next_with_colonists_in_hand(mayor, n, pend2.colonists_hands)
                if placement_next is None:
                    self._mayor_refill_ship_and_advance(s2, pend2)
                    return None
                self._state = dataclasses.replace(
                    s2,
                    pending=dataclasses.replace(
                        pend2,
                        subphase="placement",
                        placement_next=placement_next,
                    ),
                )
            else:
                self._state = dataclasses.replace(s2, pending=pend2)
            return None
        if pend.subphase == "placement":
            if pend.placement_next is None or player_id != pend.placement_next:
                return "not placement turn"
            hands = list(pend.colonists_hands)
            if player_id < 0 or player_id >= len(hands):
                return "invalid mayor colonists_hands index"
            if hands[player_id] <= 0:
                return "empty hand"
            p = s.players[player_id]
            if isinstance(action, MayorPlaceColonistIsland):
                si = action.island_slot
                if si < 0 or si >= len(p.island_spaces):
                    return "bad island slot"
                sp = p.island_spaces[si]
                if sp.tile is None:
                    return "empty island space"
                mx = island_tile_max_colonists(sp.tile)
                if sp.colonists >= mx:
                    return "island full"
                new_pl = list(p.island_spaces)
                new_pl[si] = IslandSpace(tile=sp.tile, colonists=sp.colonists + 1)
                hands[player_id] -= 1
                s2 = _replace_player(s, player_id, dataclasses.replace(p, island_spaces=tuple(new_pl)))
            elif isinstance(action, MayorPlaceColonistBuilding):
                bi, ci = action.building_index, action.circle_index
                if bi < 0 or bi >= len(p.city_buildings):
                    return "bad building index"
                pb = p.city_buildings[bi]
                if ci < 0 or ci >= len(pb.colonists):
                    return "bad circle index"
                if pb.colonists[ci] != 0:
                    return "circle occupied"
                col = list(pb.colonists)
                col[ci] = 1
                new_pb = dataclasses.replace(pb, colonists=tuple(col))
                new_list = list(p.city_buildings)
                new_list[bi] = new_pb
                hands[player_id] -= 1
                s2 = _replace_player(s, player_id, dataclasses.replace(p, city_buildings=tuple(new_list)))
            elif isinstance(action, MayorPlaceColonistSanJuan):
                if self._any_empty_circle(p):
                    return "must place on board first"
                hands[player_id] -= 1
                s2 = _replace_player(
                    s,
                    player_id,
                    dataclasses.replace(p, san_juan_colonists=p.san_juan_colonists + 1),
                )
            else:
                return "bad placement action"
            if sum(hands) == 0:
                self._mayor_refill_ship_and_advance(s2, pend)
                return None
            nxt = self._mayor_next_placement_after(mayor, n, hands, player_id)
            if nxt is None:
                self._mayor_refill_ship_and_advance(
                    s2,
                    dataclasses.replace(pend, colonists_hands=tuple(hands)),
                )
                return None
            self._state = dataclasses.replace(
                s2,
                pending=dataclasses.replace(
                    pend,
                    colonists_hands=tuple(hands),
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
        if player_id == chooser and _occupied_violet_building(p, Building.LARGE_MARKET):
            s = _bank_receive(s, player_id, 1)
            p = s.players[player_id]
        prod = _compute_craftsman_production(p, s.goods_supply)
        state = s
        for g, n in prod.items():
            for _ in range(n):
                state, err = _goods_supply_take(state, g, 1)
                if err:
                    return err
                state = _player_add_goods(state, player_id, g, 1)
        p = state.players[player_id]
        if action.hacienda_good is not None:
            if not _occupied_violet_building(p, Building.HACIENDA):
                return "no hacienda"
            if action.hacienda_good not in prod or prod.get(action.hacienda_good, 0) <= 0:
                return "bad hacienda good"
            state, err = _goods_supply_take(state, action.hacienda_good, 1)
            if err:
                return err
            state = _player_add_goods(state, player_id, action.hacienda_good, 1)
        if action.privilege_good is not None:
            if player_id != chooser:
                return "privilege only for chooser"
            kinds = set(prod.keys())
            if action.hacienda_good is not None:
                kinds.add(action.hacienda_good)
            if action.privilege_good not in kinds:
                return "bad privilege good"
            state, err = _goods_supply_take(state, action.privilege_good, 1)
            if err:
                return err
            state = _player_add_goods(state, player_id, action.privilege_good, 1)
        self._craftsman_finish_turn(state, pend, player_id)
        return None

    def _trader_finish_turn(self, state: GameState, pend: TraderPhasePending, player_id: int) -> None:
        order = role_action_order(pend.role_chooser, state.num_players)
        idx = order.index(player_id)
        if idx == len(order) - 1:
            s2 = dataclasses.replace(state, trading_house=TradingHouseState(goods=()), pending=None)
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
        s2, err = _player_sub_goods(s, player_id, g, 1)
        if err:
            return err
        s3 = _bank_receive(s2, player_id, price)
        new_goods = tuple(house.goods) + (g,)
        s4 = dataclasses.replace(s3, trading_house=TradingHouseState(goods=new_goods))
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
        credit = list(pend.ship_full_credit)
        st = state
        for i, sh in enumerate(ships):
            if sh.barrels < sh.capacity or sh.good is None:
                continue
            g = sh.good
            st = _goods_supply_add(st, g, sh.barrels)
            if i >= len(credit):
                ships[i] = CargoShipState(capacity=sh.capacity, good=None, barrels=0)
                continue
            captain = credit[i]
            if captain is not None:
                goods_val = _TRADER_PRICE[g] * sh.barrels
                vp_amt = 1 if st.vp_supply > 0 else goods_val
                st2, err = _award_vp_from_supply(st, captain, vp_amt)
                if err:
                    return state
                st = st2
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
