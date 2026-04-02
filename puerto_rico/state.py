# Modeling notes (read before extending rules / engine):
#
# - City layout uses `PlacedBuilding` with an anchor slot (0–11). Large buildings occupy
#   `anchor_slot` and `anchor_slot + 1`; validation of adjacency and overlaps is left to
#   game logic — state only stores consistent placements.
# - Quarry discount caps use **city column** (1–4) from anchor slot on the 3×4 grid
#   (`anchor_slot % 4 + 1`); large buildings use the left anchor’s column per rulebook examples.
# - "Occupied" for tiles/buildings means ≥1 colonist on that tile/building (rules).
#   Unoccupied tiles/buildings do not grant abilities; printed VP on buildings still counts.
# - VP from chips may exceed the physical supply (paper VP); `vp_from_chips` holds the
#   total including any paper-recorded VP for Customs House / endgame.
# - Trading house order is a tuple (FIFO/LIFO does not matter for uniqueness rules;
#   engine may treat it as a set with order for Office edge cases).
# - Colonist ship / supply: when supply is empty, rules restrict mayor privilege and
#   refills — represented numerically; "not refilled" is an engine transition, not a flag.
# - Captain phase needs multiple passes; `CaptainPhasePending` holds minimal turn state.
#   Wharf once/phase, Harbor per-load VP, and mandatory loading are enforced in rules code.
# - Role.PROSPECTOR names the role generically; Role.PROSPECTOR_A / PROSPECTOR_B distinguish the
#   two physical cards in 5p (doubloon stacks, `roles_in_play`). Use PROSPECTOR when the card
#   identity does not matter; use A/B when modeling the full 8-card deck.
# - Goods / colonists / tiles use nonnegative ints; invariants (caps, empty stacks) are
#   enforced by transitions, not by dataclass validators, to keep construction cheap for RL.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, Literal, Mapping, Optional, TypeAlias, Union


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Phase(str, Enum):
    """High-level segment of the game clock (not individual role steps)."""

    SETUP = "setup"
    """Initial dealing, stacks, and pre-first-round configuration."""

    ROLE_SELECTION = "role_selection"
    """Governor and then clockwise players each take one role card for the round."""

    SETTLER = "settler"
    """Settler role: plantations/quarries and end-of-phase face-up refresh."""

    MAYOR = "mayor"
    """Mayor role: colonist ship distribution, placement, San Juan, ship refill."""

    BUILDER = "builder"
    """Builder role: each player may build at most one building."""

    CRAFTSMAN = "craftsman"
    """Craftsman role: production from occupied plantations and production buildings."""

    TRADER = "trader"
    """Trader role: selling to the trading house (capacity 4, distinct goods)."""

    CAPTAIN = "captain"
    """Captain role: loading cargo ships, VP for barrels, then storage and ship unload."""

    PROSPECTOR = "prospector"
    """Prospector: no action; privilege is doubloons from bank (+ role card doubloons)."""

    ROUND_CLEANUP = "round_cleanup"
    """After all roles: doubloons on unused roles, return role cards, pass governor."""

    GAME_OVER = "game_over"
    """Game ended; final scoring happens outside or atop this state."""


class Role(str, Enum):
    """Role cards: full 5p deck uses two prospector cards (A/B); 3p omits both prospectors.

    ``PROSPECTOR`` is the generic prospector role (rules text). ``PROSPECTOR_A`` / ``PROSPECTOR_B``
    are the two distinguishable cards for stacking doubloons and ``roles_in_play`` in 8-card games.
    """

    SETTLER = "settler"
    MAYOR = "mayor"
    BUILDER = "builder"
    CRAFTSMAN = "craftsman"
    TRADER = "trader"
    CAPTAIN = "captain"
    PROSPECTOR = "prospector"
    PROSPECTOR_A = "prospector_a"
    PROSPECTOR_B = "prospector_b"


class Good(str, Enum):
    """Barrel types (finished goods) in the supply and on windroses."""

    COFFEE = "coffee"
    TOBACCO = "tobacco"
    CORN = "corn"
    SUGAR = "sugar"
    INDIGO = "indigo"


class IslandTile(str, Enum):
    """Plantation types and the quarry tile drawn from the island tile supply."""

    QUARRY = "quarry"
    COFFEE = "coffee"
    TOBACCO = "tobacco"
    CORN = "corn"
    SUGAR = "sugar"
    INDIGO = "indigo"


class Building(str, Enum):
    """All buildable buildings (production, violet, large). Two copies exist in the supply
    for most violet buildings; uniqueness is enforced by game logic, not by this enum."""

    # Production
    SMALL_INDIGO_PLANT = "small_indigo_plant"
    LARGE_INDIGO_PLANT = "large_indigo_plant"
    SMALL_SUGAR_MILL = "small_sugar_mill"
    LARGE_SUGAR_MILL = "large_sugar_mill"
    TOBACCO_STORAGE = "tobacco_storage"
    COFFEE_ROASTER = "coffee_roaster"

    # Violet (small / large)
    SMALL_MARKET = "small_market"
    HACIENDA = "hacienda"
    CONSTRUCTION_HUT = "construction_hut"
    SMALL_WAREHOUSE = "small_warehouse"
    HOSPICE = "hospice"
    OFFICE = "office"
    LARGE_MARKET = "large_market"
    LARGE_WAREHOUSE = "large_warehouse"
    UNIVERSITY = "university"
    FACTORY = "factory"
    HARBOR = "harbor"
    WHARF = "wharf"

    # Large (unique)
    GUILD_HALL = "guild_hall"
    RESIDENCE = "residence"
    FORTRESS = "fortress"
    CUSTOMS_HOUSE = "customs_house"
    CITY_HALL = "city_hall"


# ---------------------------------------------------------------------------
# Building metadata (pure functions; rules use these for costs / circles / columns)
# ---------------------------------------------------------------------------


def building_city_spaces(building: Building) -> int:
    """Number of adjacent city spaces this building occupies (1 or 2)."""

    large = {
        Building.GUILD_HALL,
        Building.RESIDENCE,
        Building.FORTRESS,
        Building.CUSTOMS_HOUSE,
        Building.CITY_HALL,
    }
    return 2 if building in large else 1


def building_worker_circles(building: Building) -> int:
    """Colonist circles on this building (not island tiles)."""

    if building in (
        Building.LARGE_INDIGO_PLANT,
        Building.LARGE_SUGAR_MILL,
        Building.TOBACCO_STORAGE,
        Building.COFFEE_ROASTER,
    ):
        return 2
    if building_city_spaces(building) == 2:
        return 2
    return 1


def city_anchor_slot_column_1based(anchor_slot: int) -> int:
    """Which of the four city columns (1–4) this anchor slot lies in on the standard 3×4 grid.

    The rulebook caps quarry discount by **column** (1→max 1, 2→max 2, …). Row-major slots
    0–11: ``column = anchor_slot % 4 + 1``. Large buildings use the left column of the two spaces.
    """

    return (anchor_slot % 4) + 1


def max_quarry_discount_for_city_slot(anchor_slot: int) -> int:
    """Maximum number of occupied quarries that can apply to a building in this column (1–4)."""

    return city_anchor_slot_column_1based(anchor_slot)


def building_printed_cost(building: Building) -> int:
    """Printed doubloon cost (before builder privilege and quarries)."""

    return _BUILDING_COST_VP[building][0]


def building_printed_vp(building: Building) -> int:
    """Printed VP on the tile (counts even if unoccupied)."""

    return _BUILDING_COST_VP[building][1]


_BUILDING_COST_VP: Final[dict[Building, tuple[int, int]]] = {
    Building.SMALL_INDIGO_PLANT: (1, 1),
    Building.SMALL_SUGAR_MILL: (2, 1),
    Building.LARGE_INDIGO_PLANT: (3, 2),
    Building.LARGE_SUGAR_MILL: (4, 2),
    Building.TOBACCO_STORAGE: (5, 3),
    Building.COFFEE_ROASTER: (6, 3),
    Building.SMALL_MARKET: (1, 1),
    Building.HACIENDA: (2, 1),
    Building.CONSTRUCTION_HUT: (2, 1),
    Building.SMALL_WAREHOUSE: (3, 1),
    Building.HOSPICE: (4, 2),
    Building.OFFICE: (5, 2),
    Building.LARGE_MARKET: (5, 2),
    Building.LARGE_WAREHOUSE: (6, 2),
    Building.UNIVERSITY: (7, 3),
    Building.FACTORY: (8, 3),
    Building.HARBOR: (8, 3),
    Building.WHARF: (9, 3),
    Building.GUILD_HALL: (10, 4),
    Building.RESIDENCE: (10, 4),
    Building.FORTRESS: (10, 4),
    Building.CUSTOMS_HOUSE: (10, 4),
    Building.CITY_HALL: (10, 4),
}


def island_tile_max_colonists(tile: IslandTile) -> int:
    """Maximum colonists that may be placed on this island tile (circles printed on tile)."""

    return 3 if tile is IslandTile.QUARRY else 1


def island_tile_is_quarry(tile: IslandTile) -> bool:
    return tile is IslandTile.QUARRY


# ---------------------------------------------------------------------------
# Goods as immutable counts
# ---------------------------------------------------------------------------

GoodsCounts: TypeAlias = tuple[tuple[Good, int], ...]
BuildingSupplyCounts: TypeAlias = tuple[tuple[Building, int], ...]


def goods_total(counts: Mapping[Good, int] | GoodsCounts) -> int:
    if isinstance(counts, Mapping):
        return sum(counts.values())
    return sum(c for _, c in counts)


def good_count(counts: Mapping[Good, int] | GoodsCounts, good: Good) -> int:
    if isinstance(counts, tuple) and counts and isinstance(counts[0], tuple):
        # GoodsCounts = tuple[tuple[Good, int], ...]
        for g, n in counts:
            if g == good:
                return int(n)
        return 0
    if not counts:
        return 0
    if isinstance(counts, tuple) and not (isinstance(counts[0], tuple) if counts else False):
        return 0
    return int(counts.get(good, 0))


# ---------------------------------------------------------------------------
# Player board pieces
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IslandSpace:
    """One of 12 island spaces (plantation or quarry).

    Attributes
    ----------
    tile:
        Plantation or quarry placed on this space; ``None`` if empty. Tiles are never removed.
    colonists:
        Colonists on this tile's printed circle(s). Quarries have three circles; plantations
        typically have one. Unoccupied tiles do not produce or grant quarry discounts.
    """

    tile: Optional[IslandTile]
    colonists: int


@dataclass(frozen=True, slots=True)
class PlacedBuilding:
    """A building built in the city (up to 12 spaces; large buildings span two adjacent).

    Attributes
    ----------
    building:
        Which building type; printed VP counts even when unoccupied; abilities need occupation.
    anchor_slot:
        Index (0–11) of the leftmost space; large violet and large production use two spaces.
    colonists:
        Colonists per worker circle (order matches building artwork). Only occupied circles function.
    """

    building: Building
    anchor_slot: int
    colonists: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PlayerState:
    """Per-player board and personal supply (windrose, San Juan, city, island).

    Attributes
    ----------
    doubloons:
        Coins from the bank; spent on buildings and received from roles/trades.
    vp_from_chips:
        Total VP represented by chips (1s and 5s) plus any VP recorded on paper when the chip
        supply ran out (for Customs House and endgame). Kept secret in play but modeled here.
    san_juan_colonists:
        Colonists not placed on tiles/buildings after mayor; must be placed when circles exist.
    island_spaces:
        Twelve spaces for plantations/quarries; order is a fixed seat convention for the engine.
    city_buildings:
        Built production and violet buildings; placement legality is enforced outside this struct.
    goods:
        Barrels on the windrose by type. Captain-phase storage rules apply at end of captain.
    vp_chips_1:
        Number of 1-VP chips (optional bookkeeping; ``vp_from_chips`` is authoritative total).
    vp_chips_5:
        Number of 5-VP chips (optional bookkeeping for exchanges).
    vp_on_paper:
        VP recorded on paper when the VP supply could not fully pay a **building** purchase
        (captain VP shortfall is folded into ``vp_from_chips``). Used for Customs House–style
        bookkeeping and display.
    """

    doubloons: int
    vp_from_chips: int
    vp_on_paper: int
    san_juan_colonists: int
    island_spaces: tuple[IslandSpace, ...]
    city_buildings: tuple[PlacedBuilding, ...]
    goods: GoodsCounts
    vp_chips_1: int
    vp_chips_5: int


# ---------------------------------------------------------------------------
# Global / board components
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CargoShipState:
    """One cargo ship (three in play). Each holds at most one good type.

    Attributes
    ----------
    capacity:
        Max barrels (setup: 4/5/6 for 3 players through 6/7/8 for 5 players).
    good:
        Kind of good on this ship; two ships may not hold the same kind (captain rules).
    barrels:
        Loaded count; cannot exceed ``capacity``; full ships may be emptied to supply at phase end.
    """

    capacity: int
    good: Optional[Good]
    barrels: int


@dataclass(frozen=True, slots=True)
class TradingHouseState:
    """Trading house (capacity four; distinct goods unless Office allows duplicates).

    Attributes
    ----------
    goods:
        Up to four sold goods, in sale order. If not full after trader phase, goods remain for
        the next trader phase (making it harder to sell new kinds).
    """

    goods: tuple[Good, ...]


@dataclass(frozen=True, slots=True)
class SettlerPhasePending:
    """Optional stepwise state while resolving Settler (chooser privilege vs others).

    Attributes
    ----------
    settler_role_chooser:
        Player who took Settler (may take quarry privilege or face-up plantation first).
    next_actor_index:
        Next player to resolve in clockwise order after the chooser; None if between sub-steps.
    awaiting_normal_pick:
        True only after the active player has used Hacienda and must now resolve their normal
        settler pick (face-up plantation or legal quarry).
    """

    settler_role_chooser: int
    next_actor_index: Optional[int]
    awaiting_normal_pick: bool = False


@dataclass(frozen=True, slots=True)
class MayorPhasePending:
    """Mayor as micro-steps: privilege → draft from ship (one at a time) → placement → refill ship.

    privilege: mayor may take 1 from supply before the ship is distributed (optional).
    draft: active player is ``(mayor + drafted_count) % N`` where drafted_count =
    ``ship_size_at_start - colonists_from_ship_remaining``.
    placement: ``placement_next`` acts; one colonist per action until all hands empty.
    ship_refill: automatic transition after placement (no player choice); colonist end check here.
    """

    mayor_role_chooser: int
    ship_size_at_start: int
    colonists_from_ship_remaining: int
    colonists_hands: tuple[int, ...]
    subphase: Literal["privilege", "draft", "placement", "ship_refill"]
    privilege_done: bool
    placement_next: Optional[int]


@dataclass(frozen=True, slots=True)
class CaptainPhasePending:
    """Captain phase: loading loop → windrose storage → unload full ships to supply.

    loading: clockwise passes until no legal load remains for anyone.
    storage: each player trims windrose to allowed retention (default 1 + warehouses).
    unload_full_ships: automatic; full ships empty to goods supply (captain duty at phase end).
    ship_full_credit: which player last completed loading each ship to capacity (for VP at unload).
    """

    captain_role_chooser: int
    active_player_index: int
    captain_privilege_vp_awarded: bool
    wharf_used: tuple[bool, ...]
    subphase: Literal["loading", "storage", "unload_full_ships"]
    storage_next_actor: Optional[int]
    storage_done: tuple[bool, ...]
    ship_full_credit: tuple[Optional[int], Optional[int], Optional[int]]


@dataclass(frozen=True, slots=True)
class CraftsmanPhasePending:
    """Craftsman: chooser acts first then clockwise; each step applies production then optional privilege."""

    role_chooser: int
    next_actor: int


@dataclass(frozen=True, slots=True)
class TraderPhasePending:
    """Trader: optional sell per player; chooser first then clockwise."""

    role_chooser: int
    next_actor: int


@dataclass(frozen=True, slots=True)
class BuilderPhasePending:
    """Builder: at most one build per player; chooser first then clockwise."""

    role_chooser: int
    next_actor: int


@dataclass(frozen=True, slots=True)
class GenericRolePending:
    """Fallback pending context for roles without a dedicated struct yet.

    Attributes
    ----------
    role:
        Role being resolved.
    role_chooser:
        Player who chose the role (privilege and first in clockwise action order).
    next_actor_index:
        Cursor for the next player to act, if the engine resolves one player at a time.
    """

    role: Role
    role_chooser: int
    next_actor_index: Optional[int]


PhasePending: TypeAlias = Union[
    SettlerPhasePending,
    MayorPhasePending,
    CaptainPhasePending,
    CraftsmanPhasePending,
    TraderPhasePending,
    BuilderPhasePending,
    GenericRolePending,
    None,
]


@dataclass(frozen=True, slots=True)
class GameState:
    """Canonical match state for Puerto Rico (single source of truth for the engine).

    Attributes
    ----------
    num_players:
        Player count (3, 4, or 5). Sets starting colonist ship, VP supply, cargo ship capacities,
        role cards in play, face-up plantation count (players + 1), and starting doubloons.
    phase:
        Current segment (setup, role selection, a role phase, round cleanup, or game over).
    governor_index:
        Who has the Governor card this round (selects first role, then clockwise picks).
    players:
        Fixed seating order index 0..N-1; clockwise order is derived with modular arithmetic.
    roles_in_play:
        Subset of ``Role`` for this game (e.g. both Prospectors only in 5-player; 3-player drops both).
    role_card_doubloons:
        Doubloons placed on each **unclaimed** role card this round; chooser takes them when taking
        the card. Stored as sorted (role, count) pairs for an immutable map-like view.
    player_roles_this_round:
        After selection, which role each player holds until round cleanup (one per player each round).
    next_role_selector_index:
        During ``ROLE_SELECTION``, who picks the next role (governor first, then clockwise). None
        when selection is done or in other phases.
    plantation_stacks:
        Five shuffled face-down stacks of plantation tiles (not quarries). Convention: index 0 is
        stack "one" through five as in setup; order within tuple is bottom-to-top of that stack.
    face_up_plantations:
        Face-up plantations available in the settler phase (not including quarries in the quarry supply).
    plantation_discard:
        Face-up tiles discarded after settler and reshuffled when stacks are exhausted.
    quarries_remaining:
        Quarries left in the central quarry supply (all eight start face-up; taken in settler phase).
    colonist_ship:
        Colonists on the ship (drafted one at a time in mayor). Initial size 3/4/5 by player count.
    colonist_supply:
        Colonists not on the ship or boards. Mayor refills the ship from here; game can end if refill fails.
    bank_doubloons:
        Doubloons not in front of players (bank stock).
    goods_supply:
        Available barrels in the general supply by type (craftsman draws here; captain returns here).
    vp_supply:
        VP chips left to take during play (captain); when empty, track extra VP on paper per rules.
    cargo_ships:
        Three ships; capacities depend on player count; each holds at most one good type.
    trading_house:
        Holds up to four goods between trader phases; may retain goods if not full.
    round_number:
        Round counter (approximately 15 rounds per game).
    round_role_order:
        Roles chosen this round in order, each ``(Role, chooser_player_index)``. Filled during
        role-selection phase; drives execution order (not the printed role-card order).
    current_role_execution_index:
        Index into ``round_role_order`` for the role currently resolving; ``None`` during
        ``ROLE_SELECTION`` before the first role begins.
    building_supply:
        Remaining building tiles in the central supply (copies per ``Building``).
    game_end_colonists:
        Set when mayor refill cannot place required colonists on the ship (checked end of mayor).
    game_end_city12:
        Set when a player builds on their 12th city space (checked during builder).
    game_end_vp:
        Set when the last VP chip is taken from supply during captain (paper VP continues).
    pending:
        Role-specific resolution state (captain loading loop, settler/mayor cursors, etc.).
    """

    num_players: int
    phase: Phase
    governor_index: int
    players: tuple[PlayerState, ...]
    roles_in_play: frozenset[Role]
    role_card_doubloons: tuple[tuple[Role, int], ...]
    player_roles_this_round: tuple[Optional[Role], ...]
    next_role_selector_index: Optional[int]
    round_role_order: tuple[tuple[Role, int], ...]
    current_role_execution_index: Optional[int]
    building_supply: BuildingSupplyCounts
    game_end_colonists: bool
    game_end_city12: bool
    game_end_vp: bool
    plantation_stacks: tuple[tuple[IslandTile, ...], ...]
    face_up_plantations: tuple[IslandTile, ...]
    plantation_discard: tuple[IslandTile, ...]
    quarries_remaining: int
    colonist_ship: int
    colonist_supply: int
    bank_doubloons: int
    goods_supply: GoodsCounts
    vp_supply: int
    cargo_ships: tuple[CargoShipState, ...]
    trading_house: TradingHouseState
    round_number: int
    pending: PhasePending


# ---------------------------------------------------------------------------
# Pure helpers: order
# ---------------------------------------------------------------------------


def clockwise_indices(start: int, num_players: int) -> tuple[int, ...]:
    """All player indices in clockwise order starting at `start` (0-based)."""

    if num_players <= 0:
        return ()
    return tuple((start + k) % num_players for k in range(num_players))


def role_selection_order(governor_index: int, num_players: int) -> tuple[int, ...]:
    """Order in which players choose role cards: governor first, then clockwise."""

    return clockwise_indices(governor_index, num_players)


def role_action_order(role_chooser_index: int, num_players: int) -> tuple[int, ...]:
    """Order of role actions: chooser first, then clockwise (same as clockwise from chooser)."""

    return clockwise_indices(role_chooser_index, num_players)


# ---------------------------------------------------------------------------
# Occupancy and counting
# ---------------------------------------------------------------------------


def island_space_occupied(space: IslandSpace) -> bool:
    """True if the tile is present and has at least one colonist (rules: then it functions)."""

    return space.tile is not None and space.colonists > 0


def building_occupied(pb: PlacedBuilding) -> bool:
    """True if at least one worker circle has a colonist."""

    return any(c > 0 for c in pb.colonists)


def tile_or_building_occupied_for_function(
    *,
    island_space: Optional[IslandSpace] = None,
    placed_building: Optional[PlacedBuilding] = None,
) -> bool:
    """Unified occupancy check for island tiles and buildings (abilities require occupation)."""

    if island_space is not None:
        return island_space_occupied(island_space)
    if placed_building is not None:
        return building_occupied(placed_building)
    return False


def count_occupied_quarries(player: PlayerState) -> int:
    """Island tiles that are quarries and occupied (colonist present)."""

    n = 0
    for sp in player.island_spaces:
        if sp.tile is IslandTile.QUARRY and island_space_occupied(sp):
            n += 1
    return n


def count_empty_building_circles(player: PlayerState) -> int:
    """Empty worker circles on buildings only (mayor refill uses this at end of mayor)."""

    empty = 0
    for pb in player.city_buildings:
        for c in pb.colonists:
            if c == 0:
                empty += 1
    return empty


def count_filled_island_spaces(player: PlayerState) -> int:
    """Island spaces with a tile placed (plantation or quarry)."""

    return sum(1 for sp in player.island_spaces if sp.tile is not None)


def total_colonists_on_board(player: PlayerState) -> int:
    """All colonists on this player's board: San Juan, island, and city buildings."""

    s = player.san_juan_colonists
    for sp in player.island_spaces:
        s += sp.colonists
    for pb in player.city_buildings:
        s += sum(pb.colonists)
    return s


def goods_dict(counts: GoodsCounts) -> dict[Good, int]:
    """Helper: materialize goods tuple to a dict (last duplicate wins if malformed)."""

    if not counts:
        return {}
    if isinstance(counts, tuple) and not isinstance(counts[0], tuple):
        return {}
    d: dict[Good, int] = {}
    for g, n in counts:
        d[g] = n
    return d


def normalize_goods_counts(raw: Mapping[Good, int]) -> GoodsCounts:
    """Stable sorted tuple for canonical immutable storage."""

    return tuple(sorted(((g, int(raw[g])) for g in Good if raw.get(g, 0) != 0), key=lambda x: x[0].value))


def normalize_building_supply(raw: Mapping[Building, int]) -> BuildingSupplyCounts:
    """Stable sorted tuple for remaining building tiles in the supply."""

    return tuple(sorted(((b, int(raw[b])) for b in Building if raw.get(b, 0) != 0), key=lambda x: x[0].value))


__all__ = [
    "Building",
    "BuildingSupplyCounts",
    "BuilderPhasePending",
    "CaptainPhasePending",
    "CargoShipState",
    "CraftsmanPhasePending",
    "GameState",
    "GenericRolePending",
    "Good",
    "GoodsCounts",
    "IslandSpace",
    "IslandTile",
    "MayorPhasePending",
    "Phase",
    "PhasePending",
    "PlacedBuilding",
    "PlayerState",
    "Role",
    "SettlerPhasePending",
    "TraderPhasePending",
    "TradingHouseState",
    "building_city_spaces",
    "building_occupied",
    "building_printed_cost",
    "building_printed_vp",
    "building_worker_circles",
    "city_anchor_slot_column_1based",
    "clockwise_indices",
    "count_empty_building_circles",
    "count_filled_island_spaces",
    "count_occupied_quarries",
    "goods_dict",
    "goods_total",
    "good_count",
    "island_space_occupied",
    "island_tile_is_quarry",
    "island_tile_max_colonists",
    "max_quarry_discount_for_city_slot",
    "normalize_building_supply",
    "normalize_goods_counts",
    "role_action_order",
    "role_selection_order",
    "tile_or_building_occupied_for_function",
    "total_colonists_on_board",
]
