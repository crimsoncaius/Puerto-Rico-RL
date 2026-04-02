# Rule coverage checklist (Setup / components encoded here):
# ---------------------------------------------------------------------------
# [x] Quarry tiles: QUARRY_TILE_COUNT — all face-up in quarries_remaining (setup.py).
# [x] Plantation tiles: PLANTATION_TILE_COUNTS + PLANTATION_TILE_TOTAL — shuffled into 5 stacks.
# [x] Goods: GOOD_SUPPLY_COUNTS — sorted into supply (setup.py goods_supply).
# [x] Trading house: empty at start (TradingHouseState in setup.py).
# [x] Colonist ship: COLONIST_SHIP_BY_PLAYER_COUNT — 3/4/5 by player count.
# [x] Colonist supply: COLONIST_SUPPLY_BY_PLAYER_COUNT — 55/75/95.
# [x] Role cards: roles_for_player_count() — 6/7/8 (no prospectors / one / two).
# [x] Cargo ships: CARGO_SHIP_CAPACITIES_BY_PLAYER_COUNT — capacities per player count.
# [x] Victory points: VP_SUPPLY_TOTAL_BY_PLAYER_COUNT — 75/100/122 VP in pool.
# [x] Starting doubloons: STARTING_DOUBLOONS_BY_PLAYER_COUNT — 2/3/4 per player.
# [x] Bank: TOTAL_DOUBLOON_VALUE − players' starting cash (setup.py bank_doubloons).
# [x] Starting plantations: STARTING_PLANTATIONS_BY_PLAYER_COUNT — governor indigo + table.
# [x] Face-up plantations after shuffle: face_up_plantation_count() — players + 1.
# [x] Buildings: BUILDING_METADATA — cost, printed VP, city size, circles, category, copies.
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from .state import (
    Building,
    Good,
    IslandTile,
    Role,
    building_city_spaces,
    building_printed_cost,
    building_printed_vp,
    building_worker_circles,
)


# ---------------------------------------------------------------------------
# Player counts
# ---------------------------------------------------------------------------

SUPPORTED_PLAYER_COUNTS: Final[tuple[int, ...]] = (3, 4, 5)


# ---------------------------------------------------------------------------
# Island / goods inventory (rulebook component counts)
# ---------------------------------------------------------------------------

QUARRY_TILE_COUNT: Final[int] = 8

# 50 plantation tiles: coffee 8, tobacco 9, corn 10, sugar 11, indigo 12
PLANTATION_TILE_COUNTS: Final[dict[IslandTile, int]] = {
    IslandTile.COFFEE: 8,
    IslandTile.TOBACCO: 9,
    IslandTile.CORN: 10,
    IslandTile.SUGAR: 11,
    IslandTile.INDIGO: 12,
}

PLANTATION_TILE_TOTAL: Final[int] = sum(PLANTATION_TILE_COUNTS.values())

# 50 goods barrels — matches plantation maxima
GOOD_SUPPLY_COUNTS: Final[dict[Good, int]] = {
    Good.COFFEE: 9,
    Good.TOBACCO: 9,
    Good.CORN: 10,
    Good.SUGAR: 11,
    Good.INDIGO: 11,
}

GOOD_SUPPLY_TOTAL: Final[int] = sum(GOOD_SUPPLY_COUNTS.values())

# Physical doubloon pieces: 46×1 + 8×5 → total currency value in the box
DOUBLOON_ONES: Final[int] = 46
DOUBLOON_FIVES: Final[int] = 8
TOTAL_DOUBLOON_VALUE: Final[int] = DOUBLOON_ONES + 5 * DOUBLOON_FIVES

# VP chips (physical): 32×1 + 18×5 — max pool value 122 (5-player game uses all)
VP_CHIP_ONES: Final[int] = 32
VP_CHIP_FIVES: Final[int] = 18
VP_CHIPS_PHYSICAL_TOTAL_VALUE: Final[int] = VP_CHIP_ONES + 5 * VP_CHIP_FIVES

# Colonist ship at game start (rulebook: 3 / 4 / 5 for 3 / 4 / 5 players)
COLONIST_SHIP_BY_PLAYER_COUNT: Final[dict[int, int]] = {
    3: 3,
    4: 4,
    5: 5,
}

# Colonist supply (colonists on board supply track, not on ship), by player count
COLONIST_SUPPLY_BY_PLAYER_COUNT: Final[dict[int, int]] = {
    3: 55,
    4: 75,
    5: 95,
}

# Unused colonist discs remain in the box (100 total in the game box)
COLONIST_DISCS_TOTAL: Final[int] = 100


# ---------------------------------------------------------------------------
# Setup tables by player count
# ---------------------------------------------------------------------------

STARTING_DOUBLOONS_BY_PLAYER_COUNT: Final[dict[int, int]] = {
    3: 2,
    4: 3,
    5: 4,
}

VP_SUPPLY_TOTAL_BY_PLAYER_COUNT: Final[dict[int, int]] = {
    3: 75,
    4: 100,
    5: 122,
}

CARGO_SHIP_CAPACITIES_BY_PLAYER_COUNT: Final[dict[int, tuple[int, int, int]]] = {
    3: (4, 5, 6),
    4: (5, 6, 7),
    5: (6, 7, 8),
}

# Face-up plantation market after setup (and after each settler refresh): players + 1
def face_up_plantation_count(num_players: int) -> int:
    return num_players + 1


class BuildingCategory(str, Enum):
    """High-level grouping for metadata and end-game scoring hooks."""

    PRODUCTION = "production"
    """Production buildings (mills, roasters, storage)."""

    VIOLET_SMALL = "violet_small"
    """Small / medium violet buildings (one city space each)."""

    LARGE_UNIQUE = "large_unique"
    """The five unique large buildings (two city spaces each)."""


@dataclass(frozen=True, slots=True)
class BuildingSpec:
    """Static tile data for the base game (cost/VP before discounts; placement uses state helpers)."""

    building: Building
    cost: int
    printed_vp: int
    city_spaces: int
    worker_circles: int
    category: BuildingCategory
    copies_in_supply: int


def _bs(
    building: Building,
    cost: int,
    vp: int,
    city_spaces: int,
    circles: int,
    category: BuildingCategory,
    copies: int,
) -> BuildingSpec:
    return BuildingSpec(
        building=building,
        cost=cost,
        printed_vp=vp,
        city_spaces=city_spaces,
        worker_circles=circles,
        category=category,
        copies_in_supply=copies,
    )


# Production: 4 copies each of the four indigo/sugar buildings; 2 each of tobacco and coffee (20 total)
BUILDING_SPECS: Final[tuple[BuildingSpec, ...]] = (
    _bs(Building.SMALL_INDIGO_PLANT, 1, 1, 1, 1, BuildingCategory.PRODUCTION, 4),
    _bs(Building.SMALL_SUGAR_MILL, 2, 1, 1, 1, BuildingCategory.PRODUCTION, 4),
    _bs(Building.LARGE_INDIGO_PLANT, 3, 2, 1, 2, BuildingCategory.PRODUCTION, 4),
    _bs(Building.LARGE_SUGAR_MILL, 4, 2, 1, 2, BuildingCategory.PRODUCTION, 4),
    _bs(Building.TOBACCO_STORAGE, 5, 3, 1, 2, BuildingCategory.PRODUCTION, 2),
    _bs(Building.COFFEE_ROASTER, 6, 3, 1, 2, BuildingCategory.PRODUCTION, 2),
    # Violet small: two copies each except Office, Large Warehouse, Factory, Harbor, Wharf
    _bs(Building.SMALL_MARKET, 1, 1, 1, 1, BuildingCategory.VIOLET_SMALL, 2),
    _bs(Building.HACIENDA, 2, 1, 1, 1, BuildingCategory.VIOLET_SMALL, 2),
    _bs(Building.CONSTRUCTION_HUT, 2, 1, 1, 1, BuildingCategory.VIOLET_SMALL, 2),
    _bs(Building.SMALL_WAREHOUSE, 3, 1, 1, 1, BuildingCategory.VIOLET_SMALL, 2),
    _bs(Building.HOSPICE, 4, 2, 1, 1, BuildingCategory.VIOLET_SMALL, 2),
    _bs(Building.OFFICE, 5, 2, 1, 1, BuildingCategory.VIOLET_SMALL, 1),
    _bs(Building.LARGE_MARKET, 5, 2, 1, 1, BuildingCategory.VIOLET_SMALL, 2),
    _bs(Building.LARGE_WAREHOUSE, 6, 2, 1, 1, BuildingCategory.VIOLET_SMALL, 1),
    _bs(Building.UNIVERSITY, 7, 3, 1, 1, BuildingCategory.VIOLET_SMALL, 2),
    _bs(Building.FACTORY, 8, 3, 1, 1, BuildingCategory.VIOLET_SMALL, 1),
    _bs(Building.HARBOR, 8, 3, 1, 1, BuildingCategory.VIOLET_SMALL, 1),
    _bs(Building.WHARF, 9, 3, 1, 1, BuildingCategory.VIOLET_SMALL, 1),
    # Large unique (one copy each)
    _bs(Building.GUILD_HALL, 10, 4, 2, 2, BuildingCategory.LARGE_UNIQUE, 1),
    _bs(Building.RESIDENCE, 10, 4, 2, 2, BuildingCategory.LARGE_UNIQUE, 1),
    _bs(Building.FORTRESS, 10, 4, 2, 2, BuildingCategory.LARGE_UNIQUE, 1),
    _bs(Building.CUSTOMS_HOUSE, 10, 4, 2, 2, BuildingCategory.LARGE_UNIQUE, 1),
    _bs(Building.CITY_HALL, 10, 4, 2, 2, BuildingCategory.LARGE_UNIQUE, 1),
)

BUILDING_METADATA: Final[dict[Building, BuildingSpec]] = {s.building: s for s in BUILDING_SPECS}

PRODUCTION_BUILDINGS: Final[frozenset[Building]] = frozenset(
    s.building for s in BUILDING_SPECS if s.category is BuildingCategory.PRODUCTION
)
VIOLET_SMALL_BUILDINGS: Final[frozenset[Building]] = frozenset(
    s.building for s in BUILDING_SPECS if s.category is BuildingCategory.VIOLET_SMALL
)
LARGE_UNIQUE_BUILDINGS: Final[frozenset[Building]] = frozenset(
    s.building for s in BUILDING_SPECS if s.category is BuildingCategory.LARGE_UNIQUE
)
# All violet tiles (small + the five large unique); City Hall scoring counts these.
VIOLET_BUILDINGS: Final[frozenset[Building]] = VIOLET_SMALL_BUILDINGS | LARGE_UNIQUE_BUILDINGS

# Physical building tiles in the box: 20 production + 19 small violet + 5 large.
# (The rules summary line "49 buildings" counts differently from the detailed copy counts.)
EXPECTED_PHYSICAL_BUILDING_TILE_TOTAL: Final[int] = 44
BUILDING_TILE_COUNT_TOTAL: Final[int] = sum(s.copies_in_supply for s in BUILDING_SPECS)
assert BUILDING_TILE_COUNT_TOTAL == EXPECTED_PHYSICAL_BUILDING_TILE_TOTAL


def roles_for_player_count(num_players: int) -> frozenset[Role]:
    """Role cards in play: 3p — 6 (no prospectors); 4p — 7 (one prospector); 5p — 8 (both)."""

    base = frozenset(
        {
            Role.SETTLER,
            Role.MAYOR,
            Role.BUILDER,
            Role.CRAFTSMAN,
            Role.TRADER,
            Role.CAPTAIN,
        }
    )
    if num_players == 3:
        return base
    if num_players == 4:
        return base | {Role.PROSPECTOR_A}
    if num_players == 5:
        return base | {Role.PROSPECTOR_A, Role.PROSPECTOR_B}
    raise ValueError(f"Unsupported player count: {num_players}")


# Starting plantations: governor (seat 0) always gets indigo; others clockwise per rulebook table
STARTING_PLANTATIONS_BY_PLAYER_COUNT: Final[dict[int, tuple[IslandTile, ...]]] = {
    # Index 0 = governor; remaining seats clockwise
    3: (IslandTile.INDIGO, IslandTile.INDIGO, IslandTile.CORN),
    4: (IslandTile.INDIGO, IslandTile.INDIGO, IslandTile.CORN, IslandTile.CORN),
    5: (IslandTile.INDIGO, IslandTile.INDIGO, IslandTile.INDIGO, IslandTile.CORN, IslandTile.CORN),
}

NUM_PLANTATION_STACKS: Final[int] = 5


def _verify_building_specs_match_state() -> None:
    """Ensure `BUILDING_SPECS` matches `state.py` cost/VP/circles/city size (single source of truth)."""

    assert set(BUILDING_METADATA.keys()) == set(Building), "Every Building enum must have metadata"
    for spec in BUILDING_SPECS:
        b = spec.building
        assert building_printed_cost(b) == spec.cost, b
        assert building_printed_vp(b) == spec.printed_vp, b
        assert building_city_spaces(b) == spec.city_spaces, b
        assert building_worker_circles(b) == spec.worker_circles, b


_verify_building_specs_match_state()


__all__ = [
    "BUILDING_METADATA",
    "BUILDING_SPECS",
    "BUILDING_TILE_COUNT_TOTAL",
    "EXPECTED_PHYSICAL_BUILDING_TILE_TOTAL",
    "CARGO_SHIP_CAPACITIES_BY_PLAYER_COUNT",
    "COLONIST_DISCS_TOTAL",
    "COLONIST_SHIP_BY_PLAYER_COUNT",
    "COLONIST_SUPPLY_BY_PLAYER_COUNT",
    "DOUBLOON_FIVES",
    "DOUBLOON_ONES",
    "GOOD_SUPPLY_COUNTS",
    "GOOD_SUPPLY_TOTAL",
    "LARGE_UNIQUE_BUILDINGS",
    "NUM_PLANTATION_STACKS",
    "PLANTATION_TILE_COUNTS",
    "PLANTATION_TILE_TOTAL",
    "PRODUCTION_BUILDINGS",
    "QUARRY_TILE_COUNT",
    "STARTING_DOUBLOONS_BY_PLAYER_COUNT",
    "STARTING_PLANTATIONS_BY_PLAYER_COUNT",
    "SUPPORTED_PLAYER_COUNTS",
    "TOTAL_DOUBLOON_VALUE",
    "VP_CHIP_FIVES",
    "VP_CHIP_ONES",
    "VP_CHIPS_PHYSICAL_TOTAL_VALUE",
    "VP_SUPPLY_TOTAL_BY_PLAYER_COUNT",
    "VIOLET_BUILDINGS",
    "VIOLET_SMALL_BUILDINGS",
    "BuildingCategory",
    "BuildingSpec",
    "face_up_plantation_count",
    "roles_for_player_count",
]
