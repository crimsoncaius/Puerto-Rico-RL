"""PettingZoo AEC environment wrapping ``PuertoRicoEngine`` with structured semantic actions."""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Any, Final, Optional

import numpy as np
from gymnasium.spaces import Box, Dict as GymDict, Discrete, Space
from pettingzoo import AECEnv

from .constants import BUILDING_METADATA, COLONIST_DISCS_TOTAL, face_up_plantation_count
from .engine import (
    BuilderBuild,
    BuilderNoOp,
    BuilderPass,
    CaptainLoad,
    CaptainPassLoading,
    CaptainStorageCommit,
    CaptainUseWharf,
    CraftsmanTurn,
    EngineAction,
    MayorPrivilegeSkip,
    MayorPrivilegeTake,
    MayorSubmitPlacement,
    PickRole,
    ProspectorCollect,
    PuertoRicoEngine,
    SettlerPass,
    SettlerTakeFaceUp,
    SettlerTakeHacienda,
    SettlerTakeQuarryConstructionHut,
    SettlerTakeQuarryPrivilege,
    TraderPass,
    TraderSell,
)
from .state import (
    Building,
    GameState,
    Good,
    IslandTile,
    MayorPhasePending,
    Phase,
    PlayerState,
    Role,
    building_worker_circles,
    goods_dict,
    island_tile_max_colonists,
    normalize_goods_counts,
    total_colonists_on_board,
)

# ---------------------------------------------------------------------------
# Encodings (stable int IDs for numpy observations)
# ---------------------------------------------------------------------------

_GOOD_ORDER: Final[tuple[Good, ...]] = tuple(sorted(Good, key=lambda g: g.value))
_GOOD_TO_I: Final[dict[Good, int]] = {g: i for i, g in enumerate(_GOOD_ORDER)}

_ISLAND_TILE_ORDER: Final[tuple[IslandTile, ...]] = tuple(sorted(IslandTile, key=lambda t: t.value))
_ISLAND_TILE_TO_I: Final[dict[IslandTile, int]] = {t: i for i, t in enumerate(_ISLAND_TILE_ORDER)}

_BUILDING_ORDER: Final[tuple[Building, ...]] = tuple(sorted(Building, key=lambda b: b.value))
_BUILDING_TO_I: Final[dict[Building, int]] = {b: i for i, b in enumerate(_BUILDING_ORDER)}

_ROLE_ORDER: Final[tuple[Role, ...]] = tuple(sorted(Role, key=lambda r: r.value))
_ROLE_TO_I: Final[dict[Role, int]] = {r: i for i, r in enumerate(_ROLE_ORDER)}

_PHASE_ORDER: Final[tuple[Phase, ...]] = (
    Phase.SETUP,
    Phase.ROLE_SELECTION,
    Phase.SETTLER,
    Phase.MAYOR,
    Phase.BUILDER,
    Phase.CRAFTSMAN,
    Phase.TRADER,
    Phase.CAPTAIN,
    Phase.PROSPECTOR,
    Phase.ROUND_CLEANUP,
    Phase.GAME_OVER,
)
_PHASE_TO_I: Final[dict[Phase, int]] = {p: i for i, p in enumerate(_PHASE_ORDER)}

_MAX_FACE_UP: Final[int] = face_up_plantation_count(5)
_MAX_KEEP_PER_GOOD: Final[int] = 12
_MAX_ISLAND_COLONISTS: Final[int] = max(island_tile_max_colonists(tile) for tile in IslandTile)
_MAX_BUILDING_WORKERS: Final[int] = max(building_worker_circles(building) for building in Building)

# ---------------------------------------------------------------------------
# Structured action branches
# ---------------------------------------------------------------------------

DECISION_TYPES: Final[tuple[str, ...]] = (
    "role",
    "settler",
    "mayor_privilege",
    "mayor_placement",
    "builder",
    "craftsman",
    "trader",
    "captain_loading",
    "captain_storage",
    "prospector",
)

_SETTLER_PASS_INDEX: Final[int] = 0
_SETTLER_HACIENDA_INDEX: Final[int] = 1
_SETTLER_QUARRY_PRIVILEGE_INDEX: Final[int] = 2
_SETTLER_QUARRY_HUT_INDEX: Final[int] = 3
_SETTLER_FACE_UP_OFFSET: Final[int] = 4
_SETTLER_CHOICE_COUNT: Final[int] = _SETTLER_FACE_UP_OFFSET + _MAX_FACE_UP

_MAYOR_PRIVILEGE_TAKE_INDEX: Final[int] = 0
_MAYOR_PRIVILEGE_SKIP_INDEX: Final[int] = 1
_MAYOR_PRIVILEGE_CHOICE_COUNT: Final[int] = 2

_BUILDER_BUILD_COUNT: Final[int] = len(_BUILDING_ORDER) * 12
_BUILDER_PASS_INDEX: Final[int] = _BUILDER_BUILD_COUNT
_BUILDER_NOOP_INDEX: Final[int] = _BUILDER_BUILD_COUNT + 1
_BUILDER_CHOICE_COUNT: Final[int] = _BUILDER_NOOP_INDEX + 1

_CRAFTSMAN_NONE_INDEX: Final[int] = 0
_CRAFTSMAN_CHOICE_COUNT: Final[int] = len(_GOOD_ORDER) + 1

_TRADER_PASS_INDEX: Final[int] = 0
_TRADER_CHOICE_COUNT: Final[int] = len(_GOOD_ORDER) + 1

_CAPTAIN_LOAD_COUNT: Final[int] = len(_GOOD_ORDER) * 3
_CAPTAIN_WHARF_OFFSET: Final[int] = _CAPTAIN_LOAD_COUNT
_CAPTAIN_PASS_INDEX: Final[int] = _CAPTAIN_WHARF_OFFSET + len(_GOOD_ORDER)
_CAPTAIN_LOADING_CHOICE_COUNT: Final[int] = _CAPTAIN_PASS_INDEX + 1

_PROSPECTOR_CHOICE_COUNT: Final[int] = 1


def _good_vec(counts: Any) -> np.ndarray:
    d = goods_dict(counts)
    return np.array([d.get(g, 0) for g in _GOOD_ORDER], dtype=np.int32)


def _island_tiles_vec(tiles: tuple[IslandTile, ...], pad: int) -> np.ndarray:
    out = np.full((pad,), -1, dtype=np.int32)
    for i, tile in enumerate(tiles[:pad]):
        out[i] = _ISLAND_TILE_TO_I[tile]
    return out


def _building_supply_vec(st: GameState) -> np.ndarray:
    d = dict(st.building_supply)
    return np.array([d.get(b, 0) for b in _BUILDING_ORDER], dtype=np.int32)


def _role_doubloon_vec(st: GameState) -> np.ndarray:
    d = {role: count for role, count in st.role_card_doubloons}
    return np.array([d.get(role, 0) for role in _ROLE_ORDER], dtype=np.int32)


def _current_role_index(st: GameState) -> int:
    if st.current_role_execution_index is None or not st.round_role_order:
        return -1
    idx = st.current_role_execution_index
    if idx < 0 or idx >= len(st.round_role_order):
        return -1
    role, _chooser = st.round_role_order[idx]
    return _ROLE_TO_I[role]


def _cargo_matrix(st: GameState) -> np.ndarray:
    rows: list[list[int]] = []
    for ship in st.cargo_ships:
        good_id = -1 if ship.good is None else _GOOD_TO_I[ship.good]
        rows.append([ship.capacity, good_id, ship.barrels])
    return np.array(rows, dtype=np.int32)


def _trading_house_vec(trading_house: Any) -> np.ndarray:
    out = np.full((4,), -1, dtype=np.int32)
    for i, good in enumerate(trading_house.goods[:4]):
        out[i] = _GOOD_TO_I[good]
    return out


def _encode_city_buildings(player: PlayerState) -> np.ndarray:
    out = np.full((12, 5), -1, dtype=np.int32)
    for i, placed in enumerate(player.city_buildings[:12]):
        out[i, 0] = _BUILDING_TO_I[placed.building]
        out[i, 1] = placed.anchor_slot
        for j, colonist in enumerate(placed.colonists[:3]):
            out[i, 2 + j] = colonist
    return out.reshape(-1)


def _encode_island(player: PlayerState) -> np.ndarray:
    out = np.zeros((12, 3), dtype=np.int32)
    for i, space in enumerate(player.island_spaces):
        out[i, 0] = -1 if space.tile is None else _ISLAND_TILE_TO_I[space.tile]
        out[i, 1] = space.colonists
        out[i, 2] = 0 if space.tile is None else island_tile_max_colonists(space.tile)
    return out.reshape(-1)


def _total_vp(player: PlayerState) -> int:
    return int(player.vp_from_chips + player.vp_on_paper)


def _observation_vector(st: GameState, obs_idx: int, acting: Optional[int]) -> np.ndarray:
    player = st.players[obs_idx]
    parts = [
        np.array([_PHASE_TO_I.get(st.phase, 0)], dtype=np.int32),
        np.array([st.round_number], dtype=np.int32),
        np.array([st.governor_index], dtype=np.int32),
        np.array([acting if acting is not None else -1], dtype=np.int32),
        np.array([player.doubloons], dtype=np.int32),
        np.array([_total_vp(player)], dtype=np.int32),
        np.array([player.vp_chips_1], dtype=np.int32),
        np.array([player.vp_chips_5], dtype=np.int32),
        np.array([player.san_juan_colonists], dtype=np.int32),
        np.array([total_colonists_on_board(player)], dtype=np.int32),
        _good_vec(player.goods),
        _encode_city_buildings(player),
        _encode_island(player),
        _island_tiles_vec(st.face_up_plantations, _MAX_FACE_UP),
        _good_vec(st.goods_supply),
        _building_supply_vec(st),
        np.array([st.colonist_ship], dtype=np.int32),
        np.array([st.colonist_supply], dtype=np.int32),
        _cargo_matrix(st).reshape(-1),
        _trading_house_vec(st.trading_house),
        _role_doubloon_vec(st),
        np.array([_current_role_index(st)], dtype=np.int32),
    ]
    return np.concatenate(parts).astype(np.int32, copy=False)


_OBSERVATION_VECTOR_SIZE: Final[int] = len(
    _observation_vector(PuertoRicoEngine().state, obs_idx=0, acting=None)
)


@lru_cache(maxsize=1)
def build_captain_storage_registry() -> tuple[CaptainStorageCommit, ...]:
    actions: list[CaptainStorageCommit] = []
    for counts in product(range(_MAX_KEEP_PER_GOOD + 1), repeat=len(_GOOD_ORDER)):
        raw = {_GOOD_ORDER[i]: counts[i] for i in range(len(_GOOD_ORDER))}
        actions.append(CaptainStorageCommit(keep_counts=normalize_goods_counts(raw)))
    return tuple(actions)


_CAPTAIN_STORAGE_REGISTRY: Final[tuple[CaptainStorageCommit, ...]] = build_captain_storage_registry()
_CAPTAIN_STORAGE_TO_INDEX: Final[dict[CaptainStorageCommit, int]] = {
    action: idx for idx, action in enumerate(_CAPTAIN_STORAGE_REGISTRY)
}


def _settler_choice_index(action: EngineAction) -> int:
    if isinstance(action, SettlerPass):
        return _SETTLER_PASS_INDEX
    if isinstance(action, SettlerTakeHacienda):
        return _SETTLER_HACIENDA_INDEX
    if isinstance(action, SettlerTakeQuarryPrivilege):
        return _SETTLER_QUARRY_PRIVILEGE_INDEX
    if isinstance(action, SettlerTakeQuarryConstructionHut):
        return _SETTLER_QUARRY_HUT_INDEX
    if isinstance(action, SettlerTakeFaceUp):
        return _SETTLER_FACE_UP_OFFSET + action.face_up_index
    raise TypeError(f"unsupported settler action {action!r}")


def _decode_settler_choice(index: int) -> EngineAction:
    if index == _SETTLER_PASS_INDEX:
        return SettlerPass()
    if index == _SETTLER_HACIENDA_INDEX:
        return SettlerTakeHacienda()
    if index == _SETTLER_QUARRY_PRIVILEGE_INDEX:
        return SettlerTakeQuarryPrivilege()
    if index == _SETTLER_QUARRY_HUT_INDEX:
        return SettlerTakeQuarryConstructionHut()
    return SettlerTakeFaceUp(index - _SETTLER_FACE_UP_OFFSET)


def _builder_choice_index(action: EngineAction) -> int:
    if isinstance(action, BuilderBuild):
        return _BUILDING_TO_I[action.building] * 12 + action.anchor_slot
    if isinstance(action, BuilderPass):
        return _BUILDER_PASS_INDEX
    if isinstance(action, BuilderNoOp):
        return _BUILDER_NOOP_INDEX
    raise TypeError(f"unsupported builder action {action!r}")


def _decode_builder_choice(index: int) -> EngineAction:
    if index == _BUILDER_PASS_INDEX:
        return BuilderPass()
    if index == _BUILDER_NOOP_INDEX:
        return BuilderNoOp()
    building = _BUILDING_ORDER[index // 12]
    anchor_slot = index % 12
    return BuilderBuild(building=building, anchor_slot=anchor_slot)


def _craftsman_choice_index(action: CraftsmanTurn) -> int:
    if action.privilege_good is None:
        return _CRAFTSMAN_NONE_INDEX
    return 1 + _GOOD_TO_I[action.privilege_good]


def _decode_craftsman_choice(index: int) -> CraftsmanTurn:
    privilege_good = None if index == _CRAFTSMAN_NONE_INDEX else _GOOD_ORDER[index - 1]
    return CraftsmanTurn(privilege_good=privilege_good, hacienda_good=None)


def _trader_choice_index(action: EngineAction) -> int:
    if isinstance(action, TraderPass):
        return _TRADER_PASS_INDEX
    if isinstance(action, TraderSell):
        return 1 + _GOOD_TO_I[action.good]
    raise TypeError(f"unsupported trader action {action!r}")


def _decode_trader_choice(index: int) -> EngineAction:
    if index == _TRADER_PASS_INDEX:
        return TraderPass()
    return TraderSell(_GOOD_ORDER[index - 1])


def _captain_loading_choice_index(action: EngineAction) -> int:
    if isinstance(action, CaptainLoad):
        return _GOOD_TO_I[action.good] * 3 + action.ship_index
    if isinstance(action, CaptainUseWharf):
        return _CAPTAIN_WHARF_OFFSET + _GOOD_TO_I[action.good]
    if isinstance(action, CaptainPassLoading):
        return _CAPTAIN_PASS_INDEX
    raise TypeError(f"unsupported captain loading action {action!r}")


def _decode_captain_loading_choice(index: int) -> EngineAction:
    if index == _CAPTAIN_PASS_INDEX:
        return CaptainPassLoading()
    if index >= _CAPTAIN_WHARF_OFFSET:
        return CaptainUseWharf(_GOOD_ORDER[index - _CAPTAIN_WHARF_OFFSET])
    return CaptainLoad(good=_GOOD_ORDER[index // 3], ship_index=index % 3)


def _mayor_capacity_vectors(player: PlayerState) -> tuple[np.ndarray, np.ndarray]:
    island_capacity = np.zeros((12,), dtype=np.int32)
    for idx, space in enumerate(player.island_spaces):
        island_capacity[idx] = 0 if space.tile is None else island_tile_max_colonists(space.tile)

    building_capacity = np.zeros((12,), dtype=np.int32)
    for idx, placed in enumerate(player.city_buildings[:12]):
        building_capacity[idx] = len(placed.colonists)
    return island_capacity, building_capacity


def _masked_allocation_sample(
    rng: np.random.Generator,
    capacities: np.ndarray,
    total: int,
) -> np.ndarray:
    alloc = np.zeros_like(capacities, dtype=np.int32)
    if total <= 0:
        return alloc

    order = np.arange(len(capacities))
    rng.shuffle(order)
    remaining = int(total)
    suffix_capacity = np.zeros((len(order) + 1,), dtype=np.int32)
    for pos in range(len(order) - 1, -1, -1):
        suffix_capacity[pos] = suffix_capacity[pos + 1] + int(capacities[order[pos]])

    for pos, idx in enumerate(order):
        capacity = int(capacities[idx])
        if pos == len(order) - 1:
            alloc[idx] = remaining
            break
        remaining_capacity = int(suffix_capacity[pos + 1])
        minimum_here = max(0, remaining - remaining_capacity)
        maximum_here = min(capacity, remaining)
        alloc[idx] = int(rng.integers(minimum_here, maximum_here + 1))
        remaining -= int(alloc[idx])
    return alloc


class MayorPlacementSpace(Space[dict[str, Any]]):
    """Custom space for mayor final-allocation submissions."""

    def __init__(self) -> None:
        super().__init__(shape=None, dtype=None)

    def sample(self, mask: dict[str, Any] | None = None, probability: Any | None = None) -> dict[str, Any]:
        if probability is not None:
            raise ValueError("MayorPlacementSpace only supports mask sampling")
        if mask is None:
            total_pool = 0
            island_capacity = np.zeros((12,), dtype=np.int32)
            building_capacity = np.zeros((12,), dtype=np.int32)
        else:
            total_pool = int(np.asarray(mask["total_pool"], dtype=np.int32).reshape(-1)[0])
            island_capacity = np.asarray(mask["island_capacity"], dtype=np.int32).reshape(12)
            building_capacity = np.asarray(mask["building_capacity"], dtype=np.int32).reshape(12)

        board_capacity = np.concatenate((island_capacity, building_capacity)).astype(np.int32, copy=False)
        max_on_board = int(board_capacity.sum())
        if total_pool >= max_on_board:
            board_alloc = board_capacity.copy()
        else:
            board_alloc = _masked_allocation_sample(self.np_random, board_capacity, total_pool)

        return {
            "island": board_alloc[:12].astype(np.int32),
            "buildings": board_alloc[12:].astype(np.int32),
            "san_juan": int(total_pool - int(board_alloc.sum())),
        }

    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict):
            return False
        if set(x) != {"island", "buildings", "san_juan"}:
            return False
        island = np.asarray(x["island"], dtype=np.int32)
        buildings = np.asarray(x["buildings"], dtype=np.int32)
        san_juan = int(np.asarray(x["san_juan"], dtype=np.int32).reshape(-1)[0])
        return (
            island.shape == (12,)
            and buildings.shape == (12,)
            and np.all((0 <= island) & (island <= _MAX_ISLAND_COLONISTS))
            and np.all((0 <= buildings) & (buildings <= _MAX_BUILDING_WORKERS))
            and san_juan >= 0
        )


def _action_mask_space() -> GymDict:
    return GymDict(
        {
            "role": Box(0, 2, shape=(len(_ROLE_ORDER),), dtype=np.int8),
            "settler": Box(0, 2, shape=(_SETTLER_CHOICE_COUNT,), dtype=np.int8),
            "mayor_privilege": Box(0, 2, shape=(_MAYOR_PRIVILEGE_CHOICE_COUNT,), dtype=np.int8),
            "mayor_placement": GymDict(
                {
                    "total_pool": Box(0, COLONIST_DISCS_TOTAL + 1, shape=(1,), dtype=np.int32),
                    "island_capacity": Box(
                        0,
                        _MAX_ISLAND_COLONISTS + 1,
                        shape=(12,),
                        dtype=np.int32,
                    ),
                    "building_capacity": Box(
                        0,
                        _MAX_BUILDING_WORKERS + 1,
                        shape=(12,),
                        dtype=np.int32,
                    ),
                }
            ),
            "builder": Box(0, 2, shape=(_BUILDER_CHOICE_COUNT,), dtype=np.int8),
            "craftsman": Box(0, 2, shape=(_CRAFTSMAN_CHOICE_COUNT,), dtype=np.int8),
            "trader": Box(0, 2, shape=(_TRADER_CHOICE_COUNT,), dtype=np.int8),
            "captain_loading": Box(0, 2, shape=(_CAPTAIN_LOADING_CHOICE_COUNT,), dtype=np.int8),
            "captain_storage": Box(0, 2, shape=(len(_CAPTAIN_STORAGE_REGISTRY),), dtype=np.int8),
            "prospector": Box(0, 2, shape=(_PROSPECTOR_CHOICE_COUNT,), dtype=np.int8),
        }
    )


def _acting_player_id(engine: PuertoRicoEngine) -> Optional[int]:
    return engine.acting_player()


def _require_index(value: Any, size: int, field_name: str) -> int:
    index = int(value)
    if index < 0 or index >= size:
        raise ValueError(f"{field_name} out of range: {index}")
    return index


def _require_count_tuple(value: Any, length: int, field_name: str) -> tuple[int, ...]:
    arr = np.asarray(value, dtype=np.int32)
    if arr.shape != (length,):
        raise ValueError(f"{field_name} must have shape {(length,)}, got {arr.shape}")
    return tuple(int(v) for v in arr.tolist())


class PuertoRicoEnv(AECEnv):
    """PettingZoo AEC environment with phase-structured semantic actions."""

    metadata = {
        "render_modes": ["ansi"],
        "name": "puerto_rico_v0",
        "is_parallelizable": False,
    }

    def __init__(
        self,
        num_players: int = 3,
        render_mode: Optional[str] = None,
        reward_mode: str = "vp_delta",
    ) -> None:
        super().__init__()
        if num_players not in (3, 4, 5):
            raise ValueError("num_players must be 3, 4, or 5")
        self._num_players = num_players
        self.render_mode = render_mode
        self.reward_mode = reward_mode

        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self.agents: list[str] = []
        self._engine = PuertoRicoEngine()

        self._action_space = GymDict(
            {
                "role": Discrete(len(_ROLE_ORDER)),
                "settler": Discrete(_SETTLER_CHOICE_COUNT),
                "mayor_privilege": Discrete(_MAYOR_PRIVILEGE_CHOICE_COUNT),
                "mayor_placement": MayorPlacementSpace(),
                "builder": Discrete(_BUILDER_CHOICE_COUNT),
                "craftsman": Discrete(_CRAFTSMAN_CHOICE_COUNT),
                "trader": Discrete(_TRADER_CHOICE_COUNT),
                "captain_loading": Discrete(_CAPTAIN_LOADING_CHOICE_COUNT),
                "captain_storage": Discrete(len(_CAPTAIN_STORAGE_REGISTRY)),
                "prospector": Discrete(_PROSPECTOR_CHOICE_COUNT),
            }
        )
        self._observation_space = GymDict(
            {
                "observation": Box(-1, 100_000, shape=(_OBSERVATION_VECTOR_SIZE,), dtype=np.int32),
                "action_mask": _action_mask_space(),
            }
        )
        self.observation_spaces = {agent: self._observation_space for agent in self.possible_agents}
        self.action_spaces = {agent: self._action_space for agent in self.possible_agents}

        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}

    def observation_space(self, agent: str) -> GymDict:
        return self._observation_space

    def action_space(self, agent: str) -> GymDict:
        return self._action_space

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        self._engine.reset(self._num_players, seed=seed)
        self.agents = self.possible_agents[:]
        for agent in self.possible_agents:
            self.rewards[agent] = 0.0
            self._cumulative_rewards[agent] = 0.0
            self.terminations[agent] = False
            self.truncations[agent] = False
            self.infos[agent] = {}
        acting = _acting_player_id(self._engine)
        self.agent_selection = self.possible_agents[acting] if acting is not None else self.possible_agents[0]

    def _empty_action_mask(self) -> dict[str, Any]:
        return {
            "role": np.zeros((len(_ROLE_ORDER),), dtype=np.int8),
            "settler": np.zeros((_SETTLER_CHOICE_COUNT,), dtype=np.int8),
            "mayor_privilege": np.zeros((_MAYOR_PRIVILEGE_CHOICE_COUNT,), dtype=np.int8),
            "mayor_placement": {
                "total_pool": np.zeros((1,), dtype=np.int32),
                "island_capacity": np.zeros((12,), dtype=np.int32),
                "building_capacity": np.zeros((12,), dtype=np.int32),
            },
            "builder": np.zeros((_BUILDER_CHOICE_COUNT,), dtype=np.int8),
            "craftsman": np.zeros((_CRAFTSMAN_CHOICE_COUNT,), dtype=np.int8),
            "trader": np.zeros((_TRADER_CHOICE_COUNT,), dtype=np.int8),
            "captain_loading": np.zeros((_CAPTAIN_LOADING_CHOICE_COUNT,), dtype=np.int8),
            "captain_storage": np.zeros((len(_CAPTAIN_STORAGE_REGISTRY),), dtype=np.int8),
            "prospector": np.zeros((_PROSPECTOR_CHOICE_COUNT,), dtype=np.int8),
        }

    def observe(self, agent: str) -> dict[str, Any]:
        st = self._engine.state
        obs_idx = int(agent.split("_", 1)[1])
        acting = _acting_player_id(self._engine)
        return {
            "observation": _observation_vector(st, obs_idx, acting),
            "action_mask": self._compute_action_mask(obs_idx, acting),
        }

    def _compute_action_mask(self, obs_idx: int, acting: Optional[int]) -> dict[str, Any]:
        mask = self._empty_action_mask()
        if acting is None or obs_idx != acting:
            return mask

        st = self._engine.state
        if st.phase is Phase.MAYOR and isinstance(st.pending, MayorPhasePending) and st.pending.subphase == "placement":
            pools = st.pending.placement_pools
            pool = pools[obs_idx] if obs_idx < len(pools) else 0
            island_capacity, building_capacity = _mayor_capacity_vectors(st.players[obs_idx])
            mask["mayor_placement"] = {
                "total_pool": np.array([pool], dtype=np.int32),
                "island_capacity": island_capacity,
                "building_capacity": building_capacity,
            }
            return mask

        legal = self._engine.legal_actions(obs_idx)
        for action in legal:
            if isinstance(action, PickRole):
                mask["role"][_ROLE_TO_I[action.role]] = 1
            elif isinstance(
                action,
                (
                    SettlerPass,
                    SettlerTakeHacienda,
                    SettlerTakeQuarryPrivilege,
                    SettlerTakeQuarryConstructionHut,
                    SettlerTakeFaceUp,
                ),
            ):
                mask["settler"][_settler_choice_index(action)] = 1
            elif isinstance(action, MayorPrivilegeTake):
                mask["mayor_privilege"][_MAYOR_PRIVILEGE_TAKE_INDEX] = 1
            elif isinstance(action, MayorPrivilegeSkip):
                mask["mayor_privilege"][_MAYOR_PRIVILEGE_SKIP_INDEX] = 1
            elif isinstance(action, (BuilderBuild, BuilderPass, BuilderNoOp)):
                mask["builder"][_builder_choice_index(action)] = 1
            elif isinstance(action, CraftsmanTurn):
                mask["craftsman"][_craftsman_choice_index(action)] = 1
            elif isinstance(action, (TraderPass, TraderSell)):
                mask["trader"][_trader_choice_index(action)] = 1
            elif isinstance(action, (CaptainLoad, CaptainUseWharf, CaptainPassLoading)):
                mask["captain_loading"][_captain_loading_choice_index(action)] = 1
            elif isinstance(action, CaptainStorageCommit):
                mask["captain_storage"][_CAPTAIN_STORAGE_TO_INDEX[action]] = 1
            elif isinstance(action, ProspectorCollect):
                mask["prospector"][0] = 1
            else:
                raise RuntimeError(f"Unhandled legal engine action for env mask: {action!r}")
        return mask

    def _decode_action(self, action: Any) -> EngineAction:
        if not isinstance(action, dict):
            raise ValueError("structured action must be a dict")

        st = self._engine.state
        if st.phase is Phase.ROLE_SELECTION:
            role_index = _require_index(action["role"], len(_ROLE_ORDER), "role")
            return PickRole(_ROLE_ORDER[role_index])

        if st.phase is Phase.SETTLER:
            choice = _require_index(action["settler"], _SETTLER_CHOICE_COUNT, "settler")
            return _decode_settler_choice(choice)

        if st.phase is Phase.MAYOR and isinstance(st.pending, MayorPhasePending):
            if st.pending.subphase == "privilege":
                choice = _require_index(
                    action["mayor_privilege"],
                    _MAYOR_PRIVILEGE_CHOICE_COUNT,
                    "mayor_privilege",
                )
                return MayorPrivilegeTake() if choice == _MAYOR_PRIVILEGE_TAKE_INDEX else MayorPrivilegeSkip()

            placement = action["mayor_placement"]
            if not isinstance(placement, dict):
                raise ValueError("mayor_placement must be a dict")
            island_targets = _require_count_tuple(placement["island"], 12, "mayor_placement.island")
            building_targets = _require_count_tuple(placement["buildings"], 12, "mayor_placement.buildings")
            san_juan = int(np.asarray(placement["san_juan"], dtype=np.int32).reshape(-1)[0])
            return MayorSubmitPlacement(
                island_targets=island_targets,
                building_targets=building_targets,
                san_juan=san_juan,
            )

        if st.phase is Phase.BUILDER:
            choice = _require_index(action["builder"], _BUILDER_CHOICE_COUNT, "builder")
            return _decode_builder_choice(choice)

        if st.phase is Phase.CRAFTSMAN:
            choice = _require_index(action["craftsman"], _CRAFTSMAN_CHOICE_COUNT, "craftsman")
            return _decode_craftsman_choice(choice)

        if st.phase is Phase.TRADER:
            choice = _require_index(action["trader"], _TRADER_CHOICE_COUNT, "trader")
            return _decode_trader_choice(choice)

        if st.phase is Phase.CAPTAIN and st.pending is not None:
            if st.pending.subphase == "loading":
                choice = _require_index(
                    action["captain_loading"],
                    _CAPTAIN_LOADING_CHOICE_COUNT,
                    "captain_loading",
                )
                return _decode_captain_loading_choice(choice)
            choice = _require_index(
                action["captain_storage"],
                len(_CAPTAIN_STORAGE_REGISTRY),
                "captain_storage",
            )
            return _CAPTAIN_STORAGE_REGISTRY[choice]

        if st.phase is Phase.PROSPECTOR:
            _require_index(action["prospector"], _PROSPECTOR_CHOICE_COUNT, "prospector")
            return ProspectorCollect()

        raise RuntimeError(f"cannot decode action in phase {st.phase}")

    def step(self, action: Any) -> None:
        if self.agent_selection is None:
            return

        agent = self.agent_selection
        if self.truncations[agent] or self.terminations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0
        for a in self.possible_agents:
            self.rewards[a] = 0.0

        acting = _acting_player_id(self._engine)
        if acting is None:
            raise RuntimeError("no legal actor but game not terminal")

        player_id = int(agent.split("_", 1)[1])
        if player_id != acting:
            raise RuntimeError(f"expected acting player {acting}, got {player_id}")

        eng_action = self._decode_action(action)
        if not self._engine.is_legal(player_id, eng_action):
            raise ValueError(f"illegal action for {agent}: {eng_action!r}")

        vp_before = _total_vp(self._engine.state.players[player_id])
        self._engine.apply(player_id, eng_action)
        vp_after = _total_vp(self._engine.state.players[player_id])

        if self.reward_mode == "vp_delta":
            self.rewards[agent] = float(vp_after - vp_before)
        else:
            self.rewards[agent] = 0.0

        terminal = self._engine.is_terminal()
        if terminal:
            for a in self.agents:
                self.terminations[a] = True
            self.agent_selection = self.agents[0] if self.agents else self.possible_agents[0]
        else:
            for a in self.possible_agents:
                self.truncations[a] = False
            nxt = _acting_player_id(self._engine)
            if nxt is None:
                raise RuntimeError("expected next actor after non-terminal step")
            self.agent_selection = self.possible_agents[nxt]

        for a in self.possible_agents:
            self.infos[a] = {}

        self._accumulate_rewards()

    def render(self) -> Optional[str]:
        if self.render_mode != "ansi":
            return None
        st = self._engine.state
        lines = [
            f"Puerto Rico | phase={st.phase.value} round={st.round_number} governor={st.governor_index}",
            f"Face-up: {[tile.value for tile in st.face_up_plantations]}",
            f"Goods supply: {dict(st.goods_supply)}",
            f"Colonists ship={st.colonist_ship} supply={st.colonist_supply}",
            f"VP supply: {st.vp_supply}",
        ]
        for i, player in enumerate(st.players):
            lines.append(
                f"  P{i}: $={player.doubloons} VP={_total_vp(player)} goods={dict(player.goods)} "
                f"buildings={len(player.city_buildings)}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        pass


__all__ = ["PuertoRicoEnv", "DECISION_TYPES", "build_captain_storage_registry", "_CAPTAIN_STORAGE_REGISTRY"]
