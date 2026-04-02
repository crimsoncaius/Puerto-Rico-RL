"""PettingZoo AEC environment wrapping ``PuertoRicoEngine``."""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Any, Final, Optional, Sequence, Union

import numpy as np

from gymnasium.spaces import Box, Dict, Discrete

from pettingzoo import AECEnv

from .constants import BUILDING_METADATA, face_up_plantation_count, roles_for_player_count
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
    MayorDraftTake,
    MayorPlaceColonistBuilding,
    MayorPlaceColonistIsland,
    MayorPlaceColonistSanJuan,
    MayorPrivilegeSkip,
    MayorPrivilegeTake,
    PickRole,
    ProspectorCollect,
    PuertoRicoEngine,
    RoundCleanupAdvance,
    SettlerPass,
    SettlerTakeHacienda,
    SettlerTakeFaceUp,
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
    Phase,
    PlayerState,
    Role,
    good_count,
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

# Face-up market: at most (5 + 1) tiles in a 5-player game
_MAX_FACE_UP: Final[int] = face_up_plantation_count(5)

# Captain storage: upper bound per good on the windrose (registry coverage)
_MAX_KEEP_PER_GOOD: Final[int] = 12  # 0..12 inclusive → 13 values per good


def _good_vec(counts: Any) -> np.ndarray:
    d = goods_dict(counts)
    return np.array([d.get(g, 0) for g in _GOOD_ORDER], dtype=np.int32)


def _island_tiles_vec(tiles: Sequence[IslandTile], pad: int) -> np.ndarray:
    out = np.full((pad,), -1, dtype=np.int32)
    for i, t in enumerate(tiles[:pad]):
        out[i] = _ISLAND_TILE_TO_I[t]
    return out


def _building_supply_vec(st: GameState) -> np.ndarray:
    d = dict(st.building_supply)
    return np.array([d.get(b, 0) for b in _BUILDING_ORDER], dtype=np.int32)


def _role_doubloon_vec(st: GameState) -> np.ndarray:
    m = {r: n for r, n in st.role_card_doubloons}
    return np.array([m.get(r, 0) for r in _ROLE_ORDER], dtype=np.int32)


def _current_role_index(st: GameState) -> int:
    if st.current_role_execution_index is None or not st.round_role_order:
        return -1
    idx = st.current_role_execution_index
    if idx < 0 or idx >= len(st.round_role_order):
        return -1
    role, _chooser = st.round_role_order[idx]
    return _ROLE_TO_I[role]


def _cargo_matrix(st: GameState) -> np.ndarray:
    """Shape (3, 3): capacity, good_id (-1 if none), barrels."""
    rows = []
    for sh in st.cargo_ships:
        gid = -1 if sh.good is None else _GOOD_TO_I[sh.good]
        rows.append([sh.capacity, gid, sh.barrels])
    return np.array(rows, dtype=np.int32)


def _trading_house_vec(h) -> np.ndarray:
    g = h.goods
    out = np.full((4,), -1, dtype=np.int32)
    for i, gg in enumerate(g[:4]):
        out[i] = _GOOD_TO_I[gg]
    return out


def _encode_city_buildings(p: PlayerState) -> np.ndarray:
    """Fixed 12 rows × 5: building_id, anchor, up to 3 colonist slots (padded -1)."""
    out = np.full((12, 5), -1, dtype=np.int32)
    for i, pb in enumerate(p.city_buildings[:12]):
        out[i, 0] = _BUILDING_TO_I[pb.building]
        out[i, 1] = pb.anchor_slot
        for j, c in enumerate(pb.colonists[:3]):
            out[i, 2 + j] = c
    return out.flatten()


def _encode_island(p: PlayerState) -> np.ndarray:
    """12 × 3: tile_id (-1 empty), colonists, max_colonists."""
    out = np.zeros((12, 3), dtype=np.int32)
    for i, sp in enumerate(p.island_spaces):
        if sp.tile is None:
            out[i, 0] = -1
        else:
            out[i, 0] = _ISLAND_TILE_TO_I[sp.tile]
        out[i, 1] = sp.colonists
        if sp.tile is not None:
            out[i, 2] = island_tile_max_colonists(sp.tile)
        else:
            out[i, 2] = 0
    return out.flatten()


# ---------------------------------------------------------------------------
# Action registry (fixed global table)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def build_action_registry() -> tuple[EngineAction, ...]:
    """All engine actions with discrete indices; masking marks legality."""
    actions: list[EngineAction] = []

    for r in sorted(roles_for_player_count(5), key=lambda x: x.value):
        actions.append(PickRole(r))

    actions.append(SettlerPass())
    actions.append(SettlerTakeHacienda())
    actions.append(SettlerTakeQuarryPrivilege())
    actions.append(SettlerTakeQuarryConstructionHut())
    for i in range(_MAX_FACE_UP):
        actions.append(SettlerTakeFaceUp(i))

    actions.append(MayorPrivilegeTake())
    actions.append(MayorPrivilegeSkip())
    actions.append(MayorDraftTake())
    for si in range(12):
        actions.append(MayorPlaceColonistIsland(si))
    for bi in range(12):
        for ci in range(3):
            actions.append(MayorPlaceColonistBuilding(bi, ci))
    actions.append(MayorPlaceColonistSanJuan())

    for b in sorted(BUILDING_METADATA.keys(), key=lambda x: x.value):
        for slot in range(12):
            actions.append(BuilderBuild(b, slot))
    actions.append(BuilderPass())
    actions.append(BuilderNoOp())

    priv_opts: list[Optional[Good]] = [None, *_GOOD_ORDER]
    hac_opts: list[Optional[Good]] = [None, *_GOOD_ORDER]
    for pg in priv_opts:
        for hg in hac_opts:
            actions.append(CraftsmanTurn(privilege_good=pg, hacienda_good=hg))

    actions.append(TraderPass())
    for g in _GOOD_ORDER:
        actions.append(TraderSell(g))

    for g in _GOOD_ORDER:
        for ship_i in range(3):
            actions.append(CaptainLoad(good=g, ship_index=ship_i))
    for g in _GOOD_ORDER:
        actions.append(CaptainUseWharf(good=g))
    actions.append(CaptainPassLoading())

    actions.append(ProspectorCollect())
    actions.append(RoundCleanupAdvance())

    for counts in product(range(_MAX_KEEP_PER_GOOD + 1), repeat=len(_GOOD_ORDER)):
        raw = {_GOOD_ORDER[i]: counts[i] for i in range(len(_GOOD_ORDER))}
        actions.append(CaptainStorageCommit(keep_counts=normalize_goods_counts(raw)))

    return tuple(actions)


_ACTION_REGISTRY: Final[tuple[EngineAction, ...]] = build_action_registry()
_ACTION_TO_INDEX: Final[dict[EngineAction, int]] = {a: i for i, a in enumerate(_ACTION_REGISTRY)}
assert len(_ACTION_TO_INDEX) == len(_ACTION_REGISTRY), "duplicate actions in registry"


def _acting_player_id(engine: PuertoRicoEngine) -> Optional[int]:
    if engine.is_terminal():
        return None
    n = engine.state.num_players
    for i in range(n):
        if engine.legal_actions(i):
            return i
    return None


def _total_vp(p: PlayerState) -> int:
    return int(p.vp_from_chips + p.vp_on_paper)


class PuertoRicoEnv(AECEnv):
    """PettingZoo ``AECEnv`` for Puerto Rico (engine-backed)."""

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

        self._observation_space = self._build_observation_space()
        self._action_space = Discrete(len(_ACTION_REGISTRY))
        self.observation_spaces = {a: self._observation_space for a in self.possible_agents}
        self.action_spaces = {a: self._action_space for a in self.possible_agents}

        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}

    # -- PettingZoo spaces ----------------------------------------------------

    def observation_space(self, agent: str) -> Dict:
        return self._observation_space

    def action_space(self, agent: str) -> Discrete:
        return self._action_space

    def _build_observation_space(self) -> Dict:
        n_roles = len(_ROLE_ORDER)
        # Gymnasium Box high bound is exclusive for int/float ranges.
        return Dict(
            {
                "phase": Box(0, len(_PHASE_ORDER), shape=(1,), dtype=np.int32),
                "round_number": Box(0, 10_000, shape=(1,), dtype=np.int32),
                "governor_index": Box(0, self._num_players, shape=(1,), dtype=np.int32),
                "current_agent": Box(-1, self._num_players, shape=(1,), dtype=np.int32),
                "action_mask": Box(0, 2, shape=(len(_ACTION_REGISTRY),), dtype=np.int8),
                "doubloons": Box(0, 10_000, shape=(1,), dtype=np.int32),
                "vp_total": Box(0, 500, shape=(1,), dtype=np.int32),
                "vp_chips_1": Box(0, 200, shape=(1,), dtype=np.int32),
                "vp_chips_5": Box(0, 200, shape=(1,), dtype=np.int32),
                "san_juan_colonists": Box(0, 100, shape=(1,), dtype=np.int32),
                "colonists_on_board": Box(0, 100, shape=(1,), dtype=np.int32),
                "goods": Box(0, 50, shape=(len(_GOOD_ORDER),), dtype=np.int32),
                "city_buildings": Box(-1, 100, shape=(12 * 5,), dtype=np.int32),
                "island": Box(-1, 100, shape=(12 * 3,), dtype=np.int32),
                "face_up_plantations": Box(-1, len(_ISLAND_TILE_ORDER), shape=(_MAX_FACE_UP,), dtype=np.int32),
                "goods_supply": Box(0, 100, shape=(len(_GOOD_ORDER),), dtype=np.int32),
                "building_supply": Box(0, 20, shape=(len(_BUILDING_ORDER),), dtype=np.int32),
                "colonist_ship": Box(0, 100, shape=(1,), dtype=np.int32),
                "colonist_supply": Box(0, 200, shape=(1,), dtype=np.int32),
                "cargo_ships": Box(-1, 100, shape=(3, 3), dtype=np.int32),
                "trading_house": Box(-1, len(_GOOD_ORDER), shape=(4,), dtype=np.int32),
                "role_card_doubloons": Box(0, 50, shape=(n_roles,), dtype=np.int32),
                "current_role": Box(-1, n_roles, shape=(1,), dtype=np.int32),
            }
        )

    # -- Core API ------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> None:
        self._engine.reset(self._num_players, seed=seed)
        self.agents = self.possible_agents[:]
        for a in self.possible_agents:
            self.rewards[a] = 0.0
            self._cumulative_rewards[a] = 0.0
            self.terminations[a] = False
            self.truncations[a] = False
            self.infos[a] = {}
        act = _acting_player_id(self._engine)
        self.agent_selection = self.possible_agents[act] if act is not None else self.possible_agents[0]

    def observe(self, agent: str) -> dict[str, Union[np.ndarray, list[int]]]:
        st = self._engine.state
        obs_idx = int(agent.split("_", 1)[1])
        acting = _acting_player_id(self._engine)
        mask = self._compute_action_mask(obs_idx, acting)

        p = st.players[obs_idx]
        obs: dict[str, Union[np.ndarray, list[int]]] = {
            "phase": np.array([_PHASE_TO_I.get(st.phase, 0)], dtype=np.int32),
            "round_number": np.array([st.round_number], dtype=np.int32),
            "governor_index": np.array([st.governor_index], dtype=np.int32),
            "current_agent": np.array([acting if acting is not None else -1], dtype=np.int32),
            "action_mask": np.array(mask, dtype=np.int8),
            "doubloons": np.array([p.doubloons], dtype=np.int32),
            "vp_total": np.array([_total_vp(p)], dtype=np.int32),
            "vp_chips_1": np.array([p.vp_chips_1], dtype=np.int32),
            "vp_chips_5": np.array([p.vp_chips_5], dtype=np.int32),
            "san_juan_colonists": np.array([p.san_juan_colonists], dtype=np.int32),
            "colonists_on_board": np.array([total_colonists_on_board(p)], dtype=np.int32),
            "goods": _good_vec(p.goods),
            "city_buildings": _encode_city_buildings(p),
            "island": _encode_island(p),
            "face_up_plantations": _island_tiles_vec(st.face_up_plantations, _MAX_FACE_UP),
            "goods_supply": _good_vec(st.goods_supply),
            "building_supply": _building_supply_vec(st),
            "colonist_ship": np.array([st.colonist_ship], dtype=np.int32),
            "colonist_supply": np.array([st.colonist_supply], dtype=np.int32),
            "cargo_ships": _cargo_matrix(st),
            "trading_house": _trading_house_vec(st.trading_house),
            "role_card_doubloons": _role_doubloon_vec(st),
            "current_role": np.array([_current_role_index(st)], dtype=np.int32),
        }
        return obs

    def _compute_action_mask(self, obs_idx: int, acting: Optional[int]) -> list[int]:
        n = len(_ACTION_REGISTRY)
        if acting is None or obs_idx != acting:
            return [0] * n
        legal = self._engine.legal_actions(obs_idx)
        mask = [0] * n
        missing: list[EngineAction] = []
        for a in legal:
            idx = _ACTION_TO_INDEX.get(a)
            if idx is not None:
                mask[idx] = 1
            else:
                missing.append(a)
        if missing:
            raise RuntimeError(
                f"Legal engine actions not in registry (update registry): {missing[:5]}"
            )
        return mask

    def step(self, action: Any) -> None:
        if self.agent_selection is None:
            return

        agent = self.agent_selection
        if self.truncations[agent] or self.terminations[agent]:
            self._was_dead_step(action)
            return

        idx = int(action)  # type: ignore[arg-type]
        if idx < 0 or idx >= len(_ACTION_REGISTRY):
            raise ValueError(f"action out of range: {idx}")

        self._cumulative_rewards[agent] = 0.0
        for a in self.possible_agents:
            self.rewards[a] = 0.0

        acting = _acting_player_id(self._engine)
        if acting is None:
            raise RuntimeError("no legal actor but game not terminal")

        pid = int(agent.split("_", 1)[1])
        if pid != acting:
            raise RuntimeError(f"expected acting player {acting}, got {pid}")

        mask = self._compute_action_mask(pid, acting)
        if not mask[idx]:
            raise ValueError(f"illegal action index {idx} for {agent}")

        eng_action = _ACTION_REGISTRY[idx]
        vp_before = _total_vp(self._engine.state.players[pid])
        self._engine.apply(pid, eng_action)
        vp_after = _total_vp(self._engine.state.players[pid])

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
            f"Face-up: {[t.value for t in st.face_up_plantations]}",
            f"Goods supply: {dict(st.goods_supply)}",
            f"Colonists ship={st.colonist_ship} supply={st.colonist_supply}",
            f"VP supply: {st.vp_supply}",
        ]
        for i, p in enumerate(st.players):
            lines.append(
                f"  P{i}: $={p.doubloons} VP={_total_vp(p)} goods={dict(p.goods)} "
                f"buildings={len(p.city_buildings)}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        pass


__all__ = ["PuertoRicoEnv", "build_action_registry", "_ACTION_REGISTRY"]
