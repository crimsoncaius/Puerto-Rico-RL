"""Tests for ``PuertoRicoEngine`` behavior (rules and scoring hooks)."""

from __future__ import annotations

import dataclasses
import random
from typing import Optional

import pytest

from puerto_rico.engine import (
    CaptainLoad,
    CaptainStorageCommit,
    MayorPrivilegeTake,
    PuertoRicoEngine,
    _wharf_available,
    _builder_doubloon_cost_base,
)
from puerto_rico.setup import initial_game_state
from puerto_rico.state import (
    Building,
    CaptainPhasePending,
    Good,
    IslandSpace,
    IslandTile,
    MayorPhasePending,
    Phase,
    PlacedBuilding,
    PlayerState,
    Role,
    good_count,
    goods_dict,
    normalize_goods_counts,
    role_selection_order,
)


def _empty_goods() -> tuple[tuple[Good, int], ...]:
    return normalize_goods_counts({})


def _make_player_with_quarries(
    *,
    doubloons: int,
    quarry_slots: list[tuple[int, bool]],
) -> PlayerState:
    """quarry_slots: (island_index, occupied) — QUARRY tiles with optional colonists."""
    island = [IslandSpace(tile=None, colonists=0) for _ in range(12)]
    for idx, occ in quarry_slots:
        island[idx] = IslandSpace(tile=IslandTile.QUARRY, colonists=1 if occ else 0)
    return PlayerState(
        doubloons=doubloons,
        vp_from_chips=0,
        vp_on_paper=0,
        san_juan_colonists=0,
        island_spaces=tuple(island),
        city_buildings=(),
        goods=_empty_goods(),
        vp_chips_1=0,
        vp_chips_5=0,
    )


def test_role_selection_order_governor_first_clockwise() -> None:
    """Governor picks first; ``next_role_selector_index`` follows clockwise."""
    eng = PuertoRicoEngine()
    eng.reset(4, seed=0)
    assert eng.state.phase is Phase.ROLE_SELECTION
    order = role_selection_order(eng.state.governor_index, eng.state.num_players)
    assert order == (0, 1, 2, 3)

    for pid in order:
        assert eng.state.next_role_selector_index == pid
        acts = eng.legal_actions(pid)
        assert acts
        eng.apply(pid, acts[0])

    assert eng.state.next_role_selector_index is None
    assert len(eng.state.round_role_order) == 4
    assert eng.state.phase is not Phase.ROLE_SELECTION


def test_round_cleanup_unused_roles_gain_one_doubloon() -> None:
    """After one full round, each unchosen role has +1 doubloon on its stack."""
    rng = random.Random(202)
    eng = PuertoRicoEngine()
    eng.reset(3, seed=11)

    max_steps = 50_000
    steps = 0
    while steps < max_steps:
        if eng.is_terminal():
            pytest.skip("Game ended before completing one round")
        st = eng.state
        if st.round_number == 2 and st.phase is Phase.ROLE_SELECTION:
            break
        actor = _first_actor(eng)
        if actor is None:
            break
        legal = eng.legal_actions(actor)
        assert legal
        eng.apply(actor, rng.choice(legal))
        steps += 1
    else:
        pytest.fail("timeout advancing to round 2")

    st = eng.state
    n = st.num_players
    unused_count = len(st.roles_in_play) - n
    assert len(st.role_card_doubloons) == unused_count
    assert sum(c for _, c in st.role_card_doubloons) == unused_count
    for _, c in st.role_card_doubloons:
        assert c >= 1


def _first_actor(eng: PuertoRicoEngine) -> Optional[int]:
    n = eng.state.num_players
    for i in range(n):
        if eng.legal_actions(i):
            return i
    return None


def test_builder_quarry_discount_column_cap() -> None:
    """Column 1 caps quarry discount at 1 even with two occupied quarries."""
    p = _make_player_with_quarries(doubloons=20, quarry_slots=[(1, True), (5, True)])
    cost = _builder_doubloon_cost_base(
        p, Building.LARGE_INDIGO_PLANT, anchor_slot=0, quarry_discount_applies=True
    )
    assert cost == 2  # printed 3 - 1


def test_builder_quarry_discount_column_4() -> None:
    """Column 4 allows up to four quarries discount; two quarries → −2 from printed cost."""
    p = _make_player_with_quarries(doubloons=30, quarry_slots=[(3, True), (7, True)])
    cost = _builder_doubloon_cost_base(
        p, Building.GUILD_HALL, anchor_slot=3, quarry_discount_applies=True
    )
    assert cost == 8  # printed 10 - 2


def test_builder_quarry_discount_only_for_chooser() -> None:
    """Quarry discount applies only when ``quarry_discount_applies`` (chooser) is True."""
    p = _make_player_with_quarries(doubloons=20, quarry_slots=[(1, True), (5, True)])
    chooser = _builder_doubloon_cost_base(
        p, Building.LARGE_INDIGO_PLANT, 0, quarry_discount_applies=True
    )
    other = _builder_doubloon_cost_base(
        p, Building.LARGE_INDIGO_PLANT, 0, quarry_discount_applies=False
    )
    assert chooser == 2
    assert other == 3


def test_final_scores_only_after_game_over() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    with pytest.raises(RuntimeError, match="GAME_OVER"):
        eng.final_scores()


def test_random_play_reaches_game_over_and_scores() -> None:
    rng = random.Random(4242)
    eng = PuertoRicoEngine()
    eng.reset(3, seed=99)
    steps = 0
    while not eng.is_terminal() and steps < 200_000:
        actor = _first_actor(eng)
        assert actor is not None
        legal = eng.legal_actions(actor)
        eng.apply(actor, rng.choice(legal))
        steps += 1
    assert eng.is_terminal()
    assert eng.state.phase is Phase.GAME_OVER
    scores = eng.final_scores()
    assert len(scores) == 3
    assert all(isinstance(s, int) for s in scores)


def test_game_end_sets_phase_game_over() -> None:
    rng = random.Random(7)
    eng = PuertoRicoEngine()
    eng.reset(4, seed=1)
    for _ in range(300_000):
        if eng.is_terminal():
            assert eng.state.phase is Phase.GAME_OVER
            return
        actor = _first_actor(eng)
        assert actor is not None
        eng.apply(actor, rng.choice(eng.legal_actions(actor)))
    pytest.fail("expected game to end")


def test_initial_game_state_matches_engine_reset() -> None:
    eng = PuertoRicoEngine()
    eng.reset(5, seed=123)
    assert eng.state == initial_game_state(5, seed=123)


def test_legal_actions_returns_empty_for_invalid_round_role_order_index() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)

    eng._state = dataclasses.replace(  # noqa: SLF001 - malformed-state regression
        eng.state,
        phase=Phase.SETTLER,
        round_role_order=(),
        current_role_execution_index=0,
    )
    assert eng.legal_actions(0) == ()

    eng._state = dataclasses.replace(  # noqa: SLF001 - malformed-state regression
        eng.state,
        phase=Phase.SETTLER,
        round_role_order=((Role.SETTLER, 0),),
        current_role_execution_index=1,
    )
    assert eng.legal_actions(0) == ()


def test_wharf_available_treats_short_wharf_used_as_false() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    player = dataclasses.replace(
        eng.state.players[0],
        city_buildings=(PlacedBuilding(building=Building.WHARF, anchor_slot=0, colonists=(1,)),),
        goods=normalize_goods_counts({Good.CORN: 1}),
    )
    state = dataclasses.replace(eng.state, players=(player,) + eng.state.players[1:])
    pending = CaptainPhasePending(
        captain_role_chooser=0,
        active_player_index=0,
        captain_privilege_vp_awarded=False,
        wharf_used=(),
        subphase="loading",
        storage_next_actor=None,
        storage_done=(False, False, False),
        ship_full_credit=(None, None, None),
    )

    assert _wharf_available(state, 0, pending) is True


def test_mayor_placement_legal_actions_return_empty_for_short_hands() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    eng._state = dataclasses.replace(  # noqa: SLF001 - malformed-state regression
        eng.state,
        phase=Phase.MAYOR,
        round_role_order=((Role.MAYOR, 0),),
        current_role_execution_index=0,
        pending=MayorPhasePending(
            mayor_role_chooser=0,
            ship_size_at_start=3,
            colonists_from_ship_remaining=0,
            colonists_hands=(),
            subphase="placement",
            privilege_done=True,
            placement_next=0,
        ),
    )

    assert eng.legal_actions(0) == ()


def test_mayor_next_with_colonists_in_hand_skips_short_hands_safely() -> None:
    assert PuertoRicoEngine._mayor_next_with_colonists_in_hand(0, 3, ()) is None


def test_apply_mayor_privilege_returns_error_for_short_hands() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    eng._state = dataclasses.replace(  # noqa: SLF001 - malformed-state regression
        eng.state,
        phase=Phase.MAYOR,
        round_role_order=((Role.MAYOR, 0),),
        current_role_execution_index=0,
        pending=MayorPhasePending(
            mayor_role_chooser=0,
            ship_size_at_start=3,
            colonists_from_ship_remaining=3,
            colonists_hands=(),
            subphase="privilege",
            privilege_done=False,
            placement_next=None,
        ),
    )

    assert eng._apply_impl(0, MayorPrivilegeTake()) == "invalid mayor colonists_hands index"  # noqa: SLF001


def test_apply_captain_load_returns_error_for_short_ship_full_credit() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    player0 = dataclasses.replace(eng.state.players[0], goods=normalize_goods_counts({Good.CORN: 1}))
    eng._state = dataclasses.replace(  # noqa: SLF001 - malformed-state regression
        eng.state,
        phase=Phase.CAPTAIN,
        players=(player0,) + eng.state.players[1:],
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
            ship_full_credit=(),
        ),
    )

    assert eng._apply_impl(0, CaptainLoad(Good.CORN, 0)) is None  # empty ship_full_credit: guard resets ship and succeeds  # noqa: SLF001


def test_apply_captain_storage_returns_error_for_short_storage_done() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    eng._state = dataclasses.replace(  # noqa: SLF001 - malformed-state regression
        eng.state,
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
            storage_done=(),
            ship_full_credit=(None, None, None),
        ),
    )

    assert (
        eng._apply_impl(0, CaptainStorageCommit(keep_counts=normalize_goods_counts({})))  # noqa: SLF001
        == "invalid captain storage_done index"
    )


def test_goods_helpers_return_safe_empty_values_for_malformed_tuples() -> None:
    assert good_count((Good.CORN, 1), Good.CORN) == 0  # type: ignore[arg-type]
    assert goods_dict((Good.CORN, 1)) == {}  # type: ignore[arg-type]
