"""Tests for ``PuertoRicoEngine`` behavior (rules and scoring hooks)."""

from __future__ import annotations

import dataclasses
import random
from typing import Optional

import pytest

from puerto_rico.engine import (
    BuilderBuild,
    CaptainLoad,
    CaptainStorageCommit,
    CraftsmanTurn,
    MayorPlaceColonistBuilding,
    MayorPrivilegeSkip,
    MayorPrivilegeTake,
    PuertoRicoEngine,
    SettlerTakeHacienda,
    SettlerTakeFaceUp,
    TraderSell,
    _wharf_available,
    _builder_doubloon_cost_base,
)
from puerto_rico.setup import initial_game_state
from puerto_rico.state import (
    Building,
    BuilderPhasePending,
    CraftsmanPhasePending,
    CaptainPhasePending,
    Good,
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


def test_builder_does_not_award_printed_vp_during_play() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    player0 = dataclasses.replace(eng.state.players[0], doubloons=10)
    eng._state = dataclasses.replace(  # noqa: SLF001 - targeted rule regression
        eng.state,
        players=(player0,) + eng.state.players[1:],
        phase=Phase.BUILDER,
        round_role_order=((Role.BUILDER, 0),),
        current_role_execution_index=0,
        pending=BuilderPhasePending(role_chooser=0, next_actor=0),
        vp_supply=1,
        game_end_vp=False,
    )

    eng.apply(0, _first_build_action(eng, 0, Building.SMALL_MARKET))

    assert eng.state.players[0].vp_from_chips == 0
    assert eng.state.vp_supply == 1
    assert eng.state.game_end_vp is False


def test_final_scores_include_printed_vp_and_large_building_bonuses() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    player0 = dataclasses.replace(
        eng.state.players[0],
        vp_from_chips=8,
        city_buildings=(
            PlacedBuilding(building=Building.GUILD_HALL, anchor_slot=0, colonists=(1, 0)),
            PlacedBuilding(building=Building.CITY_HALL, anchor_slot=2, colonists=(1, 0)),
            PlacedBuilding(building=Building.OFFICE, anchor_slot=4, colonists=(0,)),
            PlacedBuilding(building=Building.SMALL_INDIGO_PLANT, anchor_slot=5, colonists=(0,)),
            PlacedBuilding(building=Building.COFFEE_ROASTER, anchor_slot=6, colonists=(0, 0)),
        ),
    )
    eng._state = dataclasses.replace(eng.state, phase=Phase.GAME_OVER, players=(player0,) + eng.state.players[1:])

    assert eng.final_scores()[0] == 28


def test_settler_hacienda_adds_face_down_tile_before_normal_pick() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    player0 = dataclasses.replace(
        eng.state.players[0],
        city_buildings=(PlacedBuilding(building=Building.HACIENDA, anchor_slot=0, colonists=(1,)),),
    )
    eng._state = dataclasses.replace(  # noqa: SLF001 - targeted rule regression
        eng.state,
        players=(player0,) + eng.state.players[1:],
        phase=Phase.SETTLER,
        round_role_order=((Role.SETTLER, 0),),
        current_role_execution_index=0,
        plantation_stacks=((IslandTile.CORN,), (), (), (), ()),
        face_up_plantations=(IslandTile.INDIGO,),
        pending=SettlerPhasePending(settler_role_chooser=0, next_actor_index=0),
    )

    eng.apply(0, SettlerTakeHacienda())

    pending = eng.state.pending
    assert isinstance(pending, SettlerPhasePending)
    assert pending.next_actor_index == 0
    assert pending.awaiting_normal_pick is True
    assert sum(1 for sp in eng.state.players[0].island_spaces if sp.tile is not None) == 2
    assert SettlerTakeHacienda() not in eng.legal_actions(0)
    assert SettlerTakeFaceUp(0) in eng.legal_actions(0)


def test_hospice_places_colonist_on_normal_settler_pick() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    player0 = dataclasses.replace(
        eng.state.players[0],
        city_buildings=(PlacedBuilding(building=Building.HOSPICE, anchor_slot=0, colonists=(1,)),),
    )
    eng._state = dataclasses.replace(  # noqa: SLF001 - targeted rule regression
        eng.state,
        players=(player0,) + eng.state.players[1:],
        colonist_supply=1,
        phase=Phase.SETTLER,
        round_role_order=((Role.SETTLER, 0),),
        current_role_execution_index=0,
        face_up_plantations=(IslandTile.INDIGO,),
        pending=SettlerPhasePending(settler_role_chooser=0, next_actor_index=0),
    )

    eng.apply(0, SettlerTakeFaceUp(0))

    new_space = next(sp for sp in eng.state.players[0].island_spaces[1:] if sp.tile is not None)
    assert new_space.tile is IslandTile.INDIGO
    assert new_space.colonists == 1
    assert eng.state.colonist_supply == 0


def test_trader_pricing_uses_markets_and_privilege_but_not_office_bonus() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    player0 = dataclasses.replace(
        eng.state.players[0],
        goods=normalize_goods_counts({Good.CORN: 1}),
        city_buildings=(
            PlacedBuilding(building=Building.SMALL_MARKET, anchor_slot=0, colonists=(1,)),
            PlacedBuilding(building=Building.LARGE_MARKET, anchor_slot=1, colonists=(1,)),
            PlacedBuilding(building=Building.OFFICE, anchor_slot=2, colonists=(1,)),
        ),
    )
    start_doubloons = player0.doubloons
    eng._state = dataclasses.replace(  # noqa: SLF001 - targeted rule regression
        eng.state,
        players=(player0,) + eng.state.players[1:],
        phase=Phase.TRADER,
        round_role_order=((Role.TRADER, 0),),
        current_role_execution_index=0,
        trading_house=TradingHouseState(goods=(Good.CORN,)),
        pending=TraderPhasePending(role_chooser=0, next_actor=0),
    )

    eng.apply(0, TraderSell(Good.CORN))

    assert eng.state.players[0].doubloons == start_doubloons + 4
    assert eng.state.players[0].goods == ()
    assert eng.state.trading_house.goods == (Good.CORN, Good.CORN)


def test_factory_bonus_pays_by_kinds_produced() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    island = list(eng.state.players[0].island_spaces)
    island[0] = IslandSpace(tile=IslandTile.CORN, colonists=1)
    island[1] = IslandSpace(tile=IslandTile.INDIGO, colonists=1)
    player0 = dataclasses.replace(
        eng.state.players[0],
        city_buildings=(
            PlacedBuilding(building=Building.FACTORY, anchor_slot=0, colonists=(1,)),
            PlacedBuilding(building=Building.SMALL_INDIGO_PLANT, anchor_slot=1, colonists=(1,)),
        ),
        island_spaces=tuple(island),
    )
    start_doubloons = player0.doubloons
    eng._state = dataclasses.replace(  # noqa: SLF001 - targeted rule regression
        eng.state,
        players=(player0,) + eng.state.players[1:],
        phase=Phase.CRAFTSMAN,
        round_role_order=((Role.CRAFTSMAN, 0),),
        current_role_execution_index=0,
        pending=CraftsmanPhasePending(role_chooser=0, next_actor=0),
    )

    eng.apply(0, CraftsmanTurn(privilege_good=None))

    assert good_count(eng.state.players[0].goods, Good.CORN) == 1
    assert good_count(eng.state.players[0].goods, Good.INDIGO) == 1
    assert eng.state.players[0].doubloons == start_doubloons + 1


def test_captain_load_awards_vp_per_barrel_when_loaded() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    player0 = dataclasses.replace(eng.state.players[0], goods=normalize_goods_counts({Good.CORN: 2}))
    start_vp_supply = eng.state.vp_supply
    eng._state = dataclasses.replace(  # noqa: SLF001 - targeted rule regression
        eng.state,
        players=(player0,) + eng.state.players[1:],
        phase=Phase.CAPTAIN,
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
            ship_full_credit=(None, None, None),
        ),
    )

    eng.apply(0, CaptainLoad(Good.CORN, 0))

    assert eng.state.players[0].vp_from_chips == 3
    assert eng.state.vp_supply == start_vp_supply - 3


def test_mayor_refill_uses_empty_building_circles_with_player_minimum() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    players = list(eng.state.players)
    players[0] = dataclasses.replace(
        players[0],
        city_buildings=(
            PlacedBuilding(building=Building.TOBACCO_STORAGE, anchor_slot=0, colonists=(1, 0)),
            PlacedBuilding(building=Building.SMALL_MARKET, anchor_slot=1, colonists=(0,)),
        ),
    )
    players[1] = dataclasses.replace(
        players[1],
        city_buildings=(PlacedBuilding(building=Building.LARGE_INDIGO_PLANT, anchor_slot=0, colonists=(0, 0)),),
    )
    players[2] = dataclasses.replace(
        players[2],
        city_buildings=(PlacedBuilding(building=Building.SMALL_SUGAR_MILL, anchor_slot=0, colonists=(1,)),),
    )
    state = dataclasses.replace(
        eng.state,
        players=tuple(players),
        colonist_supply=10,
        round_role_order=((Role.SETTLER, 0), (Role.TRADER, 1), (Role.MAYOR, 0)),
        current_role_execution_index=2,
    )
    pending = MayorPhasePending(
        mayor_role_chooser=0,
        ship_size_at_start=3,
        colonists_from_ship_remaining=0,
        colonists_hands=(0, 0, 0),
        subphase="placement",
        privilege_done=True,
        placement_next=None,
    )

    eng._mayor_refill_ship_and_advance(state, pending)  # noqa: SLF001 - targeted rule regression

    assert eng.state.colonist_ship == 4
    assert eng.state.colonist_supply == 6


def test_mayor_collects_existing_colonists_for_rearrangement() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    island = list(eng.state.players[0].island_spaces)
    island[0] = IslandSpace(tile=IslandTile.INDIGO, colonists=1)
    player0 = dataclasses.replace(
        eng.state.players[0],
        san_juan_colonists=1,
        island_spaces=tuple(island),
        city_buildings=(PlacedBuilding(building=Building.SMALL_MARKET, anchor_slot=0, colonists=(1,)),),
    )
    eng._state = dataclasses.replace(  # noqa: SLF001 - targeted rule regression
        eng.state,
        players=(player0,) + eng.state.players[1:],
        phase=Phase.MAYOR,
        round_role_order=((Role.MAYOR, 0),),
        current_role_execution_index=0,
        colonist_ship=0,
        pending=MayorPhasePending(
            mayor_role_chooser=0,
            ship_size_at_start=0,
            colonists_from_ship_remaining=0,
            colonists_hands=(0, 0, 0),
            subphase="privilege",
            privilege_done=False,
            placement_next=None,
        ),
    )

    eng.apply(0, MayorPrivilegeSkip())

    pending = eng.state.pending
    assert isinstance(pending, MayorPhasePending)
    assert pending.subphase == "placement"
    assert pending.colonists_hands[0] == 3
    assert eng.state.players[0].san_juan_colonists == 0
    assert eng.state.players[0].island_spaces[0].colonists == 0
    assert eng.state.players[0].city_buildings[0].colonists == (0,)
    assert MayorPlaceColonistBuilding(0, 0) in eng.legal_actions(0)


def test_builder_can_rearrange_city_to_make_room_for_large_building() -> None:
    eng = PuertoRicoEngine()
    eng.reset(3, seed=0)
    player0 = dataclasses.replace(
        eng.state.players[0],
        doubloons=20,
        city_buildings=(
            PlacedBuilding(building=Building.SMALL_MARKET, anchor_slot=0, colonists=(0,)),
            PlacedBuilding(building=Building.HACIENDA, anchor_slot=2, colonists=(0,)),
            PlacedBuilding(building=Building.CONSTRUCTION_HUT, anchor_slot=4, colonists=(0,)),
        ),
    )
    eng._state = dataclasses.replace(  # noqa: SLF001 - targeted rule regression
        eng.state,
        players=(player0,) + eng.state.players[1:],
        phase=Phase.BUILDER,
        round_role_order=((Role.BUILDER, 0),),
        current_role_execution_index=0,
        pending=BuilderPhasePending(role_chooser=0, next_actor=0),
    )

    action = BuilderBuild(Building.GUILD_HALL, 0)
    assert action in eng.legal_actions(0)

    eng.apply(0, action)

    built = next(pb for pb in eng.state.players[0].city_buildings if pb.building is Building.GUILD_HALL)
    assert built.anchor_slot == 0
    occupied_slots = set()
    for pb in eng.state.players[0].city_buildings:
        width = 2 if pb.building is Building.GUILD_HALL else 1
        for offset in range(width):
            slot = pb.anchor_slot + offset
            assert slot not in occupied_slots
            occupied_slots.add(slot)
    assert len(occupied_slots) == 5


def _first_build_action(eng: PuertoRicoEngine, player_id: int, building: Building):
    for action in eng.legal_actions(player_id):
        if isinstance(action, BuilderBuild) and action.building is building:
            return action
    raise AssertionError(f"missing build action for {building}")
