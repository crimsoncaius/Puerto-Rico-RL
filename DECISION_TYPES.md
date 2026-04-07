# RL Environment Decision Types

The PettingZoo environment exposes 10 structured decision types through `action_space`,
`action_mask`, and `_decode_action` in `puerto_rico/env.py`.

## How To Read Actions

The agent submits a dictionary. Only one branch is active at a time, based on the current phase.

Examples:

```python
{"role": 7}
{"settler": 4}
{"mayor_privilege": 1}
{
    "mayor_placement": {
        "island": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "buildings": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "san_juan": 1,
    }
}
{"builder": 120}
{"craftsman": 3}
{"trader": 2}
{"captain_loading": 6}
{"captain_storage": 12345}
{"prospector": 0}
```

The active branch is determined by phase:

- `Phase.ROLE_SELECTION` -> `role`
- `Phase.SETTLER` -> `settler`
- `Phase.MAYOR` with `subphase == "privilege"` -> `mayor_privilege`
- `Phase.MAYOR` with `subphase == "placement"` -> `mayor_placement`
- `Phase.BUILDER` -> `builder`
- `Phase.CRAFTSMAN` -> `craftsman`
- `Phase.TRADER` -> `trader`
- `Phase.CAPTAIN` with `subphase == "loading"` -> `captain_loading`
- `Phase.CAPTAIN` with `subphase == "storage"` -> `captain_storage`
- `Phase.PROSPECTOR` -> `prospector`

The legal subset is always given by the corresponding `action_mask` branch.

## Shared Index Orders

These orders are used by the environment when converting indices to engine actions.

`role` order:
`["builder", "captain", "craftsman", "mayor", "prospector", "prospector_a", "prospector_b", "settler", "trader"]`

`good` order:
`["coffee", "corn", "indigo", "sugar", "tobacco"]`

`building` order:
`["city_hall", "coffee_roaster", "construction_hut", "customs_house", "factory", "fortress", "guild_hall", "hacienda", "harbor", "hospice", "large_indigo_plant", "large_market", "large_sugar_mill", "large_warehouse", "office", "residence", "small_indigo_plant", "small_market", "small_sugar_mill", "small_warehouse", "tobacco_storage", "university", "wharf"]`

## Decision Types

### 1. `role`

When used:
`Phase.ROLE_SELECTION`

Agent input:
`{"role": role_index}`

Choice count:
`9`

Possible decoded actions:
`PickRole(role)`

Index meaning:
- `0` -> `PickRole(Role.BUILDER)`
- `1` -> `PickRole(Role.CAPTAIN)`
- `2` -> `PickRole(Role.CRAFTSMAN)`
- `3` -> `PickRole(Role.MAYOR)`
- `4` -> `PickRole(Role.PROSPECTOR)`
- `5` -> `PickRole(Role.PROSPECTOR_A)`
- `6` -> `PickRole(Role.PROSPECTOR_B)`
- `7` -> `PickRole(Role.SETTLER)`
- `8` -> `PickRole(Role.TRADER)`

Notes:
- Not every role is legal in every player count.
- Use the mask to see which roles are available this round.

### 2. `settler`

When used:
`Phase.SETTLER`

Agent input:
`{"settler": choice_index}`

Choice count:
`9`

Possible decoded actions:
- `SettlerPass()`
- `SettlerTakeHacienda()`
- `SettlerTakeQuarryPrivilege()`
- `SettlerTakeQuarryConstructionHut()`
- `SettlerTakeFaceUp(face_up_index)`

Index meaning:
- `0` -> `SettlerPass()`
- `1` -> `SettlerTakeHacienda()`
- `2` -> `SettlerTakeQuarryPrivilege()`
- `3` -> `SettlerTakeQuarryConstructionHut()`
- `4` -> `SettlerTakeFaceUp(0)`
- `5` -> `SettlerTakeFaceUp(1)`
- `6` -> `SettlerTakeFaceUp(2)`
- `7` -> `SettlerTakeFaceUp(3)`
- `8` -> `SettlerTakeFaceUp(4)`

What the agent is choosing:
- Pass.
- Use Hacienda.
- Take a quarry via settler privilege.
- Take a quarry via Construction Hut.
- Take one of the face-up plantations by market index.

### 3. `mayor_privilege`

When used:
`Phase.MAYOR` and `pending.subphase == "privilege"`

Agent input:
`{"mayor_privilege": choice_index}`

Choice count:
`2`

Possible decoded actions:
- `MayorPrivilegeTake()`
- `MayorPrivilegeSkip()`

Index meaning:
- `0` -> `MayorPrivilegeTake()`
- `1` -> `MayorPrivilegeSkip()`

What the agent is choosing:
- Take the mayor privilege colonist.
- Skip the privilege.

### 4. `mayor_placement`

When used:
`Phase.MAYOR` and `pending.subphase == "placement"`

Agent input:

```python
{
    "mayor_placement": {
        "island": [int] * 12,
        "buildings": [int] * 12,
        "san_juan": int,
    }
}
```

Decoded action:
`MayorSubmitPlacement(island_targets=..., building_targets=..., san_juan=...)`

What the agent is choosing:
- Final colonist allocation across all 12 island spaces.
- Final worker allocation across up to 12 placed buildings.
- How many colonists remain in San Juan.

How to interpret the mask:
- `total_pool` is the number of colonists the player must place.
- `island_capacity[i]` is the max colonists allowed on island slot `i`.
- `building_capacity[i]` is the max workers allowed on building entry `i`.

Important constraint:
- The submitted counts must exactly allocate the full pool between island, buildings, and San Juan.

### 5. `builder`

When used:
`Phase.BUILDER`

Agent input:
`{"builder": choice_index}`

Choice count:
`278`

Possible decoded actions:
- `BuilderBuild(building, anchor_slot)`
- `BuilderPass()`
- `BuilderNoOp()`

Index meaning:
- `0..275` -> build choice
- `276` -> `BuilderPass()`
- `277` -> `BuilderNoOp()`

Build-choice formula:
`building_index = choice_index // 12`

`anchor_slot = choice_index % 12`

`BuilderBuild(building=_BUILDING_ORDER[building_index], anchor_slot=anchor_slot)`

What the agent is choosing:
- Which building to buy.
- Which city anchor slot to place it at.
- Or pass / no-op if that is the only legal outcome.

### 6. `craftsman`

When used:
`Phase.CRAFTSMAN`

Agent input:
`{"craftsman": choice_index}`

Choice count:
`6`

Possible decoded actions:
`CraftsmanTurn(privilege_good=good_or_none, hacienda_good=None)`

Index meaning:
- `0` -> `CraftsmanTurn(privilege_good=None, hacienda_good=None)`
- `1` -> `CraftsmanTurn(privilege_good=Good.COFFEE, hacienda_good=None)`
- `2` -> `CraftsmanTurn(privilege_good=Good.CORN, hacienda_good=None)`
- `3` -> `CraftsmanTurn(privilege_good=Good.INDIGO, hacienda_good=None)`
- `4` -> `CraftsmanTurn(privilege_good=Good.SUGAR, hacienda_good=None)`
- `5` -> `CraftsmanTurn(privilege_good=Good.TOBACCO, hacienda_good=None)`

What the agent is choosing:
- No privilege good.
- Or which extra privilege barrel to take, if legal.

### 7. `trader`

When used:
`Phase.TRADER`

Agent input:
`{"trader": choice_index}`

Choice count:
`6`

Possible decoded actions:
- `TraderPass()`
- `TraderSell(good)`

Index meaning:
- `0` -> `TraderPass()`
- `1` -> `TraderSell(Good.COFFEE)`
- `2` -> `TraderSell(Good.CORN)`
- `3` -> `TraderSell(Good.INDIGO)`
- `4` -> `TraderSell(Good.SUGAR)`
- `5` -> `TraderSell(Good.TOBACCO)`

What the agent is choosing:
- Pass.
- Or which good to sell.

### 8. `captain_loading`

When used:
`Phase.CAPTAIN` and `pending.subphase == "loading"`

Agent input:
`{"captain_loading": choice_index}`

Choice count:
`21`

Possible decoded actions:
- `CaptainLoad(good, ship_index)`
- `CaptainUseWharf(good)`
- `CaptainPassLoading()`

Index meaning:
- `0..14` -> `CaptainLoad`
- `15..19` -> `CaptainUseWharf`
- `20` -> `CaptainPassLoading()`

Load-choice formula:
`good_index = choice_index // 3`

`ship_index = choice_index % 3`

`CaptainLoad(good=_GOOD_ORDER[good_index], ship_index=ship_index)`

Wharf indices:
- `15` -> `CaptainUseWharf(Good.COFFEE)`
- `16` -> `CaptainUseWharf(Good.CORN)`
- `17` -> `CaptainUseWharf(Good.INDIGO)`
- `18` -> `CaptainUseWharf(Good.SUGAR)`
- `19` -> `CaptainUseWharf(Good.TOBACCO)`

What the agent is choosing:
- Which good to load.
- Which cargo ship to load onto.
- Whether to use Wharf instead.
- Or pass if no legal loading action remains.

### 9. `captain_storage`

When used:
`Phase.CAPTAIN` and `pending.subphase == "storage"`

Agent input:
`{"captain_storage": choice_index}`

Choice count:
`13^5 = 371293` registry entries

Decoded action:
`CaptainStorageCommit(keep_counts=...)`

Registry convention:
- The registry enumerates all 5-good keep-count combinations.
- Each good count ranges from `0` to `12`.
- The good order is `["coffee", "corn", "indigo", "sugar", "tobacco"]`.

What the agent is choosing:
- Exactly how many barrels of each good to keep after captain storage.
- Any barrels not kept are returned to supply.

Practical note:
- This branch has a very large discrete space.
- The intended way to act here is to sample or search only within the masked legal indices.

### 10. `prospector`

When used:
`Phase.PROSPECTOR`

Agent input:
`{"prospector": 0}`

Choice count:
`1`

Possible decoded actions:
`ProspectorCollect()`

Index meaning:
- `0` -> `ProspectorCollect()`

What the agent is choosing:
- Nothing beyond confirming the only legal prospector action.
