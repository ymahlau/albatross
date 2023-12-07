
from src.game.initialization import get_game_from_config
from src.game.overcooked.overcooked import OvercookedGame as OvercookedGameFast
from src.game.overcooked_slow.overcooked import OvercookedSlowConfig, OvercookedGame as OvercookedGameSlow
from src.game.overcooked_slow.state import SimplifiedOvercookedState


def overcooked_slow_from_fast(game: OvercookedGameFast, layout_abbr: str) -> OvercookedGameSlow:
    abbrev_dict = {
        'cr': 'cramped_room',
        'aa': 'asymmetric_advantages',
        'co': 'coordination_ring',
        'fc': 'forced_coordination',
        'cc': 'counter_circuit_o_1order',
    }
    slow_game_cfg = OvercookedSlowConfig(
        overcooked_layout=abbrev_dict[layout_abbr],
        horizon=400,
        disallow_soup_drop=False,
        mep_reproduction_setting=True,
        mep_eval_setting=True,
        flat_obs=True,
    )
    game_slow = OvercookedGameSlow(slow_game_cfg)
    state_dict = game.generate_oc_state_dict()
    state = SimplifiedOvercookedState(state_dict)
    game_slow.set_state(state)
    return game_slow
