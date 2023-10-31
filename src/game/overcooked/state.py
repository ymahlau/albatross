from dataclasses import dataclass
from typing import Any

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld


@dataclass
class SimplifiedOvercookedState:
    state_dict: dict[str, Any]


def get_pot_state(
        state: OvercookedState,
        gridworld: OvercookedGridworld,
) -> list[int]:
    all_pots = gridworld.get_pot_locations()
    pot_state = gridworld.get_pot_states(state)
    state_list = []
    for pot in all_pots:
        if pot in pot_state['empty']:
            state_list.append(0)
        elif pot in pot_state['1_items']:
            state_list.append(1)
        elif pot in pot_state['2_items']:
            state_list.append(2)
        elif pot in pot_state['3_items']:
            state_list.append(3)
        elif pot in pot_state['ready']:
            state_list.append(4)
        else:
            pot_obj = state.get_object(pot)
            state_list.append(4 + pot_obj.cook_time_remaining)
    return state_list


def get_albatross_oa_state(time_step: int) -> SimplifiedOvercookedState:
    state_dict = {
        'players': [
            {'held_object': None, 'position': (5, 2), 'orientation': (-1, 0)},
            {'held_object': None, 'position': (3, 2), 'orientation': (1, 0)},
        ],
        'objects': [{
            'name': 'soup',
            'position': (4, 2),
            '_ingredients': [{'name': 'onion', 'position': (4, 2)} for _ in range(3)],
            'cooking_tick': 20,
            'is_cooking': False,
            'is_ready': True,
            'is_idle': False,
            'cook_time': 20,
            '_cooking_tick': 20,
        }],
        'bonus_orders': [],
        'all_orders': [{'ingredients': ('onion', 'onion', 'onion')}],
        'timestep': time_step,
    }
    return SimplifiedOvercookedState(state_dict)
