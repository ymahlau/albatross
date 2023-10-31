from typing import Optional, Any

from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, SoupState

OBJECT_NAMES = [
    'onion',
    'dish',
    'soup',
]


def object_dict_from_int(obj_id: int, pos: tuple[int, int]) -> Optional[dict[str, Any]]:
    obj_dict = None
    if obj_id == 0 or obj_id == 1:
        obj_dict = {
            'name': OBJECT_NAMES[obj_id],
            'position': pos,
        }
    elif obj_id == 2:
        obj_dict = {
            'name': 'soup',
            'position': pos,
            '_ingredients': [
                {'name': 'onion', 'position': pos},
                {'name': 'onion', 'position': pos},
                {'name': 'onion', 'position': pos}
            ],
            'cooking_tick': 20,
            'is_cooking': False,
            'is_ready': True,
            'is_idle': False,
            'cook_time': 20,
            '_cooking_tick': 20,
        }
    return obj_dict


def object_from_int(obj_id: int, pos: tuple[int, int]) -> Optional[ObjectState]:
    if obj_id == 3:
        return None
    obj_dict = object_dict_from_int(obj_id=obj_id, pos=pos)
    if obj_id == 0 or obj_id == 1:
        obj = ObjectState.from_dict(obj_dict)
    else:
        obj = SoupState.from_dict(obj_dict)
    return obj


def get_pot_dict_from_int(pot_id: int, pos: tuple[int, int]) -> Optional[dict[str, Any]]:
    if pot_id < 0 or pot_id > 23:
        raise ValueError(f"Invalid pot id: {pot_id}")
    if pot_id == 0:
        return None
    if pot_id == 4:
        return object_dict_from_int(2, pos)
    if pot_id in [1, 2, 3]:
        pot_dict = {
            'name': 'soup',
            'position': pos,
            '_ingredients': [
                {'name': 'onion', 'position': pos} for _ in range(pot_id)
            ],
            'cooking_tick': -1,
            'is_cooking': False,
            'is_ready': False,
            'is_idle': True,
            'cook_time': 20,
            '_cooking_tick': -1,
        }
        return pot_dict
    # pot is currently cooking
    cook_tick = 24 - pot_id
    pot_dict = {
        'name': 'soup',
        'position': pos,
        '_ingredients': [
            {'name': 'onion', 'position': pos},
            {'name': 'onion', 'position': pos},
            {'name': 'onion', 'position': pos}
        ],
        'cooking_tick': cook_tick,
        'is_cooking': True,
        'is_ready': False,
        'is_idle': False,
        'cook_time': 20,
        '_cooking_tick': cook_tick,
    }
    return pot_dict
