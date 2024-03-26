from dataclasses import dataclass, field
from typing import Any, Optional
from queue import PriorityQueue

import random
import numpy as np

from src.agent import Agent, AgentConfig
from src.game.overcooked.overcooked import OvercookedGame

NO_ITEM = 0
ONION_ITEM = 1
DISH_ITEM = 2
SOUP_ITEM = 3

EMPTY_TILE = 0
COUNTER_TILE = 1
DISH_TILE = 2
ONION_TILE = 3
POT_TILE = 4
SERVING_TILE = 5

UP_ACTION = 0
DOWN_ACTION = 1
RIGHT_ACTION = 2
LEFT_ACTION = 3
STAY_ACTION = 4
INTERACT_ACTION = 5

EMPTY_POT = 0
ONE_POT = 1
TWO_POT = 2
THREE_POT = 3
DONE_POT = 4

def _new_pos(pos: tuple[int, int], direction: int) -> tuple[int, int]:
    if direction == 0:
        return pos[0], pos[1] - 1
    elif direction == 1:
        return pos[0], pos[1] + 1
    elif direction == 2:
        return pos[0] + 1, pos[1]
    elif direction == 3:
        return pos[0] - 1, pos[1]
    else:
        raise ValueError(f"Unknown direction: {direction}")

def a_star_search(
        start_field: tuple[int, int],
        search_field: tuple[int, int],
        blocked_spaces: np.ndarray,  # array of shape w, h
) -> tuple[bool, int, int]:  # returns success, distance and direction
    save_start_field_block = blocked_spaces[*start_field]
    save_search_field_block = blocked_spaces[*search_field]
    blocked_spaces[*start_field] = False
    blocked_spaces[*search_field] = False
    w, h = blocked_spaces.shape[0], blocked_spaces.shape[1],
    came_from = {}
    cost_so_far = {}
    prio_q = PriorityQueue()
    prio_q.put_nowait((0, start_field))
    cost_so_far[start_field] = 0
    pos = None
    while not prio_q.empty():
        dist, pos = prio_q.get_nowait()
        if pos == search_field:
            break
        for direction in range(4):
            new_pos = _new_pos(pos, direction)
            if new_pos[0] < 0 or new_pos[0] >= w or new_pos[1] < 0 or new_pos[1] >= h:
                continue
            if blocked_spaces[new_pos[0], new_pos[1]]:
                continue
            new_cost = cost_so_far[pos] + 1
            if new_pos not in cost_so_far or cost_so_far[new_pos] > new_cost:
                cost_so_far[new_pos] = new_cost
                euc_dist = abs(new_pos[0] - search_field[0]) + abs(new_pos[1] - search_field[1])
                total_cost = euc_dist + new_cost
                prio_q.put_nowait((total_cost, new_pos))
                came_from[new_pos] = (pos, direction)
    last_dir = None
    cost = 0
    success = search_field in came_from
    blocked_spaces[*start_field] = save_start_field_block
    blocked_spaces[*search_field] = save_search_field_block
    while pos != start_field:
        last_dir = came_from[pos][1]
        pos = came_from[pos][0]
        cost += 1
    if last_dir is None:
        random_action = random.choice(list(range(4)))
        return False, 0, random_action
    return success, cost, last_dir


@dataclass
class PlaceOnionAgentConfig(AgentConfig):
    name: str = field(default="Place-Onion-Agent")
    
    
class PlaceOnionAgent(Agent):

    def __init__(self, cfg: PlaceOnionAgentConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        self.goal_position: tuple[int, int] | None = None
    
    def reset_episode(self):
        super().reset_episode()
        self.goal_position = None
    
    def _act(
        self,
        game: OvercookedGame,
        player: int,
        time_limit: Optional[float] = None,
        iterations: Optional[int] = None,
        save_probs = None,  # mp.Array
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        p_info = game.get_player_info()
        cur_p_item = p_info[f'p{player}_item']
        cur_x, cur_y = p_info[f'p{player}_x'], p_info[f'p{player}_y']
        cur_orientation = p_info[f'p{player}_or']
        
        other_player = 1 - player
        other_x, other_y = p_info[f'p{other_player}_x'], p_info[f'p{other_player}_y']
        
        board = np.asarray(game.cfg.board).T
        walkable_mask = board == EMPTY_TILE
        walkable_mask[other_x, other_y] = False
        walkable_mask[cur_x, cur_y] = True
        blocked_spaces = np.logical_not(walkable_mask)
        
        pot_states = game.get_pot_positions_and_states()
        
        # find goal to walk towards
        if self.goal_position is None:
            if cur_p_item == ONION_ITEM:
                # find pot and put onion into it
                viable_pots = []
                for pot_pos in game.get_pot_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=pot_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable and pot_states[pot_pos] in [0, 1, 2]:
                        viable_pots.append(pot_pos)
                if not viable_pots:
                    # random move action
                    random_action = random.choice(list(range(4)))
                    return np.eye(6)[random_action], {}
                self.goal_position = random.choice(viable_pots)
            elif cur_p_item == NO_ITEM:
                # find onion dispenser and go there
                viable_dispenser = []
                for disp_pos in game.get_onion_disp_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=disp_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable:
                        viable_dispenser.append(disp_pos)
                self.goal_position = random.choice(viable_dispenser)
            else:
                # player has other item, place on random empty counter
                counter_states = game.get_counter_positions_and_states()
                viable_counter = []
                for c_pos, c_state in counter_states.items():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=c_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable and c_state == NO_ITEM:
                        viable_counter.append(c_pos)
                self.goal_position = random.choice(viable_counter)
        
        assert self.goal_position is not None
        # we have a goal to walk towards.
        reachable, dist, action = a_star_search(
            start_field=(cur_x, cur_y),
            search_field=self.goal_position,
            blocked_spaces=blocked_spaces,
        )
        if not reachable:
            # random walking move
            random_action = random.choice(list(range(4)))
            one_hot_probs = np.eye(6)[random_action]
            return one_hot_probs, {}
        if dist == 1:
            # check if we are already facing the goal position
            pos_looking_at = _new_pos((cur_x, cur_y), cur_orientation)
            if pos_looking_at != self.goal_position:
                # if not, turn towards tile
                return np.eye(6)[action], {}
            # else interact with goal position
            if self.goal_position in pot_states.keys() \
                    and pot_states[self.goal_position] == TWO_POT \
                    and cur_p_item == ONION_ITEM:
                # if the agent put the third onion in the pot, then start cooking afterwards
                # Do not update goal position to interact with the pot again
                pass
            else:
                self.goal_position = None
            return np.eye(6)[INTERACT_ACTION], {}
        # else follow path to goal
        return np.eye(6)[action], {}
    

@dataclass
class PlaceOnionDeliverAgentConfig(AgentConfig):
    name: str = field(default="Place-Onion-Agent")
    
    
class PlaceOnionDeliverAgent(Agent):

    def __init__(self, cfg: PlaceOnionDeliverAgentConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        self.goal_position: tuple[int, int] | None = None
        self.cur_task: int | None = None  # 0 is onion placement, 1 is soup delivery
    
    def reset_episode(self):
        super().reset_episode()
        self.goal_position = None
        self.cur_task = None
    
    def _act(
        self,
        game: OvercookedGame,
        player: int,
        time_limit: Optional[float] = None,
        iterations: Optional[int] = None,
        save_probs = None,  # mp.Array
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        p_info = game.get_player_info()
        cur_p_item = p_info[f'p{player}_item']
        cur_x, cur_y = p_info[f'p{player}_x'], p_info[f'p{player}_y']
        cur_orientation = p_info[f'p{player}_or']
        
        other_player = 1 - player
        other_x, other_y = p_info[f'p{other_player}_x'], p_info[f'p{other_player}_y']
        
        board = np.asarray(game.cfg.board).T
        walkable_mask = board == EMPTY_TILE
        walkable_mask[other_x, other_y] = False
        walkable_mask[cur_x, cur_y] = True
        blocked_spaces = np.logical_not(walkable_mask)
        
        pot_states = game.get_pot_positions_and_states()
        counter_states = game.get_counter_positions_and_states()
        
        # find goal to walk towards
        # 0 is onion placement, 1 is soup delivery
        if self.cur_task is None:
            if all(list(x != 4 for x in pot_states.values())):
                self.cur_task = 0
            else:
                r_val = random.random()
                self.cur_task = 0 if r_val < 0.5 else 1
        
        assert self.cur_task is not None
        if self.cur_task == 0 and self.goal_position is None:
            if cur_p_item == ONION_ITEM:
                # find pot and put onion into it
                viable_pots = []
                for pot_pos in game.get_pot_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=pot_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable and pot_states[pot_pos] in [0, 1, 2]:
                        viable_pots.append(pot_pos)
                if not viable_pots:
                    # random move action
                    random_action = random.choice(list(range(4)))
                    return np.eye(6)[random_action], {}
                self.goal_position = random.choice(viable_pots)
            elif cur_p_item == NO_ITEM:
                # find onion dispenser and go there
                viable_dispenser = []
                for disp_pos in game.get_onion_disp_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=disp_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable:
                        viable_dispenser.append(disp_pos)
                self.goal_position = random.choice(viable_dispenser)
            else:
                # player has other item, place on random empty counter
                counter_states = game.get_counter_positions_and_states()
                viable_counter = []
                for c_pos, c_state in counter_states.items():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=c_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable and c_state == NO_ITEM:
                        viable_counter.append(c_pos)
                self.goal_position = random.choice(viable_counter)
        elif self.cur_task == 1 and self.goal_position is None:
            # deliver soup
            if cur_p_item == SOUP_ITEM:
                # find serving location and deliver soup
                viable_serving_locs = []
                for serv_pos in game.get_serving_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=serv_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable:
                        viable_serving_locs.append(serv_pos)
                if not viable_serving_locs:
                    # random move action
                    random_action = random.choice(list(range(4)))
                    return np.eye(6)[random_action], {}
                self.goal_position = random.choice(viable_serving_locs)
            elif cur_p_item == DISH_ITEM:
                # find pot and retrieve the soup
                viable_pots = []
                for pot_pos in game.get_pot_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=pot_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable and pot_states[pot_pos] == DONE_POT:
                        viable_pots.append(pot_pos)
                if not viable_pots:
                    sorted_pot_list: list[tuple[tuple[int, int], Any]] = sorted(
                        pot_states.items(), 
                        key=lambda x: 24 - x[1] if x[1] > 3 else x[1],
                        reverse=True,
                    )
                    max_pot_tpl = sorted_pot_list[0]
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=max_pot_tpl[0],
                        blocked_spaces=blocked_spaces,
                    )
                    return np.eye(6)[action], {}
                    # random move action
                    # random_action = random.choice(list(range(4)))
                    # self.goal_position = None
                    # self.cur_task = None
                    # return np.eye(6)[random_action], {}
                self.goal_position = random.choice(viable_pots)
            elif cur_p_item == NO_ITEM:
                # find dish dispenser and go there
                viable_dispenser = []
                for disp_pos in game.get_dish_disp_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=disp_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable:
                        viable_dispenser.append(disp_pos)
                for counter_pos in game.get_counter_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=counter_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable and counter_states[counter_pos] == DISH_ITEM:
                        viable_dispenser.append(counter_pos)
                self.goal_position = random.choice(viable_dispenser)
            else:
                # player has other item, place on random empty counter
                counter_states = game.get_counter_positions_and_states()
                viable_counter = []
                for c_pos, c_state in counter_states.items():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=c_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable and c_state == NO_ITEM:
                        viable_counter.append(c_pos)
                self.goal_position = random.choice(viable_counter)
        
        assert self.goal_position is not None
        # we have a goal to walk towards.
        reachable, dist, action = a_star_search(
            start_field=(cur_x, cur_y),
            search_field=self.goal_position,
            blocked_spaces=blocked_spaces,
        )
        if not reachable:
            # random walking move
            random_action = random.choice(list(range(4)))
            one_hot_probs = np.eye(6)[random_action]
            return one_hot_probs, {}
        if dist == 1:
            # check if we are already facing the goal position
            pos_looking_at = _new_pos((cur_x, cur_y), cur_orientation)
            if pos_looking_at != self.goal_position:
                # if not, turn towards tile
                return np.eye(6)[action], {}
            # else interact with goal position
            if self.goal_position in pot_states.keys() \
                    and pot_states[self.goal_position] == TWO_POT \
                    and cur_p_item == ONION_ITEM:
                # if the agent put the third onion in the pot, then start cooking afterwards
                # Do not update goal position to interact with the pot again
                pass
            else:
                if self.goal_position in pot_states.keys() and \
                        (cur_p_item == NO_ITEM or cur_p_item == ONION_ITEM):
                    self.cur_task = None
                if self.goal_position in game.get_serving_positions():
                    self.cur_task = None
                self.goal_position = None
            return np.eye(6)[INTERACT_ACTION], {}
        # else follow path to goal
        return np.eye(6)[action], {}

@dataclass
class PlaceOnionEverywhereAgentConfig(AgentConfig):
    name: str = field(default="Place-Onion-Everywhere-Agent")
    
class PlaceOnionEverywhereAgent(Agent):

    def __init__(self, cfg: PlaceOnionEverywhereAgentConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        self.goal_position: tuple[int, int] | None = None
        
    def reset_episode(self):
        super().reset_episode()
        self.goal_position = None
        
    def _act(
        self,
        game: OvercookedGame,
        player: int,
        time_limit: Optional[float] = None,
        iterations: Optional[int] = None,
        save_probs = None,  # mp.Array
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        p_info = game.get_player_info()
        cur_p_item = p_info[f'p{player}_item']
        cur_x, cur_y = p_info[f'p{player}_x'], p_info[f'p{player}_y']
        cur_orientation = p_info[f'p{player}_or']
        
        other_player = 1 - player
        other_x, other_y = p_info[f'p{other_player}_x'], p_info[f'p{other_player}_y']
        
        board = np.asarray(game.cfg.board).T
        walkable_mask = board == EMPTY_TILE
        walkable_mask[other_x, other_y] = False
        walkable_mask[cur_x, cur_y] = True
        blocked_spaces = np.logical_not(walkable_mask)
        
        pot_states = game.get_pot_positions_and_states()
        counter_states = game.get_counter_positions_and_states()
        
        # find goal to walk towards
        if self.goal_position is None:
            if cur_p_item == ONION_ITEM:
                # find counter and put onion into it
                viable_counter = []
                for counter_pos in game.get_counter_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=counter_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable:
                        viable_counter.append(counter_pos)
                for pot_pos in game.get_pot_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=pot_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable and pot_states[pot_pos] in [0, 1, 2]:
                        viable_counter.append(pot_pos)
                if not viable_counter:
                    # random move action
                    random_action = random.choice(list(range(4)))
                    return np.eye(6)[random_action], {}
                self.goal_position = random.choice(viable_counter)
            elif cur_p_item == NO_ITEM:
                # find onion dispenser and go there
                viable_dispenser = []
                for disp_pos in game.get_onion_disp_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=disp_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable:
                        viable_dispenser.append(disp_pos)
                for counter_pos in game.get_counter_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=counter_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable and counter_states[counter_pos] == ONION_ITEM:
                        viable_dispenser.append(counter_pos)
                if not viable_dispenser:
                    # random move action
                    random_action = random.choice(list(range(4)))
                    return np.eye(6)[random_action], {}
                self.goal_position = random.choice(viable_dispenser)
            else:
                # player has other item, place on random empty counter
                counter_states = game.get_counter_positions_and_states()
                viable_counter = []
                for c_pos, c_state in counter_states.items():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=c_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable and c_state == NO_ITEM:
                        viable_counter.append(c_pos)
                self.goal_position = random.choice(viable_counter)
        
        assert self.goal_position is not None
        # we have a goal to walk towards.
        reachable, dist, action = a_star_search(
            start_field=(cur_x, cur_y),
            search_field=self.goal_position,
            blocked_spaces=blocked_spaces,
        )
        if not reachable:
            # random walking move
            random_action = random.choice(list(range(4)))
            one_hot_probs = np.eye(6)[random_action]
            return one_hot_probs, {}
        if dist == 1:
            # check if we are already facing the goal position
            pos_looking_at = _new_pos((cur_x, cur_y), cur_orientation)
            if pos_looking_at != self.goal_position:
                # if not, turn towards tile
                return np.eye(6)[action], {}
            # else interact with goal position
            self.goal_position = None
            return np.eye(6)[INTERACT_ACTION], {}
        # else follow path to goal
        return np.eye(6)[action], {}
        

@dataclass
class PlaceDishEverywhereAgentConfig(AgentConfig):
    name: str = field(default="Place-Onion-Everywhere-Agent")
    
class PlaceDishEverywhereAgent(Agent):

    def __init__(self, cfg: PlaceDishEverywhereAgentConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        self.goal_position: tuple[int, int] | None = None
        
    def reset_episode(self):
        super().reset_episode()
        self.goal_position = None
        
    def _act(
        self,
        game: OvercookedGame,
        player: int,
        time_limit: Optional[float] = None,
        iterations: Optional[int] = None,
        save_probs = None,  # mp.Array
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        p_info = game.get_player_info()
        cur_p_item = p_info[f'p{player}_item']
        cur_x, cur_y = p_info[f'p{player}_x'], p_info[f'p{player}_y']
        cur_orientation = p_info[f'p{player}_or']
        
        other_player = 1 - player
        other_x, other_y = p_info[f'p{other_player}_x'], p_info[f'p{other_player}_y']
        
        board = np.asarray(game.cfg.board).T
        walkable_mask = board == EMPTY_TILE
        walkable_mask[other_x, other_y] = False
        walkable_mask[cur_x, cur_y] = True
        blocked_spaces = np.logical_not(walkable_mask)
        
        counter_states = game.get_counter_positions_and_states()
        
        # find goal to walk towards
        if self.goal_position is None:
            if cur_p_item == DISH_ITEM:
                # find counter and put dish into it
                viable_counter = []
                for counter_pos in game.get_counter_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=counter_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    viable_counter.append(counter_pos)
                    # if reachable:
                    #     viable_counter.append(counter_pos)
                if not viable_counter:
                    # random move action
                    random_action = random.choice(list(range(4)))
                    return np.eye(6)[random_action], {}
                self.goal_position = random.choice(viable_counter)
            elif cur_p_item == NO_ITEM:
                # find dish dispenser and go there
                viable_dispenser = []
                for disp_pos in game.get_dish_disp_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=disp_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    viable_dispenser.append(disp_pos)
                    # if reachable:
                    #     viable_dispenser.append(disp_pos)
                for counter_pos in game.get_counter_positions():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=counter_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    # if reachable and counter_states[counter_pos] == DISH_ITEM:
                    #     viable_dispenser.append(counter_pos)
                    if counter_states[counter_pos] == DISH_ITEM:
                        viable_dispenser.append(counter_pos)
                if not viable_dispenser:
                    # random move action
                    random_action = random.choice(list(range(4)))
                    return np.eye(6)[random_action], {}
                self.goal_position = random.choice(viable_dispenser)
            else:
                # player has other item, place on random empty counter
                counter_states = game.get_counter_positions_and_states()
                viable_counter = []
                for c_pos, c_state in counter_states.items():
                    reachable, dist, action = a_star_search(
                        start_field=(cur_x, cur_y),
                        search_field=c_pos,
                        blocked_spaces=blocked_spaces,
                    )
                    if reachable and c_state == NO_ITEM:
                        viable_counter.append(c_pos)
                self.goal_position = random.choice(viable_counter)
        
        assert self.goal_position is not None
        # we have a goal to walk towards.
        reachable, dist, action = a_star_search(
            start_field=(cur_x, cur_y),
            search_field=self.goal_position,
            blocked_spaces=blocked_spaces,
        )
        if not reachable:
            # random walking move
            random_action = random.choice(list(range(4)))
            one_hot_probs = np.eye(6)[random_action]
            return one_hot_probs, {}
        if dist == 1:
            # check if we are already facing the goal position
            pos_looking_at = _new_pos((cur_x, cur_y), cur_orientation)
            if pos_looking_at != self.goal_position:
                # if not, turn towards tile
                return np.eye(6)[action], {}
            # else interact with goal position
            self.goal_position = None
            return np.eye(6)[INTERACT_ACTION], {}
        # else follow path to goal
        return np.eye(6)[action], {}
