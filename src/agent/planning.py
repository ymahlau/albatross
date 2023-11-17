import multiprocessing as mp
import random
from dataclasses import dataclass, field
from queue import PriorityQueue
from typing import Optional, Any

import numpy as np

from src.agent import AgentConfig, Agent
from src.game.game import Game
from src.game.battlesnake.battlesnake import BattleSnakeGame


@dataclass
class AStarAgentConfig(AgentConfig):
    name: str = field(default="A*-Agent")

class AStarAgent(Agent):

    def __init__(self, cfg: AStarAgentConfig):
        super().__init__(cfg)
        self.cfg = cfg

    @staticmethod
    def _new_pos(pos: tuple[int, int], direction: int):
        if direction == 0:
            return pos[0], pos[1] + 1
        elif direction == 1:
            return pos[0] + 1, pos[1]
        elif direction == 2:
            return pos[0], pos[1] - 1
        elif direction == 3:
            return pos[0] - 1, pos[1]
        else:
            raise ValueError(f"Unknown direction: {direction}")

    @staticmethod
    def _a_star_search(
            start_field: tuple[int, int],
            search_field: tuple[int, int],
            game: BattleSnakeGame,
            blocked_spaces: np.ndarray,  # array of shape w, h
            player: int,
    ) -> tuple[bool, int, int]:  # returns success, distance and direction
        w, h = game.cfg.w, game.cfg.h
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
                new_pos = AStarAgent._new_pos(pos, direction)
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
        while pos != start_field:
            last_dir = came_from[pos][1]
            pos = came_from[pos][0]
            cost += 1
        if last_dir is None:
            ra = random.choice(game.available_actions(player))
            return False, 0, ra
        return success, cost, last_dir

    def _act(
            self,
            game: Game,
            player: int,
            time_limit: Optional[float] = None,
            iterations: Optional[int] = None,
            save_probs = None,  # mp.Array
            options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(game, BattleSnakeGame):
            raise ValueError("A*-Agent can only operate in Battlesnake Game")
        if game.num_food() == 0:
            ra = np.asarray(random.choice(game.available_actions(player)))
            return ra, {}
        # gather snake positions
        blocked_pos = np.zeros(shape=(game.cfg.w, game.cfg.h), dtype=bool)
        start_pos = None
        for p in game.players_at_turn():
            body = game.player_pos(p)
            for pos in body:
                blocked_pos[pos[0], pos[1]] = True
            head = body[0]
            if p == player:
                start_pos = head
                continue
            for direction in range(4):
                new_pos = AStarAgent._new_pos(head, direction)
                if 0 <= new_pos[0] < game.cfg.w and 0 <= new_pos[1] < game.cfg.h:
                    blocked_pos[new_pos[0], new_pos[1]] = True
        # gather food positions
        food_pos: np.ndarray = game.food_pos()  # shape=(n, 2)
        results: list[tuple[bool, int, int]] = []
        found_food = False
        # do the search for all food pos
        if start_pos is None:
            raise Exception("Start position is None")
        for food_idx in range(food_pos.shape[0]):
            cur_res = AStarAgent._a_star_search(
                start_field=start_pos,
                search_field=(food_pos[food_idx, 0].item(), food_pos[food_idx, 1].item()),
                game=game,
                blocked_spaces=blocked_pos,
                player=player,
            )
            results.append(cur_res)
            if cur_res[0]:
                found_food = True
        # find the closest food
        sorted_results = sorted(results, key=lambda x: x[1], reverse=(not found_food))
        best = sorted_results[0]
        info = {'success': best[0], 'distance': best[1]}
        move = best[2]
        one_hot_probs = np.eye(4)[move]
        return one_hot_probs, info
