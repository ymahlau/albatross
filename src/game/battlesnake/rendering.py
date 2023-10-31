import math
from pathlib import Path

import pygame

from environment.Battlesnake.model.Position import Position
from environment.Battlesnake.model.Snake import Snake
from environment.Battlesnake.model.board_state import BoardState
from environment.battlesnake_environment import BattlesnakeEnvironment
from environment.renderer import GameRenderer
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.misc.const import COLORS


def render_game_info(game: BattleSnakeGame):
    health = game.player_healths()
    print('#########################')
    for player in game.players_at_turn():
        pos = game.player_pos(player)
        print(f"Position {player}: {pos}")
        print(f"Health {player}: {health[player]}")
    game.render()

def render_pygame(
        game: BattleSnakeGame,
):
    renderer = GameRenderer(
        max_game_width=game.cfg.w,
        max_game_height=game.cfg.h,
        max_num_snakes=game.cfg.num_players,
        game_mode='Duels',
    )
    snake_list = []
    for p in game.players_alive():
        positions = game.player_pos(p)
        pos_list = [Position(pos[0], pos[1]) for pos in positions]
        cur_color_list = [round(c * 254) for c in COLORS[p]]
        cur_color_tpl: tuple[int, int, int] = cur_color_list[0], cur_color_list[1], cur_color_list[2]
        cur_head = None
        if p == 0:
            cur_head = 'all-seeing'
        snake = Snake(
            snake_id=f'{p}',
            health=game.player_healths()[p],
            body=pos_list,
            snake_color=cur_color_tpl,
            snake_head=cur_head,
        )
        snake_list.append(snake)
    board = BoardState(
        turn=game.turns_played,
        width=game.cfg.w,
        height=game.cfg.h,
        snakes=snake_list,
    )
    renderer.display(board)
    pygame.time.wait(60000)


