import unittest

from src.game.battlesnake.battlesnake import UP, LEFT, RIGHT, DOWN
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig


class TestFixedSpawning(unittest.TestCase):

    def test_init_health(self):
        snake_spawns = {0: [[0, 0]], 1: [[1, 0]]}
        snake_health = [3, 2]
        food_pos = []
        gc = BattleSnakeConfig(h=11, w=11, num_players=2, min_food=0, food_spawn_chance=0, init_snake_pos=snake_spawns,
                               init_snake_health=snake_health, init_food_pos=food_pos, init_turns_played=3,
                               init_snake_len=[3, 3], all_actions_legal=True)
        env = BattleSnakeGame(cfg=gc)
        actions = (UP, UP)
        env.render()
        self.assertEqual(3, env.turns_played)
        self.assertEqual(env.num_players_alive(), 2)
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 2)
        self.assertEqual(4, env.turns_played)
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 1)
        self.assertEqual(5, env.turns_played)
        env.close()

    def test_init_length(self):
        snake_spawns = {0: [[0, 0]], 1: [[1, 1]]}
        init_snake_len = [2, 2]
        food_pos = []
        gc = BattleSnakeConfig(h=2, w=2, num_players=2, min_food=0, food_spawn_chance=0, init_snake_pos=snake_spawns,
                               init_snake_len=init_snake_len, init_food_pos=food_pos)
        env = BattleSnakeGame(cfg=gc)
        lens = env.player_lengths()
        self.assertEqual(2, lens[0])
        self.assertEqual(2, lens[1])
        env.render()
        self.assertEqual(env.num_players_alive(), 2)
        self.assertEqual(0, env.turns_played)

        env.step((RIGHT, LEFT))
        env.render()
        self.assertEqual(env.num_players_alive(), 2)
        self.assertEqual(1, env.turns_played)

        env.step((UP, DOWN))
        env.render()
        self.assertEqual(env.num_players_alive(), 2)
        self.assertEqual(2, env.turns_played)

        env.step((LEFT, RIGHT))
        env.render()
        self.assertEqual(env.num_players_alive(), 2)
        self.assertEqual(3, env.turns_played)

    def test_init_length_h2h(self):
        snake_spawns = {0: [[0, 0]], 1: [[2, 0]]}
        init_snake_len = [1, 2]
        food_pos = []
        gc = BattleSnakeConfig(h=1, w=3, num_players=2, min_food=0, food_spawn_chance=0, init_snake_pos=snake_spawns,
                               init_snake_len=init_snake_len, init_food_pos=food_pos)
        env = BattleSnakeGame(cfg=gc)
        lens = env.player_lengths()
        self.assertEqual(1, lens[0])
        self.assertEqual(2, lens[1])
        env.render()
        self.assertEqual(env.num_players_alive(), 2)

        env.step((RIGHT, LEFT))
        env.render()
        self.assertEqual(env.num_players_alive(), 1)
