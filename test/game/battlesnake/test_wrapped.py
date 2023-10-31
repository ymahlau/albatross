import unittest

from src.game.battlesnake.battlesnake import LEFT, UP, DOWN, RIGHT
from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.battlesnake.battlesnake_enc import BestBattleSnakeEncodingConfig
from src.game.battlesnake.bootcamp import perform_choke_wrapped


class TestWrapped(unittest.TestCase):
    def test_simple_duel(self):
        init_snake_pos = {0: [[0, 0]], 1: [[2, 2]]}
        init_snake_len = [3, 3]
        gc = BattleSnakeConfig(w=3, h=3, num_players=2, min_food=0, food_spawn_chance=0, all_actions_legal=False,
                               wrapped=True,
                               init_snake_pos=init_snake_pos, init_snake_len=init_snake_len, init_food_pos=[])
        gc.ec.include_board = False
        game = BattleSnakeGame(gc)
        game.render()
        self.assertEqual(4, len(game.available_actions(0)))
        self.assertEqual(4, len(game.available_actions(1)))

        game.step((LEFT, RIGHT))
        self.assertEqual(2, len(game.available_actions(0)))
        self.assertEqual(2, len(game.available_actions(1)))
        game.render()

    def test_kill_body(self):
        init_snake_pos = {0: [[0, 0]], 1: [[2, 0]]}
        init_snake_len = [3, 3]
        gc = BattleSnakeConfig(w=3, h=3, num_players=2, min_food=0, food_spawn_chance=0, all_actions_legal=True,
                               wrapped=True,
                               init_snake_pos=init_snake_pos, init_snake_len=init_snake_len, init_food_pos=[], )
        gc.ec.include_board = False
        game = BattleSnakeGame(gc)
        game.render()
        self.assertEqual(4, len(game.available_actions(0)))
        self.assertEqual(4, len(game.available_actions(1)))

        game.step((LEFT, UP))
        game.render()
        self.assertEqual(1, game.num_players_alive())

    def test_kill_head_swap(self):
        init_snake_pos = {0: [[0, 1]], 1: [[2, 1]]}
        init_snake_len = [5, 5]
        gc = BattleSnakeConfig(w=3, h=3, num_players=2, min_food=0, food_spawn_chance=0, all_actions_legal=True,
                               wrapped=True,
                               init_snake_pos=init_snake_pos, init_snake_len=init_snake_len, init_food_pos=[], )
        gc.ec.include_board = False
        game = BattleSnakeGame(gc)
        game.render()
        self.assertEqual(4, len(game.available_actions(0)))
        self.assertEqual(4, len(game.available_actions(1)))

        game.step((LEFT, RIGHT))
        game.render()
        self.assertEqual(0, game.num_players_alive())

    def test_area_control(self):
        init_snake_pos = {0: [[0, 0]], 1: [[1, 1]]}
        init_snake_len = [5, 5]
        gc = BattleSnakeConfig(w=3, h=3, num_players=2, min_food=0, food_spawn_chance=0, all_actions_legal=True,
                               wrapped=True,
                               init_snake_pos=init_snake_pos, init_snake_len=init_snake_len, init_food_pos=[], )
        gc.ec.include_board = False
        game = BattleSnakeGame(gc)
        game.render()
        r = game.area_control()
        ac, fd, td, rt, rf = r["area_control"], r["food_distance"], r["tail_distance"], r["tail_reachable"], \
            r["food_reachable"]
        print(ac)
        self.assertEqual(2, ac[0])
        self.assertEqual(2, ac[1])

    def test_area_control_2(self):
        init_snake_pos = {0: [[0, 0]], 1: [[0, 1]]}
        init_snake_len = [3, 3]
        gc = BattleSnakeConfig(w=3, h=3, num_players=2, min_food=0, food_spawn_chance=0, all_actions_legal=True,
                               wrapped=True,
                               init_snake_pos=init_snake_pos, init_snake_len=init_snake_len, init_food_pos=[], )
        gc.ec.include_board = False
        game = BattleSnakeGame(gc)
        game.render()
        r = game.area_control()
        ac, fd, td, rt, rf = r["area_control"], r["food_distance"], r["tail_distance"], r["tail_reachable"], \
            r["food_reachable"]
        print(ac)
        self.assertEqual(2, ac[0])
        self.assertEqual(2, ac[1])

    def test_wrapped_encoding(self):
        init_snake_pos = {0: [[0, 0]], 1: [[1, 0]]}
        init_snake_len = [3, 3]
        gc = BattleSnakeConfig(w=3, h=3, num_players=2, min_food=0, food_spawn_chance=0, all_actions_legal=True,
                               wrapped=True,
                               init_snake_pos=init_snake_pos, init_snake_len=init_snake_len, init_food_pos=[[0, 1]],
                               ec=BestBattleSnakeEncodingConfig())
        gc.ec.include_board = False
        gc.ec.include_distance_map = True
        game = BattleSnakeGame(gc)
        game.render()
        enc, _, _ = game.get_obs(0)
        print(game.player_lengths())
        print(enc.shape)

        game.step((UP, LEFT))
        game.render()

    def test_wrapped_encoding_centered(self):
        init_snake_pos = {0: [[0, 0]], 1: [[1, 0]]}
        init_snake_len = [3, 3]
        init_food_pos = [[1, 1]]
        ec = BestBattleSnakeEncodingConfig()
        ec.centered = True
        ec.include_board = False
        gc = BattleSnakeConfig(w=3, h=3, num_players=2, min_food=0, food_spawn_chance=0, all_actions_legal=True,
                               wrapped=True,
                               init_snake_pos=init_snake_pos, init_snake_len=init_snake_len,
                               init_food_pos=init_food_pos,
                               ec=ec, )
        game = BattleSnakeGame(gc)
        # self.assertEqual(3, game.get_obs_shape()[0])
        # self.assertEqual(3, game.get_obs_shape()[1])
        game.render()
        enc, _, _ = game.get_obs(0)
        print(enc.shape)

        game.step((UP, UP))
        game.render()
        enc, _, _ = game.get_obs(0)
        print(enc.shape)

    def test_perform_choke_wrapped(self):
        gc = perform_choke_wrapped(False, False)
        game = BattleSnakeGame(gc)
        game.render()
        self.assertEqual(2, game.num_players_alive())

        game.step((LEFT, DOWN))
        game.render()
        self.assertEqual(2, game.num_players_alive())

        game.step((DOWN, DOWN))
        game.render()
        self.assertEqual(2, game.num_players_alive())

        game.step((DOWN, DOWN))
        game.render()
        self.assertEqual(2, game.num_players_alive())

        game.step((RIGHT, DOWN))
        game.render()
        self.assertEqual(2, game.num_players_alive())

        game.step((UP, DOWN))
        game.render()
        self.assertEqual(1, game.num_players_alive())
