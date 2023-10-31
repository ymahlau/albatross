import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame
from src.game.battlesnake.battlesnake import UP
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.game.battlesnake.battlesnake_enc import BestBattleSnakeEncodingConfig
from src.game.battlesnake.bootcamp import survive_on_7x7_4_player_royale


class TestHazards(unittest.TestCase):
    def test_hazards_simple(self):
        init_snake_pos = {0: [[0, 0]], 1: [[10, 0]]}
        init_snake_len = [3, 3]
        init_food_pos = []
        gc = BattleSnakeConfig(num_players=2, w=11, h=11, min_food=0, food_spawn_chance=0,
                               init_snake_pos=init_snake_pos,
                               init_snake_len=init_snake_len, init_food_pos=init_food_pos, wrapped=False, royale=True,
                               shrink_n_turns=1, all_actions_legal=True)
        game = BattleSnakeGame(gc)
        game.render()
        hazard_arr = game.get_hazards()
        print(hazard_arr)
        print('##############################\n\n')

        for i in range(10):
            if game.is_terminal():
                break
            game.step((UP, UP))
            game.render()
            print(f"{game.player_healths()=}")
            for p in range(2):
                print(f"{p}: {game.available_actions(p)}")
                print(f"{p}: {game.player_pos(p)}")
            hazard_arr = game.get_hazards()
            print(hazard_arr)
            print(game.player_healths())
            print('##############################\n\n')

    def test_hazard_spawning(self):
        init_snake_pos = {0: [[0, 0]], 1: [[10, 0]]}
        init_snake_len = [3, 3]
        init_food_pos = []
        # init_hazards = np.random.choice([True, False], size=(11, 11), p=[0.5, 0.5])
        init_hazards = [[3, 5], [2, 0], [1, 4], [3, 0]]
        gc = BattleSnakeConfig(num_players=2, w=11, h=11, min_food=0, food_spawn_chance=0,
                               init_snake_pos=init_snake_pos,
                               init_snake_len=init_snake_len, init_food_pos=init_food_pos, wrapped=False, royale=True,
                               shrink_n_turns=2, init_hazards=init_hazards)
        game = BattleSnakeGame(gc)
        game.render()
        hazards = game.get_hazards()
        print(hazards)
        for hazard_tile in init_hazards:
            self.assertTrue(hazards[hazard_tile[0], hazard_tile[1]])

    def test_hazard_encoding(self):
        init_snake_pos = {0: [[4, 0]], 1: [[7, 2]]}
        init_snake_len = [3, 3]
        init_food_pos = []
        # init_hazards = np.random.choice([True, False], size=(11, 11), p=[0.5, 0.5])
        ec = BestBattleSnakeEncodingConfig()
        ec.include_hazards = True
        init_hazards = [[3, 5], [2, 0], [1, 4], [3, 0]]
        gc = BattleSnakeConfig(num_players=2, w=11, h=11, min_food=0, food_spawn_chance=0,
                               init_snake_pos=init_snake_pos,
                               init_snake_len=init_snake_len, init_food_pos=init_food_pos, wrapped=False, royale=True,
                               shrink_n_turns=2, init_hazards=init_hazards, ec=ec)
        game = BattleSnakeGame(gc)
        game.render()
        for i in range(3):
            game.step((UP, UP))
            game.render()
        enc = game.get_obs(0)[0].numpy()
        print(enc.shape)

    def test_hazard_legal_actions(self):
        game_cfg = survive_on_7x7_4_player_royale()
        game = BattleSnakeGame(game_cfg)
        game.render()
        game.step((1, 1, 1, 1))
        game.render()
        for p in range(4):
            print(f"{p}: {game.available_actions(p)}")
            print(f"{p}: {game.player_pos(p)}")

