import math
import unittest

import torch

from src.game.battlesnake.battlesnake import BattleSnakeGame, LEFT, DOWN, UP
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig

import numpy as np

from src.game.battlesnake.battlesnake_enc import SimpleBattleSnakeEncodingConfig, num_layers_general, layers_per_player, \
    BestBattleSnakeEncodingConfig
from src.game.battlesnake.bootcamp.test_envs_11x11 import survive_on_11x11_4_player
from src.game.battlesnake.bootcamp.test_envs_3x3 import perform_choke_2_player
from src.game.battlesnake.bootcamp.test_envs_5x5 import perform_choke_5x5_4_player


class TestEncodingCPP(unittest.TestCase):
    def test_food_layer(self):
        init_food_pos = [[4, 3], [3, 2], [1, 2], [1, 1], [4, 4], [2, 4]]
        init_snake_pos = {0: [[0, 0]], 1: [[1, 0]]}
        ec = SimpleBattleSnakeEncodingConfig()
        gc = BattleSnakeConfig(init_food_pos=init_food_pos, init_snake_pos=init_snake_pos, ec=ec, init_snake_len=[3, 3],
                               num_players=2)
        env = BattleSnakeGame(cfg=gc)
        env.render()
        enc, _, _ = env.get_obs()
        for player in range(2):
            food_layer = enc[player, :, :, 0]
            for c in init_food_pos:
                self.assertEqual(1, food_layer[c[0] + 1, c[1] + 1].item())
        env.close()

    def test_snake_body_layer(self):
        init_food_pos = [[4, 4]]
        init_snake_pos = {0: [[2, 0], [1, 0], [0, 0]], 1: [[0, 1], [0, 2]]}
        ec = SimpleBattleSnakeEncodingConfig()
        gc = BattleSnakeConfig(init_food_pos=init_food_pos, init_snake_pos=init_snake_pos, ec=ec, num_players=2,
                               init_snake_len=[3, 3])
        env = BattleSnakeGame(cfg=gc)
        env.render()
        enc, _, _ = env.get_obs(symmetry=0)
        for player in range(2):
            for s in range(2):
                snake_layer = enc[player, :, :, num_layers_general(ec) + s * layers_per_player(ec)]
                if player == 0:
                    pos_list = init_snake_pos[s]
                else:
                    pos_list = init_snake_pos[1 - s]
                for c in pos_list:
                    val = snake_layer[c[0] + 1, c[1] + 1].item()
                    self.assertTrue(val > 0)
        env.close()

    def test_encoding_compressed(self):
        ec = BestBattleSnakeEncodingConfig()
        gc = BattleSnakeConfig(
            w=5,
            h=5,
            ec=ec,
            num_players=2,
            all_actions_legal=True,
        )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        actions = (LEFT, LEFT)
        env.step(actions)
        env.render()
        arr1, perm, inv_perm = env.get_obs()
        self.assertEqual(len(arr1.shape), 4)
        env.close()

    def test_1d_flat_encoding_single_view(self):
        ec = SimpleBattleSnakeEncodingConfig()
        ec.flatten = True
        gc = BattleSnakeConfig(ec=ec)
        env = BattleSnakeGame(cfg=gc)
        env.render()
        self.assertEqual(1, len(env.get_obs_shape()))
        obs, perm, inv_perm = env.get_obs()
        self.assertEqual(2, len(obs.shape))

    def test_symmetry_single_flat(self):
        ec = SimpleBattleSnakeEncodingConfig()
        ec.compress_enemies = True
        num_snakes = 6
        gc = BattleSnakeConfig(
            ec=ec,
            num_players=num_snakes,
            w=7,
            h=7,
        )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        self.assertEqual(num_snakes + 1 if num_snakes <= 4 else 1, env.num_food())
        self.assertEqual(8, env.get_symmetry_count())
        arr_list = []
        for s in range(8):
            arr1, perm, inv_perm = env.get_obs(symmetry=s)
            arr_list.append(arr1)
        for i in range(8):
            for j in range(i + 1, 8):
                self.assertFalse((arr_list[i] == arr_list[j]).all().item())
        env.close()  # MUY IMPORTANTE!!!

    def test_symmetry_single_only_rot(self):
        ec = SimpleBattleSnakeEncodingConfig()
        ec.compress_enemies = False
        gc = BattleSnakeConfig(
            ec=ec,
            num_players=6,
            w=7,
            h=7,
        )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        arr_list = []
        for s in range(8):
            arr1, perm, inv_perm = env.get_obs(symmetry=s)
            arr_list.append(arr1)
        for i in range(8):
            for j in range(i + 1, 8):
                self.assertFalse((arr_list[i] == arr_list[j]).all().item())
        env.close()  # MUY IMPORTANTE!!!

    def test_symmetry_all(self):
        ec = SimpleBattleSnakeEncodingConfig()
        ec.compress_enemies = False
        gc = BattleSnakeConfig(
            ec=ec,
            num_players=3,
            w=7,
            h=7,
        )
        env = BattleSnakeGame(cfg=gc)
        n = 8 * math.factorial(2)
        self.assertEqual(env.get_symmetry_count(), n)
        arr_list = []
        for s in range(n):
            arr1, perm, inv_perm = env.get_obs(symmetry=s)
            arr_list.append(arr1)
        for i in range(n):
            for j in range(i + 1, n):
                # this might fail due to randomness
                self.assertFalse((arr_list[i] == arr_list[j]).all().item())

    def test_permutation(self):
        conf = SimpleBattleSnakeEncodingConfig()
        conf.include_current_food = False
        conf.include_board = False
        conf.include_snake_head = False
        gc = BattleSnakeConfig(
            ec=conf,
            num_players=2,
            init_snake_pos={0: [[1, 0]], 1: [[2, 2]]},
            init_snake_len=[3, 3],
            init_food_pos=[],
            min_food=0,
            food_spawn_chance=0,
            w=3,
            h=3,
        )
        game = BattleSnakeGame(cfg=gc)
        legal_actions_0 = {0, 1, 3}
        legal_actions_1 = {2, 3}
        self.assertSetEqual(legal_actions_0, set(game.available_actions(0)))
        self.assertSetEqual(legal_actions_1, set(game.available_actions(1)))
        # Symmetry 0: original game
        arr0, perm0, inv_perm0 = game.get_obs(0)
        self.assertAlmostEqual(0.3, arr0[0, 2, 1, 0].item(), places=4)
        self.assertAlmostEqual(0.3, arr0[1, 2, 1, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr0[0, 3, 3, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr0[1, 3, 3, 0].item(), places=4)
        self.assertEqual(0, perm0[0])
        self.assertEqual(1, perm0[1])
        self.assertEqual(2, perm0[2])
        self.assertEqual(3, perm0[3])
        self.assertEqual(0, inv_perm0[0])
        self.assertEqual(1, inv_perm0[1])
        self.assertEqual(2, inv_perm0[2])
        self.assertEqual(3, inv_perm0[3])

        # Symmetry 1: Game mirrored on x-Axis
        arr1, perm1, inv_perm1 = game.get_obs(1)
        self.assertAlmostEqual(0.3, arr1[0, 2, 3, 0].item(), places=4)
        self.assertAlmostEqual(0.3, arr1[1, 2, 3, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr1[0, 3, 1, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr1[1, 3, 1, 0].item(), places=4)
        self.assertEqual(2, perm1[0])
        self.assertEqual(1, perm1[1])
        self.assertEqual(0, perm1[2])
        self.assertEqual(3, perm1[3])
        self.assertEqual(2, inv_perm1[0])
        self.assertEqual(1, inv_perm1[1])
        self.assertEqual(0, inv_perm1[2])
        self.assertEqual(3, inv_perm1[3])

        # Symmetry 2: Game rotated 90degree counterclockwise
        arr2, perm2, inv_perm2 = game.get_obs(2)
        self.assertAlmostEqual(0.3, arr2[0, 3, 2, 0].item(), places=4)
        self.assertAlmostEqual(0.3, arr2[1, 3, 2, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr2[0, 1, 3, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr2[1, 1, 3, 0].item(), places=4)
        self.assertEqual(3, perm2[0])
        self.assertEqual(0, perm2[1])
        self.assertEqual(1, perm2[2])
        self.assertEqual(2, perm2[3])
        self.assertEqual(1, inv_perm2[0])
        self.assertEqual(2, inv_perm2[1])
        self.assertEqual(3, inv_perm2[2])
        self.assertEqual(0, inv_perm2[3])

        # Symmetry 3: Game rotated 90degree counterclockwise and afterwards flipped along x-Axis
        arr3, perm3, inv_perm3 = game.get_obs(3)
        self.assertAlmostEqual(0.3, arr3[0, 3, 2, 0].item(), places=4)
        self.assertAlmostEqual(0.3, arr3[1, 3, 2, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr3[0, 1, 1, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr3[1, 1, 1, 0].item(), places=4)
        self.assertEqual(3, perm3[0])
        self.assertEqual(2, perm3[1])
        self.assertEqual(1, perm3[2])
        self.assertEqual(0, perm3[3])
        self.assertEqual(3, inv_perm3[0])
        self.assertEqual(2, inv_perm3[1])
        self.assertEqual(1, inv_perm3[2])
        self.assertEqual(0, inv_perm3[3])

        # Symmetry 4: Game rotated 180degree counterclockwise
        arr4, perm4, inv_perm4 = game.get_obs(4)
        self.assertAlmostEqual(0.3, arr4[0, 2, 3, 0].item(), places=4)
        self.assertAlmostEqual(0.3, arr4[1, 2, 3, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr4[0, 1, 1, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr4[1, 1, 1, 0].item(), places=4)
        self.assertEqual(2, perm4[0])
        self.assertEqual(3, perm4[1])
        self.assertEqual(0, perm4[2])
        self.assertEqual(1, perm4[3])
        self.assertEqual(2, inv_perm4[0])
        self.assertEqual(3, inv_perm4[1])
        self.assertEqual(0, inv_perm4[2])
        self.assertEqual(1, inv_perm4[3])

        # Symmetry 5: Game rotated 180degree counterclockwise and afterwards flipped along x-Axis
        arr5, perm5, inv_perm5 = game.get_obs(5)
        self.assertAlmostEqual(0.3, arr5[0, 2, 1, 0].item(), places=4)
        self.assertAlmostEqual(0.3, arr5[1, 2, 1, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr5[0, 1, 3, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr5[1, 1, 3, 0].item(), places=4)
        self.assertEqual(0, perm5[0])
        self.assertEqual(3, perm5[1])
        self.assertEqual(2, perm5[2])
        self.assertEqual(1, perm5[3])
        self.assertEqual(0, inv_perm5[0])
        self.assertEqual(3, inv_perm5[1])
        self.assertEqual(2, inv_perm5[2])
        self.assertEqual(1, inv_perm5[3])

        # Symmetry 6: Game rotated 270degree counterclockwise
        arr6, perm6, inv_perm6 = game.get_obs(6)
        self.assertAlmostEqual(0.3, arr6[0, 1, 2, 0].item(), places=4)
        self.assertAlmostEqual(0.3, arr6[1, 1, 2, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr6[0, 3, 1, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr6[1, 3, 1, 0].item(), places=4)
        self.assertEqual(1, perm6[0])
        self.assertEqual(2, perm6[1])
        self.assertEqual(3, perm6[2])
        self.assertEqual(0, perm6[3])
        self.assertEqual(3, inv_perm6[0])
        self.assertEqual(0, inv_perm6[1])
        self.assertEqual(1, inv_perm6[2])
        self.assertEqual(2, inv_perm6[3])

        # Symmetry 7: Game rotated 270degree counterclockwise and afterwards flipped along x-Axis
        arr7, perm7, inv_perm7 = game.get_obs(7)
        self.assertAlmostEqual(0.3, arr7[0, 1, 2, 0].item(), places=4)
        self.assertAlmostEqual(0.3, arr7[1, 1, 2, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr7[0, 3, 3, 1].item(), places=4)
        self.assertAlmostEqual(0.3, arr7[1, 3, 3, 0].item(), places=4)
        self.assertEqual(1, perm7[0])
        self.assertEqual(0, perm7[1])
        self.assertEqual(3, perm7[2])
        self.assertEqual(2, perm7[3])
        self.assertEqual(1, inv_perm7[0])
        self.assertEqual(0, inv_perm7[1])
        self.assertEqual(3, inv_perm7[2])
        self.assertEqual(2, inv_perm7[3])

    def test_centered_simple(self):
        conf = SimpleBattleSnakeEncodingConfig()
        conf.centered = True
        gc = BattleSnakeConfig(
            ec=conf,
            num_players=3,
            w=5,
            h=5,
        )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        arr1, perm, inv_perm = env.get_obs(symmetry=0)
        np_arr = np.asarray(arr1)
        self.assertEqual(-31, np.sum(np_arr[0, :, :, 1]))

        env.step((0, 0, 0))
        env.render()
        arr1, perm, inv_perm = env.get_obs(symmetry=0)
        np_arr = np.asarray(arr1)
        self.assertEqual(-31, np.sum(np_arr[0, :, :, 1]))
        print(np_arr.shape)

    def test_centered_all(self):
        conf = BestBattleSnakeEncodingConfig()
        conf.centered = True
        init_snake_pos = {0: [[0, 0], [1, 0], [2, 0], [3, 0]], 1: [[0, 3], [0, 4], [1, 4], [2, 4]]}
        init_snake_len = [4, 4]
        init_food_pos = [[0, 1]]
        gc = BattleSnakeConfig(
            ec=conf,
            num_players=2,
            w=5,
            h=5,
            init_snake_pos=init_snake_pos,
            init_snake_len=init_snake_len,
            init_food_pos=init_food_pos,
        )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        arr1, perm, inv_perm = env.get_obs(symmetry=0)
        np_arr = np.asarray(arr1)
        self.assertEqual(-31, np.sum(np_arr[0, :, :, 2]))

    def test_distance_simple(self):
        conf = SimpleBattleSnakeEncodingConfig()
        conf.include_distance_map = True
        gc = BattleSnakeConfig(
            ec=conf,
            num_players=3,
            w=5,
            h=5,
        )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        arr1, perm, inv_perm = env.get_obs(symmetry=0)
        np_arr = np.asarray(arr1)
        q = 1

    def test_distance_centered(self):
        conf = SimpleBattleSnakeEncodingConfig()
        conf.include_distance_map = True
        conf.centered = True
        gc = BattleSnakeConfig(
            ec=conf,
            num_players=3,
            w=5,
            h=5,
        )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        arr1, perm, inv_perm = env.get_obs(symmetry=0)
        np_arr = np.asarray(arr1)
        q = 1

    def test_all_in_game(self):
        init_snake_pos = {0: [[0, 0], [0, 1], [0, 2]], 1: [[2, 2], [2, 1], [2, 0]]}
        init_snake_len = [3, 6]
        gc = BattleSnakeConfig(num_players=2, w=3, h=3, init_snake_pos=init_snake_pos, init_snake_len=init_snake_len,
                               init_food_pos=[], ec=BestBattleSnakeEncodingConfig(), min_food=0)
        gc.ec.include_tail_distance = True
        game = BattleSnakeGame(gc)
        game.render()
        r = game.area_control()
        ac, fd, td, rt, rf = r["area_control"], r["food_distance"], r["tail_distance"], r["tail_reachable"], \
            r["food_reachable"]
        print(ac)
        enc = game.get_obs(0)[0]
        print(enc.shape)

    def test_centered_4_player(self):
        gc = survive_on_11x11_4_player(centered=True)
        game = BattleSnakeGame(gc)
        game.render()

        enc = game.get_obs(0)[0]
        print(enc.shape)

        game.step((UP, UP, UP, UP))
        game.render()
        enc = game.get_obs(0)[0]
        print(enc.shape)

    def test_tail_distance_encoding(self):
        init_snake_pos = {0: [[0, 1], [0, 0], [1, 0]], 1: [[4, 2], [4, 3], [4, 4]]}
        init_food_pos = []
        init_snake_len = [3, 3]
        gc = BattleSnakeConfig(w=5, h=5, num_players=2, init_food_pos=init_food_pos, init_snake_pos=init_snake_pos,
                               min_food=0, food_spawn_chance=0, init_snake_len=init_snake_len,
                               ec=BestBattleSnakeEncodingConfig())
        gc.ec.include_tail_distance = True
        game = BattleSnakeGame(gc)
        enc = game.get_obs(0)[0]
        print(enc.shape)

    def test_num_food_on_board(self):
        init_snake_pos = {0: [[0, 1], [0, 0], [1, 0]], 1: [[4, 2], [4, 3], [4, 4]]}
        init_food_pos = [[2, 2], [2, 3], [0, 2]]
        init_snake_len = [3, 3]
        ec = SimpleBattleSnakeEncodingConfig()
        ec.include_num_food_on_board = True
        gc = BattleSnakeConfig(w=5, h=5, num_players=2, init_food_pos=init_food_pos, init_snake_pos=init_snake_pos,
                               min_food=0, food_spawn_chance=0, init_snake_len=init_snake_len,
                               ec=ec)
        game = BattleSnakeGame(gc)
        food_layer_idx = 2
        game.render()

        obs, _, _ = game.get_obs()
        for player in range(2):
            for x in range(obs.shape[1]):
                for y in range(obs.shape[2]):
                    self.assertAlmostEqual(.3, obs[player, x, y, food_layer_idx], places=4)

        game.step((UP, LEFT))
        game.render()
        obs, _, _ = game.get_obs()
        for player in range(2):
            for x in range(obs.shape[1]):
                for y in range(obs.shape[2]):
                    self.assertAlmostEqual(.2, obs[player, x, y, food_layer_idx], places=4)

        game.step((UP, LEFT))
        game.render()
        obs, _, _ = game.get_obs()
        for player in range(2):
            for x in range(obs.shape[1]):
                for y in range(obs.shape[2]):
                    self.assertAlmostEqual(.1, obs[player, x, y, food_layer_idx], places=4)

    def test_fixed_food_spawn_rate(self):
        init_snake_pos = {0: [[0, 1], [0, 0], [1, 0]], 1: [[4, 2], [4, 3], [4, 4]]}
        init_food_pos = []
        init_snake_len = [3, 3]
        gc = BattleSnakeConfig(w=5, h=5, num_players=2, init_food_pos=init_food_pos, init_snake_pos=init_snake_pos,
                               min_food=0, food_spawn_chance=0, init_snake_len=init_snake_len,
                               ec=BestBattleSnakeEncodingConfig())
        gc.ec.fixed_food_spawn_chance = 100
        gc.ec.include_num_food_on_board = True

        food_layer_idx = 1
        game = BattleSnakeGame(gc)
        game.render()

        obs, _, _ = game.get_obs()
        for player in range(2):
            for x in range(obs.shape[1]):
                for y in range(obs.shape[2]):
                    check = obs[player, x, y, food_layer_idx] == 0 or obs[player, x, y, food_layer_idx] > .065
                    self.assertTrue(check)

    def test_temperature_single(self):
        game_cfg = perform_choke_2_player(centered=True, fully_connected=False)
        game_cfg.ec.temperature_input = True
        game_cfg.ec.single_temperature_input = True
        game = BattleSnakeGame(game_cfg)
        obs, _, _ = game.get_obs(temperatures=[2])

        temperature_layer = obs[..., 2]
        self.assertTrue((temperature_layer == .2).all().item())

    def test_temperature_multiple(self):
        game_cfg = perform_choke_5x5_4_player(centered=True)
        game_cfg.ec.temperature_input = True
        game_cfg.ec.single_temperature_input = False
        game_cfg.ec.compress_enemies = False
        game = BattleSnakeGame(game_cfg)
        obs, _, _ = game.get_obs(temperatures=[1, 2, 3, 4])

        temperature_layer1 = obs[0, :, :, 5]
        self.assertTrue((temperature_layer1 == .2).all().item())
        temperature_layer2 = obs[0, :, :, 8]
        self.assertTrue((temperature_layer2 == .3).all().item())
        temperature_layer3 = obs[0, :, :, 11]
        self.assertTrue((temperature_layer3 == .4).all().item())

        temperature_layer1 = obs[1, :, :, 5]
        self.assertTrue((temperature_layer1 == .1).all().item())
        temperature_layer2 = obs[1, :, :, 8]
        self.assertTrue((temperature_layer2 == .3).all().item())
        temperature_layer3 = obs[1, :, :, 11]
        self.assertTrue((temperature_layer3 == .4).all().item())

        temperature_layer1 = obs[2, :, :, 5]
        self.assertTrue((temperature_layer1 == .1).all().item())
        temperature_layer2 = obs[2, :, :, 8]
        self.assertTrue((temperature_layer2 == .2).all().item())
        temperature_layer3 = obs[2, :, :, 11]
        self.assertTrue((temperature_layer3 == .4).all().item())

        temperature_layer1 = obs[3, :, :, 5]
        self.assertTrue((temperature_layer1 == .1).all().item())
        temperature_layer2 = obs[3, :, :, 8]
        self.assertTrue((temperature_layer2 == .2).all().item())
        temperature_layer3 = obs[3, :, :, 11]
        self.assertTrue((temperature_layer3 == .3).all().item())

    def test_temperature_multiple_compressed(self):
        game_cfg = perform_choke_5x5_4_player(centered=True)
        game_cfg.ec.temperature_input = True
        game_cfg.ec.single_temperature_input = False
        game_cfg.ec.compress_enemies = True
        game = BattleSnakeGame(game_cfg)
        game.step((0, 0, 0, 0))
        game.step((0, 0, 0, 0))
        game.step((0, 0, 0, 0))
        obs, _, _ = game.get_obs(temperatures=[1, 2, 3, 4])
        tl = obs[0, :, :, -1]
        self.assertTrue(
            np.logical_or(
                np.logical_or(
                    np.logical_or(
                        (tl == 0.2), (tl == 0.3)
                    ),
                    (tl == 0.4)),
                (tl == 0))
            .all().item()
        )
        tl = obs[1, :, :, -1]
        self.assertTrue(
            np.logical_or(
                np.logical_or(
                    np.logical_or(
                        (tl == 0.1), (tl == 0.3)
                    ),
                    (tl == 0.4)),
                (tl == 0))
            .all().item()
        )
        tl = obs[2, :, :, -1]
        self.assertTrue(
            np.logical_or(
                np.logical_or(
                    np.logical_or(
                        (tl == 0.2), (tl == 0.1)
                    ),
                    (tl == 0.4)),
                (tl == 0))
            .all().item()
        )
        tl = obs[3, :, :, -1]
        self.assertTrue(
            np.logical_or(
                np.logical_or(
                    np.logical_or(
                        (tl == 0.2), (tl == 0.3)
                    ),
                    (tl == 0.1)),
                (tl == 0))
            .all().item()
        )
