import ctypes as ct
import time
import unittest

from src.game.battlesnake.battlesnake import BattleSnakeGame, DOWN, LEFT, UP, RIGHT
from src.game.battlesnake.battlesnake_conf import BattleSnakeConfig
from src.cpp.lib import CPP_LIB
from src.game.battlesnake.bootcamp import perform_choke_2_player
from src.game.initialization import get_game_from_config


class TestStep(unittest.TestCase):

    def test_wall_kills(self):
        snake_spawns = {0: [[0, 1]], 1: [[1, 0]], 2: [[1, 2]], 3: [[2, 1]]}
        food_pos = [[1, 1]]
        gc = BattleSnakeConfig(
            w=3,
            h=3,
            num_players=4,
            min_food=0,
            food_spawn_chance=0,
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            init_snake_len=[3, 3, 3, 3],
            all_actions_legal=True,
        )
        env = get_game_from_config(gc)
        env.render()
        for i in range(4):
            self.assertEqual(4, len(env.available_actions(i)))
        self.assertEqual(256, len(env.available_joint_actions()))
        self.assertEqual(4, len(env.players_alive()))
        self.assertEqual(4, env.num_players_alive())
        self.assertEqual(4, len(env.players_at_turn()))
        self.assertEqual(4, env.num_players_at_turn())
        self.assertFalse(env.is_terminal())

        actions = (LEFT, DOWN, UP, RIGHT)
        rewards, done, _ = env.step(actions)
        env.render()
        self.assertEqual(0, len(env.players_alive()))
        self.assertEqual(0, len(env.players_at_turn()))
        self.assertTrue(env.is_terminal())
        outcome = env.get_cum_rewards()
        for i in range(4):
            self.assertEqual(0, outcome[i])
        env.close()  # MUY IMPORTANTE!!!

    def test_available_actions(self):
        snake_spawns = {0: [[0, 1]], 1: [[1, 0]], 2: [[1, 2]], 3: [[2, 1]]}
        food_pos = [[1, 1]]
        gc = BattleSnakeConfig(
            w=3,
            h=3,
            num_players=4,
            min_food=0,
            food_spawn_chance=0,
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            init_snake_len=[3, 3, 3, 3],
            all_actions_legal=False,
        )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        for i in range(4):
            self.assertEqual(3, len(env.available_actions(i)))
        self.assertEqual(81, len(env.available_joint_actions()))
        gc.all_actions_legal = True
        env2 = BattleSnakeGame(gc)
        env2.step((UP, UP, UP, UP))
        env.render()
        for player in [0, 1, 3]:
            self.assertEqual(4, len(env2.available_actions(player)))
        self.assertEqual(0, len(env2.available_actions(2)))
        self.assertEqual(64, len(env2.available_joint_actions()))

    def test_snake_h2h_same_size(self):
        snake_spawns = {0: [[0, 0]], 1: [[4, 0]]}
        food_pos = []
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=2,
            w=5,
            h=1,
            init_snake_len=[3, 3],
        )
        env = BattleSnakeGame(cfg=gc)
        actions = (RIGHT, LEFT)
        env.render()
        env.step(actions)
        env.render()
        self.assertEqual(2, env.num_players_alive())
        env.step(actions)
        env.render()
        self.assertEqual(0, env.num_players_alive())
        self.assertTrue(env.is_terminal())
        env.close()

    def test_snake_h2h_head_swap(self):
        snake_spawns = {0: [[0, 0]], 1: [[3, 0]]}
        food_pos = []
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=2,
            w=4,
            h=1,
            init_snake_len=[3, 3],
            all_actions_legal=True,
        )
        env = BattleSnakeGame(cfg=gc)
        actions = (RIGHT, LEFT)
        env.render()
        env.step(actions)
        env.render()
        self.assertEqual(2, env.num_players_alive())
        env.step(actions)
        env.render()
        self.assertEqual(0, env.num_players_alive())
        env.close()

    def test_no_food_spawning(self):
        # snake should die after exactly 100 moves due to starvation
        snake_spawns = {0: [[0, 0]]}
        food_pos = []
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=1,
            w=101,
            h=1,
            init_snake_len=[3],
            all_actions_legal=True
        )
        env = BattleSnakeGame(cfg=gc)
        actions = (RIGHT,)
        env.render()
        for i in range(99):
            env.step(actions)
        self.assertEqual(env.num_players_alive(), 1)
        env.render()
        self.assertFalse(env.is_terminal())
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 0)
        env.close()

    def test_food_spawning(self):
        # with 10% food spawn chance snake should easily survive
        snake_spawns = {0: [[0, 0]]}
        food_pos = []
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=10,
            num_players=1,
            w=101,
            h=1,
            init_snake_len=[3],
        )
        env = BattleSnakeGame(cfg=gc)
        actions = (RIGHT,)
        env.render()
        for i in range(99):
            env.step(actions)
        self.assertEqual(env.num_players_alive(), 1)
        env.render()
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 1)
        env.close()

    def test_bigger_snake_eats_smaller(self):
        snake_spawns = {0: [[0, 0]], 1: [[4, 0]]}
        food_pos = [[3, 0]]
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=2,
            w=5,
            h=1,
            init_snake_len=[3, 3],
        )
        env = BattleSnakeGame(cfg=gc)
        actions = (RIGHT, LEFT)
        env.render()
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 2)
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 1)
        outcome = env.get_cum_rewards()
        self.assertEqual(-1, outcome[0])
        self.assertEqual(1, outcome[1])
        env.close()

    def test_bigger_snake_eats_smaller_h2h_swap(self):
        # if the heads of two snakes swap places it counts as body collision, not head to head.
        # therefore, both snakes should die
        snake_spawns = {0: [[0, 0]], 1: [[3, 0]]}
        food_pos = [[2, 0]]
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=2,
            w=4,
            h=1,
            init_snake_len=[3, 3],
            all_actions_legal=True,
        )
        env = BattleSnakeGame(cfg=gc)
        actions = (RIGHT, LEFT)
        env.render()
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 2)
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 0)
        env.close()

    def test_body_collision(self):
        # snake dies if it hits body of other snake
        snake_spawns = {0: [[0, 0]], 1: [[2, 0]]}
        food_pos = [[1, 1]]
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=2,
            w=5,
            h=5,
            init_snake_len=[3, 3],
            all_actions_legal=True,
        )
        env = BattleSnakeGame(cfg=gc)
        actions = (UP, LEFT)
        env.render()
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 2)
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 1)
        outcome = env.get_cum_rewards()
        self.assertEqual(1, outcome[0])
        self.assertEqual(-1, outcome[1])
        env.close()

    def test_tail_chase(self):
        # snake should be able to chase the tail of another snake
        snake_spawns = {0: [[0, 0]], 1: [[3, 0]]}
        food_pos = []
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=2,
            w=105,
            h=1,
            init_snake_len=[3, 3],
        )
        env = BattleSnakeGame(cfg=gc)
        actions = (RIGHT, RIGHT)
        env.render()
        for i in range(99):
            env.step(actions)
            self.assertEqual(1, len(env.available_joint_actions()))
        env.render()
        self.assertEqual(env.num_players_alive(), 2)
        env.close()

    def test_multi_head_collision_same_len(self):
        snake_spawns = {0: [[1, 0]], 1: [[0, 1]], 2: [[1, 2]], 3: [[2, 1]]}
        food_pos = []
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=4,
            w=3,
            h=3,
            init_snake_len=[3, 3, 3, 3],
        )
        env = BattleSnakeGame(cfg=gc)
        actions = (UP, RIGHT, DOWN, LEFT)
        env.render()
        self.assertEqual(env.num_players_alive(), 4)
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 0)
        outcome = env.get_cum_rewards()
        for i in range(4):
            self.assertEqual(0, outcome[i])
        env.close()

    def test_multi_head_collision_different_len(self):
        snake_spawns = {0: [[2, 0]], 1: [[0, 2]], 2: [[2, 4]], 3: [[4, 2]]}
        food_pos = [[2, 1]]
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=4,
            w=5,
            h=5,
            init_snake_len=[3, 3, 3, 3],
        )
        env = BattleSnakeGame(cfg=gc)
        actions = (UP, RIGHT, DOWN, LEFT)
        env.render()
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 4)
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 1)
        outcome = env.get_cum_rewards()
        self.assertEqual(1, outcome[0])
        for i in range(1, 4):
            self.assertEqual(-1, outcome[i])
        env.close()

    def test_self_collision(self):
        snake_spawns = {0: [[2, 0]], 1: [[0, 2]], 2: [[2, 4]], 3: [[4, 2]]}
        food_pos = [[2, 1]]
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=4,
            w=5,
            h=5,
            init_snake_len=[3, 3, 3, 3],
            all_actions_legal=True,
        )
        env = BattleSnakeGame(cfg=gc)
        actions = (UP, RIGHT, DOWN, LEFT)
        env.render()
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 4)
        actions = (DOWN, LEFT, UP, RIGHT)
        env.step(actions)
        env.render()
        self.assertEqual(env.num_players_alive(), 0)
        env.close()

    def test_collision_own_tail(self):
        snake_spawns = {0: [[0, 0]]}
        food_pos = [[0, 1], [1, 1]]
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=1,
            w=2,
            h=2,
            init_snake_len=[3],
            all_actions_legal=True,
        )
        env = BattleSnakeGame(cfg=gc)
        env.step((RIGHT,))
        env.render()
        env.step((UP,))
        env.render()
        env.step((LEFT,))
        env.render()
        self.assertEqual(env.num_players_alive(), 1)
        env.step((DOWN,))
        env.render()
        self.assertEqual(env.num_players_alive(), 0)
        env.close()

    def test_chase_own_tail(self):
        snake_spawns = {0: [[0, 0]]}
        food_pos = [[0, 1]]
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=1,
            w=2,
            h=2,
            init_snake_len=[3],
        )
        env = BattleSnakeGame(cfg=gc)
        env.step((RIGHT,))
        env.render()
        env.step((UP,))
        env.render()
        env.step((LEFT,))
        env.render()
        self.assertEqual(env.num_players_alive(), 1)
        env.step((DOWN,))
        env.render()
        self.assertEqual(env.num_players_alive(), 1)
        env.step((RIGHT,))
        env.render()
        self.assertEqual(env.num_players_alive(), 1)
        env.close()

    def test_food_spawn_prob(self):
        snake_spawns = {0: [[0, 0]]}
        food_pos = []
        num_food = 0
        num_episodes = 1000
        num_steps = 9
        spawn_prob = 25
        for i in range(num_episodes):
            gc = BattleSnakeConfig(
                init_snake_pos=snake_spawns,
                init_food_pos=food_pos,
                min_food=0,
                food_spawn_chance=spawn_prob,
                num_players=1,
                w=10,
                h=10,
                init_snake_len=[3],
                all_actions_legal=True,
            )
            env = BattleSnakeGame(cfg=gc)
            actions = (UP,)
            for _ in range(num_steps):
                env.step(actions)
            # count food:
            arr = ct.create_string_buffer(env.cfg.w * env.cfg.h * 3)
            CPP_LIB.lib.str_cpp(env.state_p, arr)
            str_repr = arr.value.decode("utf-8")
            num_food += str_repr.count('@')
            env.close()
        ratio = (num_food * 100) / (num_episodes * num_steps)
        self.assertAlmostEqual(ratio, spawn_prob, delta=2.5)

    def test_copy(self):
        snake_spawns = {0: [[0, 1]], 1: [[1, 0]], 2: [[1, 2]], 3: [[2, 1]]}
        food_pos = [[1, 1]]
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=4,
            w=3,
            h=3,
            init_snake_len=[3, 3, 3, 3],
            all_actions_legal=True,
        )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        actions = (LEFT, DOWN, UP, RIGHT)
        cpy = env.get_copy()
        cpy.step(actions)
        cpy.render()
        self.assertEqual(cpy.num_players_alive(), 0)
        self.assertEqual(env.num_players_alive(), 4)
        env.step((DOWN, RIGHT, LEFT, UP))
        self.assertEqual(env.num_players_alive(), 4)
        env.render()

    def test_equals_fixed(self):
        snake_spawns = {0: [[0, 1]], 1: [[1, 0]], 2: [[1, 2]], 3: [[2, 1]]}
        food_pos = [[1, 1]]
        gc = BattleSnakeConfig(
            init_snake_pos=snake_spawns,
            init_food_pos=food_pos,
            min_food=0,
            food_spawn_chance=0,
            num_players=4,
            w=3,
            h=3,
            init_snake_len=[3, 3, 3, 3],
            all_actions_legal=True,
        )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        self.assertTrue(env == env)
        cpy = env.get_copy()
        self.assertTrue(env == cpy)
        self.assertTrue(cpy == cpy)
        actions = (LEFT, DOWN, UP, RIGHT)
        cpy.step(actions)
        self.assertTrue(cpy == cpy)
        self.assertFalse(env == cpy)

    def test_equals_random(self):
        gc = BattleSnakeConfig(num_players=8, all_actions_legal=True)
        env = BattleSnakeGame(cfg=gc)
        cpy = env.get_copy()
        env2 = BattleSnakeGame(cfg=gc)
        env.render()
        env2.render()
        self.assertFalse(env == env2)  # this may fail due to randomness
        env.step(tuple([0 for _ in range(8)]))
        self.assertFalse(env == env2)
        self.assertFalse(env == cpy)
        env.close()
        env2.close()
        cpy.close()

    def test_all_actions_legal(self):
        snake_spawns = {0: [[0, 1]], 1: [[1, 0]], 2: [[1, 2]], 3: [[2, 1]]}
        food_pos = [[1, 1]]
        gc = BattleSnakeConfig(
            w=3,
            h=3,
            num_players=4,
            min_food=0,
            food_spawn_chance=0,
            init_snake_pos=snake_spawns,
            init_snake_len=[3, 3, 3, 3],
            init_food_pos=food_pos,
            all_actions_legal=True
        )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        for i in range(4):
            self.assertEqual(len(env.available_actions(i)), 4)
        self.assertEqual(len(env.available_joint_actions()), 4 ** 4)
        env.close()

    def test_state_info(self):
        gc = BattleSnakeConfig(num_players=2)
        env = BattleSnakeGame(cfg=gc)
        env.render()
        lengths = env.player_lengths()
        self.assertEqual(3, env.num_food())
        for i in range(env.num_players):
            self.assertEqual(3, lengths[i])
        fp = env.food_pos()
        print(fp)
        self.assertEqual(env.num_food(), len(fp))
        self.assertEqual(2, fp.shape[1])
        for i in range(env.num_players):
            self.assertEqual(1, len(env.player_pos(i)))
        self.assertEqual(env.num_players, len(env.all_player_pos()))
        print(env.all_player_pos())
        self.assertEqual(0, env.turns_played)

        env.step((UP, UP))
        env.render()
        for i in range(env.num_players):
            self.assertEqual(2, len(env.player_pos(i)))
        print(env.all_player_pos())
        self.assertEqual(1, env.turns_played)
        env.close()

    def test_reset_copy_equals(self):
        snake_spawns = {0: [[1, 2], [1, 1]], 1: [[0, 0]]}
        init_food = [[3, 3]]
        gc = BattleSnakeConfig(w=11, h=11, num_players=2, init_snake_pos=snake_spawns, init_food_pos=init_food,
                               init_turns_played=3, init_snake_len=[3, 3], )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        cpy = env.get_copy()
        self.assertTrue(env == cpy)
        for _ in range(5):
            env.step((UP, UP))
            env.render()
            self.assertTrue(env != cpy)
        env.reset()
        env.render()
        self.assertTrue(env == cpy)
        env.close()
        cpy.close()

    def test_init_food_spawning(self):
        gc = BattleSnakeConfig(w=11, h=11, num_players=4, init_food_pos=[], min_food=0, food_spawn_chance=100)
        env = BattleSnakeGame(cfg=gc)
        env.render()
        self.assertEqual(0, env.num_food())
        env.step((UP, UP, UP, UP))
        env.render()
        self.assertEqual(1, env.num_food())
        env.close()

    def test_max_health(self):
        snake_spawns = {0: [[0, 0]], 1: [[1, 0]]}
        init_food = [[0, 1]]
        gc = BattleSnakeConfig(w=11, h=11, num_players=2, init_snake_pos=snake_spawns, init_food_pos=init_food,
                               init_snake_health=[50, 50], max_snake_health=[200, 200], min_food=0, food_spawn_chance=0,
                               init_snake_len=[3, 3], )
        env = BattleSnakeGame(cfg=gc)
        env.render()
        self.assertEqual(50, env.player_healths()[0])
        self.assertEqual(50, env.player_healths()[1])

        env.step((UP, UP))
        env.render()
        self.assertEqual(200, env.player_healths()[0])
        self.assertEqual(49, env.player_healths()[1])
        env.close()

    def test_3x3_random_spawning(self):
        for num_snakes in range(1, 9):
            CPP_LIB.num_players = None
            gc = BattleSnakeConfig(w=3, h=3, num_players=num_snakes)
            env = BattleSnakeGame(gc)
            env.render()
            self.assertEqual(1, env.num_food())
            env.close()

    def test_spawning_is_not_seeded(self):
        game_conf = BattleSnakeConfig(w=7, h=7, num_players=3)
        game_list = []
        num_envs = 10
        for _ in range(num_envs):
            game = BattleSnakeGame(game_conf)
            game_list.append(game)
        for game in game_list:
            game.render()
        self.assertFalse(all(cur_game == game_list[0] for cur_game in game_list))
        for game in game_list:
            game.close()

    def test_copy_is_not_seeded(self):
        game_conf = BattleSnakeConfig(w=7, h=7, num_players=3, food_spawn_chance=100)
        game_list = []
        num_envs = 10
        game = BattleSnakeGame(game_conf)
        for _ in range(num_envs):
            new_game = game.get_copy()
            game_list.append(new_game)
        actions = (UP, UP, UP)
        for game in game_list:
            game.step(actions)
            game.render()
        self.assertFalse(all(cur_game == game_list[0] for cur_game in game_list))
        for game in game_list:
            game.close()

    def test_up_and_down(self):
        init_snake_pos = {0: [[1, 0]]}
        init_food_pos = []
        game_conf = BattleSnakeConfig(w=7, h=7, num_players=1, init_food_pos=init_food_pos,
                                      init_snake_pos=init_snake_pos,
                                      min_food=0, food_spawn_chance=0, init_snake_len=[3], all_actions_legal=True)
        game = BattleSnakeGame(game_conf)
        game.render()

        game.step((UP,))
        game.render()
        self.assertEqual(1, game.num_players_alive())

        game.step((DOWN,))
        game.render()
        self.assertEqual(0, game.num_players_alive())
        game.close()

    def test_legal_action_after_food_consumption(self):
        init_snake_pos = {0: [[3, 0], [2, 0], [1, 0]], 1: [[0, 0], [0, 1], [0, 2]]}
        init_food_pos = [[4, 0]]
        game_conf = BattleSnakeConfig(w=7, h=7, num_players=2, init_food_pos=init_food_pos,
                                      init_snake_pos=init_snake_pos,
                                      min_food=0, food_spawn_chance=0, init_snake_len=[3, 3])
        game = BattleSnakeGame(game_conf)
        game.render()

        game.step((RIGHT, RIGHT))
        game.render()
        print(game.available_actions(1))
        self.assertEqual(1, len(game.available_actions(1)))
        self.assertEqual(2, game.num_players_alive())

        game.close()

    def test_special_situation(self):
        init_snake_pos = {0: [[5, 9], [5, 9], [5, 9]], 1: [[0, 0], [0, 0], [0, 0]]}
        init_food_pos = []
        game_conf = BattleSnakeConfig(w=11, h=11, num_players=2, init_food_pos=init_food_pos,
                                      init_snake_pos=init_snake_pos,
                                      min_food=0, food_spawn_chance=0, init_snake_len=[3, 3], all_actions_legal=True)
        game = BattleSnakeGame(game_conf)
        game.render()
        self.assertEqual(2, game.num_players_alive())
        game.step((UP, UP))

        game.render()

        game.step((DOWN, DOWN))
        self.assertEqual(0, game.num_players_alive())
        game.render()
        game.close()

    def test_area_control_simple(self):
        init_snake_pos = {0: [[0, 0]], 1: [[2, 2]]}
        init_food_pos = [[0, 1], [1, 1]]
        game_conf = BattleSnakeConfig(w=3, h=3, num_players=2, init_food_pos=init_food_pos,
                                      init_snake_pos=init_snake_pos,
                                      min_food=0, food_spawn_chance=0, init_snake_len=[3, 3], )
        game = BattleSnakeGame(game_conf)
        game.render()
        start_time = time.time()
        r = game.area_control()
        ac, fd, td, rt, rf = r["area_control"], r["food_distance"], r["tail_distance"], r["tail_reachable"], \
            r["food_reachable"]
        print(f"{time.time() - start_time=}")
        self.assertEqual(2, len(ac))
        self.assertEqual(2, ac[0])
        self.assertEqual(2, ac[1])
        self.assertEqual(1, fd[0])
        self.assertEqual(6, fd[1])
        game.close()

    def test_are_control_complicated(self):
        init_snake_pos = {0: [[2, 2], [2, 1], [2, 0]], 1: [[3, 3], [3, 2], [3, 1]]}
        init_food_pos = [[2, 3]]
        game_conf = BattleSnakeConfig(w=5, h=5, num_players=2, init_food_pos=init_food_pos,
                                      init_snake_pos=init_snake_pos,
                                      min_food=0, food_spawn_chance=0, init_snake_len=[3, 3], )
        game = BattleSnakeGame(game_conf)
        game.render()
        print(game.player_pos(0))
        print(game.player_pos(1))
        r = game.area_control()
        ac, fd, td, rt, rf = r["area_control"], r["food_distance"], r["tail_distance"], r["tail_reachable"], \
            r["food_reachable"]
        print(ac)
        print(fd)
        self.assertEqual(10, fd[0])
        game.close()

    def test_are_control_different_len(self):
        init_snake_pos = {0: [[2, 2], [2, 1], [2, 0], [1, 0]], 1: [[3, 3], [3, 2], [3, 1]]}
        init_food_pos = [[0, 2], [4, 3]]
        init_snake_len = [4, 3]
        game_conf = BattleSnakeConfig(w=5, h=5, num_players=2, init_food_pos=init_food_pos,
                                      init_snake_pos=init_snake_pos,
                                      min_food=0, food_spawn_chance=0, init_snake_len=init_snake_len)
        game = BattleSnakeGame(game_conf)
        game.render()
        print(game.player_pos(0))
        print(game.player_pos(1))
        r = game.area_control()
        ac, fd, td, rt, rf = r["area_control"], r["food_distance"], r["tail_distance"], r["tail_reachable"], \
            r["food_reachable"]
        print(ac)
        print(fd)
        self.assertEqual(2, fd[0])
        self.assertEqual(1, fd[1])
        game.close()

    def test_area_control_efficiency(self):
        gc = BattleSnakeConfig(w=11, h=11, num_players=4)
        game = BattleSnakeGame(gc)
        start_time = time.time()
        ac = None
        for _ in range(50000):
            r = game.area_control()
            ac, fd, td, rt, rf = r["area_control"], r["food_distance"], r["tail_distance"], r["tail_reachable"], \
                r["food_reachable"]
        game.render()
        print(ac)
        print(f"{time.time() - start_time}")
        game.close()

    def test_tail_distance(self):
        init_snake_pos = {0: [[0, 1], [0, 0], [1, 0]], 1: [[4, 2], [4, 3], [4, 4]]}
        init_food_pos = []
        init_snake_len = [3, 3]
        game_conf = BattleSnakeConfig(w=5, h=5, num_players=2, init_food_pos=init_food_pos,
                                      init_snake_pos=init_snake_pos,
                                      min_food=0, food_spawn_chance=0, init_snake_len=init_snake_len)
        game = BattleSnakeGame(game_conf)
        game.render()
        r = game.area_control()
        ac, fd, td, rt, rf = r["area_control"], r["food_distance"], r["tail_distance"], r["tail_reachable"], \
            r["food_reachable"]
        self.assertEqual(2, td[0])
        self.assertEqual(4, td[1])

    def test_are_control_different_len_weights(self):
        init_snake_pos = {0: [[2, 2], [2, 1], [2, 0], [1, 0]], 1: [[3, 3], [3, 2], [3, 1]]}
        init_food_pos = [[0, 2], [4, 3]]
        init_snake_len = [4, 3]
        game_conf = BattleSnakeConfig(w=5, h=5, num_players=2, init_food_pos=init_food_pos,
                                      init_snake_pos=init_snake_pos,
                                      min_food=0, food_spawn_chance=0, init_snake_len=init_snake_len)
        game = BattleSnakeGame(game_conf)
        game.render()
        print(game.player_pos(0))
        print(game.player_pos(1))
        r = game.area_control(food_weight=2.0, weight=0.0)
        ac, fd, td, rt, rf = r["area_control"], r["food_distance"], r["tail_distance"], r["tail_reachable"], \
            r["food_reachable"]
        print(ac)
        print(fd)
        self.assertEqual(2, ac[0])
        self.assertEqual(2, ac[1])
        self.assertEqual(2, fd[0])
        self.assertEqual(1, fd[1])
        game.close()

    def test_area_control_choke(self):
        gc = perform_choke_2_player(centered=False, fully_connected=False)
        gc.all_actions_legal = True
        game = BattleSnakeGame(gc)
        ac_dict_0 = game.area_control()
        ac_0 = ac_dict_0["area_control"]
        game.step((UP, UP))
        ac_dict_1 = game.area_control()
        ac_1 = ac_dict_1["area_control"]
        game.step((UP, UP))
        ac_dict_2 = game.area_control()
        ac_2 = ac_dict_2["area_control"]
        self.assertGreater(ac_2[0], ac_1[0])
        self.assertGreater(ac_1[0], ac_0[0])

    def test_init_players_dead(self):
        init_players_alive = [True, False]
        game_conf = BattleSnakeConfig(w=5, h=5, num_players=2, init_snakes_alive=init_players_alive)
        game = BattleSnakeGame(game_conf)
        game.render()
        self.assertEqual(1, game.num_players_alive())
        self.assertTrue(game.is_player_alive(0))
        self.assertFalse(game.is_player_alive(1))
        self.assertTrue(game.is_terminal())

    def test_init_players_dead_in_game(self):
        init_snake_pos = {0: [[2, 2], [2, 1], [2, 0], [1, 0]], 1: [[3, 3], [3, 2], [3, 1]], 2: [[3, 3], [3, 2]]}
        init_food_pos = [[0, 2], [4, 3]]
        init_snake_len = [4, 3, 10]
        init_players_alive = [True, False, True]
        game_conf = BattleSnakeConfig(w=5, h=5, num_players=3, init_food_pos=init_food_pos,
                                      init_snake_pos=init_snake_pos, init_snakes_alive=init_players_alive,
                                      min_food=0, food_spawn_chance=0, init_snake_len=init_snake_len)
        game = BattleSnakeGame(game_conf)
        game.render()
        self.assertEqual(2, game.num_players_alive())
        self.assertTrue(game.is_player_alive(0))
        self.assertFalse(game.is_player_alive(1))
        self.assertTrue(game.is_player_alive(2))
        self.assertFalse(game.is_terminal())
        game.step((UP, LEFT))
        game.render()
        self.assertEqual(1, game.num_players_alive())
        self.assertFalse(game.is_player_alive(0))
        self.assertFalse(game.is_player_alive(1))
        self.assertTrue(game.is_player_alive(2))
        self.assertTrue(game.is_terminal())

    def test_kill_order_choke1(self):
        snake_spawns = {0: [[1, 2], [1, 1], [1, 0]], 1: [[2, 2], [2, 1], [2, 0]]}
        gc = BattleSnakeConfig(
            w=3,
            h=3,
            num_players=2,
            min_food=0,
            food_spawn_chance=0,
            init_snake_pos=snake_spawns,
            init_food_pos=[],
            init_snake_len=[3, 3],
            all_actions_legal=True,
        )
        game = get_game_from_config(gc)
        game.render()
        self.assertEqual(2, game.num_players_alive())

        game.step((DOWN, LEFT))
        game.render()
        self.assertEqual(0, game.num_players_alive())

    def test_kill_order_choke2(self):
        snake_spawns = {1: [[1, 2], [1, 1], [1, 0]], 0: [[2, 2], [2, 1], [2, 0]]}
        gc = BattleSnakeConfig(
            w=3,
            h=3,
            num_players=2,
            min_food=0,
            food_spawn_chance=0,
            init_snake_pos=snake_spawns,
            init_food_pos=[],
            init_snake_len=[3, 3],
            all_actions_legal=True,
        )
        game = get_game_from_config(gc)
        game.render()
        self.assertEqual(2, game.num_players_alive())

        game.step((LEFT, DOWN))
        game.render()
        self.assertEqual(0, game.num_players_alive())
