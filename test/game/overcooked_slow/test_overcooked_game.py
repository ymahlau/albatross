import random
import unittest

from src.game.initialization import get_game_from_config
from src.game.overcooked_slow.layouts import CrampedRoomOvercookedConfig, AsymmetricAdvantageOvercookedConfig, \
    CoordinationRingOvercookedConfig, ForcedCoordinationOvercookedConfig, CounterCircuitOvercookedConfig


class TestOvercookedGame(unittest.TestCase):
    def test_overcooked_cramped(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game_cfg.horizon = 5
        game = get_game_from_config(game_cfg)
        game.render()

        self.assertFalse(game.is_terminal())
        self.assertEqual(2, game.num_players_at_turn())
        self.assertEqual(2, game.num_players_alive())

        obs, _, _ = game.get_obs()
        print(obs.shape)
        print(game.get_obs_shape())

        for _ in range(5):
            game.step(random.choice(game.available_joint_actions()))
            game.render()
        self.assertTrue(game.is_terminal())
        self.assertEqual(0, game.num_players_at_turn())
        self.assertEqual(0, game.num_players_alive())

    def test_asymmetric_advantage(self):
        game_cfg = AsymmetricAdvantageOvercookedConfig()
        game_cfg.horizon = 5
        game = get_game_from_config(game_cfg)
        game.render()

        self.assertFalse(game.is_terminal())
        self.assertEqual(2, game.num_players_at_turn())
        self.assertEqual(2, game.num_players_alive())

        obs, _, _ = game.get_obs()
        print(obs.shape)
        print(game.get_obs_shape())

        for _ in range(5):
            game.step(random.choice(game.available_joint_actions()))
            game.render()
        self.assertTrue(game.is_terminal())
        self.assertEqual(0, game.num_players_at_turn())
        self.assertEqual(0, game.num_players_alive())

    def test_coordination_ring(self):
        game_cfg = CoordinationRingOvercookedConfig()
        game_cfg.horizon = 5
        game = get_game_from_config(game_cfg)
        game.render()

        self.assertFalse(game.is_terminal())
        self.assertEqual(2, game.num_players_at_turn())
        self.assertEqual(2, game.num_players_alive())

        obs, _, _ = game.get_obs()
        print(obs.shape)
        print(game.get_obs_shape())

        for _ in range(5):
            game.step(random.choice(game.available_joint_actions()))
            game.render()
        self.assertTrue(game.is_terminal())
        self.assertEqual(0, game.num_players_at_turn())
        self.assertEqual(0, game.num_players_alive())

    def test_forced_coordination(self):
        game_cfg = ForcedCoordinationOvercookedConfig()
        game_cfg.horizon = 5
        game = get_game_from_config(game_cfg)
        game.render()

        self.assertFalse(game.is_terminal())
        self.assertEqual(2, game.num_players_at_turn())
        self.assertEqual(2, game.num_players_alive())

        obs, _, _ = game.get_obs()
        print(obs.shape)
        print(game.get_obs_shape())

        for _ in range(5):
            game.step(random.choice(game.available_joint_actions()))
            game.render()
        self.assertTrue(game.is_terminal())
        self.assertEqual(0, game.num_players_at_turn())
        self.assertEqual(0, game.num_players_alive())

    def test_counter_circuit(self):
        game_cfg = CounterCircuitOvercookedConfig()
        game_cfg.horizon = 5
        game = get_game_from_config(game_cfg)
        game.render()

        self.assertFalse(game.is_terminal())
        self.assertEqual(2, game.num_players_at_turn())
        self.assertEqual(2, game.num_players_alive())

        obs, _, _ = game.get_obs()
        print(obs.shape)
        print(game.get_obs_shape())

        for _ in range(5):
            game.step(random.choice(game.available_joint_actions()))
            game.render()
        self.assertTrue(game.is_terminal())
        self.assertEqual(0, game.num_players_at_turn())
        self.assertEqual(0, game.num_players_alive())

    def test_orientation_mechanic(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game = get_game_from_config(game_cfg)
        game.render()

        game.step((0, 0))
        game.render()
        game.step((1, 1))
        game.render()
        game.step((5, 5))
        game.render()
        game.step((0, 0))
        game.render()

    def test_hardcoded_start_state(self):
        game_cfgs = [
            CrampedRoomOvercookedConfig(),
            AsymmetricAdvantageOvercookedConfig(),
            CoordinationRingOvercookedConfig(),
            ForcedCoordinationOvercookedConfig(),
            CounterCircuitOvercookedConfig()
        ]
        for game_cfg in game_cfgs:
            game = get_game_from_config(game_cfg)
            for _ in range(10):
                game2 = get_game_from_config(game_cfg)
                self.assertEqual(game, game2)
                game2.reset()
                self.assertEqual(game, game2)

    def test_reward(self):
        game_cfg = CrampedRoomOvercookedConfig(mep_reproduction_setting=True)
        game = get_game_from_config(game_cfg)
        game.render()

        reward, _, _ = game.step((0, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((3, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 3))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((2, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((0, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 4))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((3, 3))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((5, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        for _ in range(13):
            reward, _, _ = game.step((4, 4))
            game.render()
            print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 1))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 3))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 1))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 1))
        game.render()
        print(f"{reward}\n######################################")

        # This would test putting a soup on a counter
        # reward, _, _ = game.step((4, 2))
        # game.render()
        # print(f"{reward}\n######################################")
        #
        # reward, _, _ = game.step((4, 5))
        # game.render()
        # print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        print(f"{game.get_cum_rewards()=}")
        print(f"{game.turns_played=}")

    def test_cook_single_onion(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game = get_game_from_config(game_cfg)
        game.render()

        reward, _, _ = game.step((1, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((0, 3))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 0))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((4, 5))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((2, 2))
        game.render()
        print(f"{reward}\n######################################")

        reward, _, _ = game.step((0, 4))
        game.render()
        print(f"{reward}\n######################################")
        print(game.available_actions(0))
        self.assertEqual(5, len(game.available_actions(0)))

    def test_horizon(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game = get_game_from_config(game_cfg)

        while not game.is_terminal():
            ja = random.choice(game.available_joint_actions())
            game.step(ja)
            print('#################')
            game.render()
            print(game.turns_played)
