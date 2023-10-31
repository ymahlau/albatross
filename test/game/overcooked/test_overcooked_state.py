import unittest

from overcooked_ai_py.mdp.overcooked_mdp import SoupState

from src.game.overcooked.layouts import CrampedRoomOvercookedConfig
from src.game.overcooked.movement import cramped_at_cook_start, cramped_before_cook_soup_ready, \
    cramped_at_cook_soup_ready
from src.game.overcooked.overcooked import OvercookedGame
import overcooked_ai_py

class TestOvercookedState(unittest.TestCase):
    def test_player_position(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game = OvercookedGame(game_cfg)
        game.render()

        start_pos = game.get_player_positions()
        print(start_pos)
        self.assertEqual(1, start_pos[0][0])
        self.assertEqual(2, start_pos[0][1])
        self.assertEqual(3, start_pos[1][0])
        self.assertEqual(1, start_pos[1][1])

        game.step((1, 1))
        game.render()
        print(game.get_player_positions())

    def test_player_orientation(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game = OvercookedGame(game_cfg)
        game.render()

        start_orientation = game.get_player_orientations()
        print(start_orientation)
        self.assertEqual(0, start_orientation[0])
        self.assertEqual(0, start_orientation[1])

        game.step((0, 0))
        game.render()
        print(game.get_player_orientations())

        game.step((1, 1))
        game.render()
        print(game.get_player_orientations())

        game.step((2, 2))
        game.render()
        print(game.get_player_orientations())

        game.step((3, 3))
        game.render()
        print(game.get_player_orientations())

    def test_player_held_item(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game = OvercookedGame(game_cfg)
        game.step((0, 0))
        game.step((3, 3))
        game.step((5, 5))
        game.render()

        start_items = game.get_player_held_item()
        print(start_items)

    def test_pot_state_start(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game = OvercookedGame(game_cfg)
        game.render()

        start_pots = game.get_pot_states()
        print(start_pots)
        self.assertEqual(0, start_pots[0])

    def test_pot_state_cooking(self):
        game = cramped_at_cook_start()
        game.render()
        pot_state = game.get_pot_states()
        print(pot_state)
        self.assertEqual(3, pot_state[0])

        game.step((5, 5))
        game.render()
        pot_state = game.get_pot_states()
        print(pot_state)
        self.assertEqual(23, pot_state[0])

        game = cramped_before_cook_soup_ready()
        game.render()
        pot_state = game.get_pot_states()
        print(pot_state)
        self.assertEqual(5, pot_state[0])

        game.step((4, 0))
        game.render()
        pot_state = game.get_pot_states()
        print(pot_state)
        self.assertEqual(4, pot_state[0])

    def test_start_counter_states(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game = OvercookedGame(game_cfg)
        game.render()

        start_counters = game.get_counter_states()
        print(start_counters)
        for s in start_counters:
            self.assertEqual(3, s)

        # pickup dish and drop
        game.step((1, 1))
        game.step((5, 5))
        game.step((3, 3))
        game.step((5, 5))

        game.render()
        counters = game.get_counter_states()
        print(counters)

    def test_soup_ready(self):
        game = cramped_at_cook_soup_ready()
        game.step((5, 5))
        game.render()

        print(game.get_player_held_item())

        game.step((1, 1))
        game.step((2, 2))
        game.step((1, 1))
        game.render()
        rewards, _, _ = game.step((5, 5))
        game.render()
        print(rewards)
        print(game.get_counter_states())

    def test_state(self):
        game_cfg = CrampedRoomOvercookedConfig()
        game = OvercookedGame(game_cfg)
        game.render()
        print(game.get_state())

        game = cramped_at_cook_soup_ready()
        game.render()
        print(game.get_state())
